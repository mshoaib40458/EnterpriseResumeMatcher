import os
import json
import time
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from groq.types.chat import ChatCompletion

load_dotenv()

class ExtractedResumeData(BaseModel):
    """Core resume data structure."""
    name: str = Field(description="The full name of the candidate, or 'Not found'.")
    email: str = Field(description="The primary email address, or 'Not found'.")
    phone: str = Field(description="The contact phone number, or 'Not found'.")
    linkedin: str = Field(description="The LinkedIn profile URL, or 'Not found'.")
    skills: List[str] = Field(description="A list of technical and soft skills.")
    education: List[str] = Field(description="A list of degrees, institutions, and graduation years.")
    experience: List[str] = Field(description="A list of job roles, companies, and employment periods.")
    projects: List[str] = Field(description="A list of personal or professional projects.")
    certifications: List[str] = Field(description="A list of professional certifications.")
    languages: List[str] = Field(description="A list of spoken or programming languages.")

class AgentOutput(BaseModel):
    """The final output including core data and intelligence."""
    extracted_data: ExtractedResumeData
    role_match_score: int = Field(description="A score from 0 to 100 assessing fit for the provided Job Description.")
    match_justification: str = Field(description="A concise, 3-sentence justification for the score, citing specific skills/experience gaps or overlaps.")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("FATAL ERROR: GROQ_API_KEY environment variable is not set. Cannot initialize LLM.")
    st.stop()

try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

    parser = JsonOutputParser(pydantic_object=AgentOutput)
except Exception as e:
    st.error(f"LLM Initialization Failed: {e}")
    st.stop()


PROMPT_TEMPLATE = """
You are a highly specialized Enterprise Talent Intelligence Agent. Your task is to perform two critical actions:
1. **Extract** detailed resume information.
2. **Analyze** the candidate's fit against the provided Job Description (JD).

---
**JOB DESCRIPTION (JD):**
{job_description}
---

Your output must STRICTLY follow the JSON Schema provided below.

JSON Schema:
{format_instructions}

Rules:
- For missing fields in the resume, use an empty list for list fields (e.g., "skills": []) or "Not found" for string fields.
- The 'role_match_score' MUST be between 0 and 100.
- Output ONLY the JSON object. Do not include markdown outside of the JSON block or any conversational text.

Resume Text to Analyze:
{text}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["text", "job_description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


@retry(
    wait=wait_exponential(min=1, max=60), 
    stop=stop_after_attempt(5),         
    retry=retry_if_exception_type((ConnectionError, TimeoutError)), 
    reraise=True 
)
def invoke_llm_with_retry(payload: Dict[str, Any]) -> AgentOutput:
    """Invokes the LLM chain and automatically retries on transient errors."""
    start_time = time.time()
    

    chain = prompt | llm | parser
    
    
    response_data = chain.invoke(payload)
    
    end_time = time.time()
    
    
    st.session_state['llm_latency'] = f"{end_time - start_time:.2f} seconds"
    
    
    return AgentOutput.model_validate(response_data)



@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file) -> Optional[str]:
    """Extracts text from uploaded file and cleans up the temporary file."""
    temp_path = f"temp_{uploaded_file.name}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.lower().endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            return None
        
        docs = loader.load()
        text = " ".join([d.page_content for d in docs])
        return text
    
    except Exception as e:
        st.error(f"File Processing Error: {e}")
        return None
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



def main():
    st.set_page_config(
        page_title="Deloitte Talent Intelligence Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Configuration & Input ")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Resume", 
        type=["pdf", "docx", "txt"],
        help="Upload the candidate's resume for extraction."
    )
    
    
    job_description = st.sidebar.text_area(
        "Job Description (JD)",
        placeholder="Paste the target job description here for role matching...",
        height=300,
        help="The agent uses this text to generate the Role Match Score."
    )
    
    st.title("Enterprise Talent Intelligence Agent")
    st.markdown("### Resume & Role Fit Analyzer")
    st.markdown("---")

    if uploaded_file and job_description:
        
    
        st.session_state['llm_latency'] = "N/A"
        
        with st.spinner("Step 1/2: Extracting text from file..."):
            resume_text = extract_text_from_file(uploaded_file)
            
            if not resume_text:
                st.error("File processing failed. Please check the uploaded file.")
                return

        with st.spinner("Step 2/2: Invoking Resilient LLM Agent for analysis..."):
            try:
                payload = {"text": resume_text, "job_description": job_description}
                parsed_data: AgentOutput = invoke_llm_with_retry(payload)
                
               
                data_dict = parsed_data.model_dump()
                
                st.success(f"Analysis Complete for {data_dict['extracted_data']['name']}!")

            except Exception as e:
                error_message = str(e)
                st.error(
                    f"Agent Processing Failed: All 5 retries failed. This usually indicates a sustained Groq outage or an API key issue. "
                    f"Last reported error: {error_message}"
                )
                return

        col_score, col_name, col_latency = st.columns([1, 2, 1])

        col_name.header(data_dict['extracted_data']['name'] or "Candidate Not Found")
        col_name.markdown(f"**Email:** `{data_dict['extracted_data']['email']}` | **Phone:** `{data_dict['extracted_data']['phone']}`")
        
        col_score.metric(
            label="Role Match Score",
            value=f"{data_dict['role_match_score']}/100"
        )
        
        col_latency.metric(
            label="Agent Latency",
            value=st.session_state['llm_latency'],
            help="Time taken for the final successful LLM API call."
        )
        
        st.info(f"**Justification:** {data_dict['match_justification']}")
        st.markdown("---")


        tab1, tab2, tab3, tab4 = st.tabs([" Core Data", "💼 Experience & Education", "📜 Source Text", "💾 Raw JSON"])
        
        with tab1:
            st.subheader("Key Contact & Skills")
            col_contact, col_skills = st.columns(2)
            
            col_contact.write(f"**LinkedIn:** {data_dict['extracted_data']['linkedin']}")
            col_contact.write(f"**Certifications:** {', '.join(data_dict['extracted_data']['certifications']) or 'Not found'}")
            col_contact.write(f"**Languages:** {', '.join(data_dict['extracted_data']['languages']) or 'Not found'}")
            
            col_skills.subheader("Extracted Skills")
            col_skills.code('\n'.join(data_dict['extracted_data']['skills']) or 'Not found', language='text')

        with tab2:
            st.subheader("Experience Summary")
            st.json(data_dict['extracted_data']['experience'])
            st.subheader("Education Summary")
            st.json(data_dict['extracted_data']['education'])

        with tab3:
            st.subheader("Raw Extracted Resume Text")
            st.text_area("Source Text", resume_text, height=400, disabled=True)
            
        with tab4:
            st.subheader("Full JSON Output (for Integration)")
            st.json(data_dict)
            st.download_button(
                label="Download Full JSON",
                data=json.dumps(data_dict, indent=2),
                file_name=f"{data_dict['extracted_data']['name'].replace(' ', '_')}_analysis.json",
                mime="application/json"
            )

    elif uploaded_file and not job_description:
        st.warning("Please paste the **Job Description** in the sidebar to begin the analysis.")
    elif not uploaded_file and job_description:
        st.warning("Please upload a **Resume** file in the sidebar to begin the analysis.")
    else:
        st.info("Upload a resume and paste a job description in the sidebar to start the Talent Intelligence Agent.")


if __name__ == "__main__":
    
    if 'llm_latency' not in st.session_state:
        st.session_state['llm_latency'] = "N/A"
        
    main()