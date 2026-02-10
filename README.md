🚀 Enterprise Talent Intelligence Agent
AI Resume Screener + Job Description Matching System
An LLM-powered Talent Intelligence system that extracts structured candidate information from resumes and evaluates candidate–job fit using Groq LLaMA models.
This project demonstrates how modern AI recruitment assistants can be built using LangChain pipelines, structured output parsing, and resilient LLM orchestration.

✨ Key Features
📄 Resume Parsing Engine
Supports multiple resume formats:
PDF resumes
DOCX resumes
TXT resumes

The system automatically extracts structured candidate information:
Name
Email
Phone
LinkedIn
Skills
Education
Experience
Projects
Certifications
Languages

🎯 Role Matching Engine
The Talent Intelligence Agent analyzes resumes against a Job Description (JD) and produces:
Role Match Score (0–100)
Match Justification

🧠 Tech Stack
LLM Layer
Groq API
LLaMA-3.1-8B-Instant
LangChain

Backend
Python
pydantic (structured validation)
Tenacity (retry handling)
dotenv (environment management)

Document Processing
PyPDFLoader
Docx2txtLoader
TextLoader
Structured Candidate Profile (JSON)

This enables AI-assisted resume screening and candidate ranking workflows.
