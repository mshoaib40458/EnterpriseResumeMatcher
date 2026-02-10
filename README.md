Enterprise Talent Intelligence Agent
AI Resume Screener + Job Description Matching System

An LLM-powered resume intelligence application that extracts structured candidate information from resumes and evaluates candidate–job fit using Groq LLaMA models.
This project demonstrates how modern AI recruitment assistants can be built using LangChain pipelines, structured output parsing, and resilient LLM orchestration.

Features
Resume Parsing
Supports:
PDF resumes
DOCX resumes
TXT resumes

Extracted information:
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

Role Matching Engine
The agent analyzes resumes against a Job Description and produces:
Role Match Score (0–100)
Match Justification
Structured candidate profile JSON

Tech Stack
LLM
Groq API
LLaMA-3.1-8B-Instant
LangChain

Backend
Python
Pydantic
Tenacity
dotenv
Document Processing
PyPDFLoader
Docx2txtLoader
TextLoader

Frontend
Streamlit

Architecture
Resume → Loader → Text Extraction → Prompt → LLM → JSON Parser → UI Dashboard
