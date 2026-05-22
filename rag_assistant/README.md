# RAG Assistant

This subproject is a FastAPI-based PDF RAG backend.

## Features

- ingest PDF files
- split extracted text into chunks
- store chunks in ChromaDB
- retrieve relevant chunks for a question
- optionally generate an answer using Groq

## Structure

- `api/` FastAPI routes
- `core/` pipeline orchestration
- `services/` embeddings, vector store, LLM integration
- `utils/` logging and request context

## Run

```powershell
cd "C:\Users\ghane\Documents\ALL PROJECTS\DATA SCIENCE\PRACTICAL"
.\rag_env\Scripts\Activate.ps1
pip install -r rag_assistant\requirements.txt
uvicorn rag_assistant.api.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`
