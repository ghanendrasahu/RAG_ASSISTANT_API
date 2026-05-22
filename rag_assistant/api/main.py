from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_assistant.core.rag_pipeline import run_rag_pipeline
from rag_assistant.services.vectorstore import ingest_pdf
from rag_assistant.utils.logger import log_event
from rag_assistant.utils.request_context import set_request_id

app = FastAPI(title="RAG API", version="1.1")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict


class IngestRequest(BaseModel):
    pdf_path: str = "data/sample.pdf"
    chunk_size: int = Field(default=700, ge=50)
    overlap: int = Field(default=150, ge=0)


@app.middleware("http")
async def request_trace_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    set_request_id(request_id)

    started = perf_counter()
    log_event(
        "request_started",
        {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
        },
    )

    response = await call_next(request)
    duration_ms = round((perf_counter() - started) * 1000, 2)

    response.headers["x-request-id"] = request_id
    log_event(
        "request_finished",
        {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_event(
        "unhandled_exception",
        {
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
        },
        level="ERROR",
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        result = ingest_pdf(
            pdf_path=request.pdf_path,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
        log_event("ingest_completed", result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = run_rag_pipeline(question=request.question, top_k=request.top_k)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "metadata": result["metadata"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.on_event("startup")
def startup_ingest_default_pdf() -> None:
    default_pdf = Path("data/sample.pdf")
    if default_pdf.exists():
        ingest_pdf(str(default_pdf))
        log_event("startup_ingest", {"source": str(default_pdf)})
