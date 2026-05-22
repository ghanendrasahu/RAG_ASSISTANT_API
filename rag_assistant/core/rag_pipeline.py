from time import perf_counter

from rag_assistant.core.generator import generate_from_sources
from rag_assistant.core.retriever import get_relevant_sources
from rag_assistant.utils.logger import log_event, log_interaction


def run_rag_pipeline(question: str, top_k: int = 3) -> dict:
    started = perf_counter()

    log_event("retrieval_started", {"question": question, "top_k": top_k})
    sources = get_relevant_sources(question=question, top_k=top_k)
    log_event("retrieval_finished", {"retrieved": len(sources)})

    answer = generate_from_sources(question=question, sources=sources)

    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    metadata = {
        "top_k": top_k,
        "source_count": len(sources),
        "latency_ms": elapsed_ms,
    }

    log_interaction(
        question=question,
        top_k=top_k,
        sources=sources,
        answer=answer,
        latency_ms=elapsed_ms,
    )

    return {
        "answer": answer,
        "sources": sources,
        "metadata": metadata,
    }
