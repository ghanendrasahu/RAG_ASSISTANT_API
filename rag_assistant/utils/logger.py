import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_assistant.utils.request_context import get_request_id

LOG_FILE = Path("logs.jsonl")


def log_event(event_type: str, payload: dict[str, Any], level: str = "INFO") -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event_type": event_type,
        "request_id": get_request_id() or None,
        "payload": payload,
    }
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_interaction(
    question: str,
    top_k: int,
    sources: list[dict[str, Any]],
    answer: str,
    latency_ms: float,
) -> None:
    log_event(
        "rag_query",
        {
            "question": question,
            "top_k": top_k,
            "source_count": len(sources),
            "sources": sources,
            "answer": answer,
            "latency_ms": latency_ms,
        },
    )
