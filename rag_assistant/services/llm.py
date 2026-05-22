import os
from pathlib import Path
from typing import Iterable

from groq import Groq


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_prompt(question: str, context: str) -> str:
    return f"""
You are a strict RAG assistant.

Rules:
- Use only the supplied context.
- If the answer is not in context, reply exactly: Not found in document.
- Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def _fallback_answer(question: str, sources: Iterable[dict]) -> str:
    for source in sources:
        text = (source.get("text") or "").strip()
        if text:
            return f"LLM unavailable. Best retrieved context snippet: {text[:400]}"
    return "Not found in document."


def generate_answer(question: str, context: str, sources: list[dict]) -> str:
    _load_env_file()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_answer(question, sources)

    client = Groq(api_key=api_key)
    prompt = _build_prompt(question, context)

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "Not found in document."
