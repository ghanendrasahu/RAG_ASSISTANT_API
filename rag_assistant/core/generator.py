from rag_assistant.services.llm import generate_answer


def generate_from_sources(question: str, sources: list[dict]) -> str:
    context = "\n\n".join(source.get("text", "") for source in sources)
    return generate_answer(question=question, context=context, sources=sources)
