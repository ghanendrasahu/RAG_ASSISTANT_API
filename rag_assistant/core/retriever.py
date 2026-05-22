from rag_assistant.services.vectorstore import retrieve_docs


def get_relevant_sources(question: str, top_k: int = 3) -> list[dict]:
    return retrieve_docs(question, top_k=top_k)
