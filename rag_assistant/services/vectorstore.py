import re
from pathlib import Path
from typing import Any

import chromadb
import fitz

from rag_assistant.services.embeddings import get_embedding_function

DEFAULT_COLLECTION = "rag_collection"


class VectorStoreService:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

    @staticmethod
    def _split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                raise RuntimeError(f"PDF is password protected: {pdf_path}")

            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                text = re.sub(r"-\\n(\\w)", r"\\1", text)
                text = re.sub(r"\\n{3,}", "\\n\\n", text)
                text = re.sub(r"[ \\t]{2,}", " ", text).strip()
                if text:
                    pages.append(
                        {
                            "text": text,
                            "metadata": {"source": str(pdf_path), "page": page_number},
                        }
                    )
        return pages

    def ingest_pdf(self, pdf_path: str, chunk_size: int = 700, overlap: int = 150) -> dict[str, Any]:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages = self._extract_pages(path)
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for page in pages:
            page_chunks = self._split_words(page["text"], chunk_size=chunk_size, overlap=overlap)
            for chunk_idx, chunk in enumerate(page_chunks):
                documents.append(chunk)
                metadata = dict(page["metadata"])
                metadata["chunk_index"] = chunk_idx
                metadatas.append(metadata)

        if not documents:
            raise ValueError("No readable text extracted from PDF.")

        self.collection.delete(where={"source": str(path)})
        ids = [f"{path.name}-{i}" for i in range(len(documents))]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return {"source": str(path), "chunks_indexed": len(documents)}

    def retrieve_docs(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output: list[dict[str, Any]] = []
        for idx, text in enumerate(docs):
            distance = distances[idx] if idx < len(distances) else None
            output.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "text": text,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distance,
                    "score": (1.0 / (1.0 + distance)) if distance is not None else None,
                }
            )
        return output


_VECTOR_SERVICE: VectorStoreService | None = None


def get_vectorstore_service() -> VectorStoreService:
    global _VECTOR_SERVICE
    if _VECTOR_SERVICE is None:
        _VECTOR_SERVICE = VectorStoreService()
    return _VECTOR_SERVICE


def ingest_pdf(pdf_path: str, chunk_size: int = 700, overlap: int = 150) -> dict[str, Any]:
    return get_vectorstore_service().ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)


def retrieve_docs(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    return get_vectorstore_service().retrieve_docs(query, top_k=top_k)
