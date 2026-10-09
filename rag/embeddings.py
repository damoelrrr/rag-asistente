"""Modelo de embeddings (local, con sentence-transformers). Los textos NO salen a ningún servicio externo."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = _model(self.model_name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    # Interfaz que espera LangChain/Ragas (answer_relevancy usa embeddings)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def as_langchain_embeddings(embedder: Embedder):
    from langchain_core.embeddings import Embeddings

    class _Wrapped(Embeddings):
        def embed_documents(self, texts):  # type: ignore[override]
            return embedder.embed(texts)

        def embed_query(self, text):  # type: ignore[override]
            return embedder.embed([text])[0]

    return _Wrapped()
