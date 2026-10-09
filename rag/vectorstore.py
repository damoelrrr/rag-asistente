"""Índice vectorial persistente con Chroma. Guarda texto + metadatos para poder citar la fuente."""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .ingest import Chunk


@dataclass
class Hit:
    n: int  # número de cita [n] que verá el LLM y el usuario
    text: str
    source: str
    page: int | None
    chunk_index: int
    distance: float  # distancia coseno (menor = más parecido)

    @property
    def similarity(self) -> float:
        return max(0.0, 1.0 - self.distance)


class VectorStore:
    def __init__(self, persist_dir, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = self.client.get_or_create_collection("corpus", metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection("corpus")
        self.col = self.client.get_or_create_collection("corpus", metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.col.add(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=self.embedder.embed([c.text for c in part]),
                metadatas=[
                    {"source": c.source, "page": c.page if c.page is not None else -1, "chunk_index": c.chunk_index}
                    for c in part
                ],
            )

    def search(self, query: str, k: int) -> list[Hit]:
        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
            page = meta.get("page", -1)
            hits.append(Hit(i + 1, doc, meta["source"], None if page == -1 else page, meta["chunk_index"], dist))
        return hits
