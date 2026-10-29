"""Índice vectorial persistente con Chroma. Guarda texto + metadatos para poder citar la fuente."""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .ingest import Chunk
from .retrieval import LexicalIndex, has_content, semantic_query


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
        self._lexical: LexicalIndex | None = None
        self._lexical_count = -1

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection("corpus")
        self.col = self.client.get_or_create_collection("corpus", metadata={"hnsw:space": "cosine"})
        self._lexical = None

    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        self._lexical = None
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
        count = self.count()
        if count == 0 or k < 1:
            return []
        if self._lexical is None or self._lexical_count != count:
            corpus = self.col.get(include=["documents", "metadatas"])
            self._lexical = LexicalIndex(corpus["ids"], corpus["documents"], corpus["metadatas"])
            self._lexical_count = count

        # Buscar más candidatos permite descartar títulos sin reducir el contexto final a cero.
        candidate_count = min(max(k * 8, 32), count)
        vector = self.embedder.embed_query(semantic_query(query))
        res = self.col.query(
            query_embeddings=[vector],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )
        candidates = {}
        semantic_ids = []
        for key, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if has_content(doc):
                candidates[key] = (doc, meta, dist)
                semantic_ids.append(key)

        lexical_ids = self._lexical.search(query)
        missing = [key for key in lexical_ids[:candidate_count] if key not in candidates]
        if missing:
            extra = self.col.get(ids=missing, include=["embeddings"])
            for key, embedding in zip(extra["ids"], extra["embeddings"]):
                doc, meta = self._lexical.records[key]
                # Los embeddings se guardan normalizados: conservar la distancia coseno real
                # permite aplicar el mismo filtro de relevancia después de la fusión.
                distance = max(0.0, 1.0 - sum(float(a) * float(b) for a, b in zip(vector, embedding)))
                candidates[key] = (doc, meta, distance)

        # Reciprocal rank fusion: priorizar significado y reforzar términos exactos.
        scores = {key: 2.0 / (20 + rank) for rank, key in enumerate(semantic_ids, 1)}
        for rank, key in enumerate(lexical_ids, 1):
            if key in candidates:
                scores[key] = scores.get(key, 0.0) + 1.0 / (20 + rank)

        hits = []
        for i, key in enumerate(sorted(scores, key=lambda key: (-scores[key], key))[:k]):
            doc, meta, dist = candidates[key]
            page = meta.get("page", -1)
            hits.append(Hit(i + 1, doc, meta["source"], None if page == -1 else page, meta["chunk_index"], dist))
        return hits
