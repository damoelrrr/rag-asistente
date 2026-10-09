"""Pipeline RAG de extremo a extremo: índice -> reescritura -> recuperación -> generación."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings
from .embeddings import Embedder
from .ingest import build_chunks, load_corpus
from .llm import get_chat_model, text_of
from .prompts import CONDENSE_PROMPT, REFUSAL, build_messages
from .vectorstore import Hit, VectorStore


@dataclass
class Prepared:
    question: str
    standalone: str
    hits: list[Hit]  # los que superan el umbral de distancia
    raw_hits: list[Hit]  # todo lo recuperado (para evaluación)
    messages: list | None  # None => no hay contexto pertinente => rechazo sin llamar al LLM


@dataclass
class Answer:
    text: str
    standalone: str
    hits: list[Hit]
    raw_hits: list[Hit]
    refused: bool
    cited: list[int] = field(default_factory=list)
    seconds: float = 0.0


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(REFUSAL[:30].lower())


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.embedder = Embedder(self.s.embedding_model)
        self.store = VectorStore(self.s.index_dir, self.embedder)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_chat_model(self.s)
        return self._llm

    # ---------- Indexación ----------
    def ensure_index(self, rebuild: bool = False) -> int:
        if rebuild:
            self.store.reset()
        if self.store.count() == 0:
            pages = load_corpus(self.s.corpus_dir)
            chunks = build_chunks(pages, self.s.chunk_size, self.s.chunk_overlap)
            self.store.add(chunks)
        return self.store.count()

    # ---------- Conversación ----------
    def _history_tail(self, history: list[dict]) -> list[dict]:
        return history[-self.s.max_history_messages :] if history else []

    def rewrite(self, question: str, history: list[dict]) -> str:
        """Convierte una pregunta de seguimiento en una pregunta independiente usando el historial."""
        tail = self._history_tail(history)
        if not tail:
            return question
        hist = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}" for m in tail)
        out = text_of(self.llm.invoke(CONDENSE_PROMPT.format(history=hist, question=question)).content).strip()
        return out or question

    # ---------- Recuperación ----------
    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[Hit], list[Hit]]:
        raw = self.store.search(query, top_k or self.s.top_k)
        kept = [h for h in raw if h.distance <= self.s.max_distance]
        for i, h in enumerate(kept, 1):  # renumera [1..n] solo con los fragmentos usados
            h.n = i
        return kept, raw

    # ---------- Generación ----------
    def prepare(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> Prepared:
        history = history or []
        standalone = self.rewrite(question, history)
        hits, raw = self.retrieve(standalone, top_k)
        messages = build_messages(standalone, hits, self._history_tail(history)) if hits else None
        return Prepared(question, standalone, hits, raw, messages)

    def stream(self, prep: Prepared) -> Iterator[str]:
        if prep.messages is None:
            yield REFUSAL
            return
        for chunk in self.llm.stream(prep.messages):
            piece = text_of(chunk.content)
            if piece:
                yield piece

    def answer(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> Answer:
        t0 = time.time()
        prep = self.prepare(question, history, top_k)
        text = REFUSAL if prep.messages is None else text_of(self.llm.invoke(prep.messages).content).strip()
        return self.finish(prep, text, time.time() - t0)

    @staticmethod
    def finish(prep: Prepared, text: str, seconds: float = 0.0) -> Answer:
        refused = is_refusal(text)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)}) if not refused else []
        return Answer(text, prep.standalone, [] if refused else prep.hits, prep.raw_hits, refused, cited, seconds)
