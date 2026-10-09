"""Configuración central. Todo se puede cambiar con variables de entorno (.env o secrets de la nube)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # .env solo en local; en la nube se usan los secrets de la plataforma
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))
    # --- Recuperación ---
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    # Distancia coseno máxima (0 = idéntico, 1 = ortogonal). Más alta = más permisivo.
    max_distance: float = field(default_factory=lambda: float(_env("MAX_DISTANCE", "0.75")))
    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    # --- LLM de generación ---
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.1")))
    max_history_messages: int = field(default_factory=lambda: int(_env("MAX_HISTORY_MESSAGES", "6")))
    # --- LLM juez (Ragas). Por defecto el mismo que el de generación ---
    judge_provider: str = field(default_factory=lambda: _env("JUDGE_PROVIDER", ""))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", ""))
    # --- Rutas ---
    corpus_dir: Path = field(default_factory=lambda: ROOT / _env("CORPUS_DIR", "data/corpus"))
    index_root: Path = field(default_factory=lambda: ROOT / _env("INDEX_ROOT", "data/index"))

    @property
    def index_dir(self) -> Path:
        """Cada combinación de parámetros tiene su propio índice (así se comparan iteraciones)."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.embedding_model.split("/")[-1]).strip("-")
        return self.index_root / f"cs{self.chunk_size}_ov{self.chunk_overlap}_{slug}"

    def default_model(self, provider: str) -> str:
        return {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini", "groq": "openai/gpt-oss-120b"}.get(provider, "")

