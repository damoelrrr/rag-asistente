"""Ingesta: extracción de texto desde PDF, DOCX, TXT/MD y HTML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import split_text

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


@dataclass
class Page:
    text: str
    source: str  # nombre del archivo (se usa para citar)
    page: int | None = None  # solo PDFs


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int


def _read_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [Page(p.extract_text() or "", path.name, i + 1) for i, p in enumerate(reader.pages)]


def _read_docx(path: Path) -> list[Page]:
    from docx import Document

    doc = Document(str(path))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [Page(text, path.name)]


def _read_html(path: Path) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [Page(text, path.name)]


def _read_text(path: Path) -> list[Page]:
    return [Page(path.read_text(encoding="utf-8", errors="ignore"), path.name)]


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".html": _read_html,
    ".htm": _read_html,
    ".txt": _read_text,
    ".md": _read_text,
}


def load_corpus(corpus_dir: Path) -> list[Page]:
    files = sorted(p for p in Path(corpus_dir).rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No hay documentos soportados en {corpus_dir} ({sorted(SUPPORTED)})")
    pages: list[Page] = []
    for f in files:
        pages.extend(READERS[f.suffix.lower()](f))
    return pages


def build_chunks(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Fragmenta cada página/documento y asigna ids estables: <archivo>::<n>."""
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        for text in split_text(page.text, chunk_size, chunk_overlap):
            n = counters.get(page.source, 0)
            counters[page.source] = n + 1
            chunks.append(Chunk(f"{page.source}::{n}", text, page.source, page.page, n))
    return chunks
