"""Chunking recursivo con solapamiento (implementación propia, sin dependencias).

Criterio de corte: se intenta cortar por párrafo ("\n\n"), luego por línea ("\n"),
luego por frase (". "), luego por palabra (" ") y solo al final por caracteres.
Así los chunks respetan, en lo posible, los límites naturales del texto.
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _atomize(text: str, size: int, seps: list[str]) -> list[str]:
    """Divide el texto en piezas de a lo sumo `size` caracteres usando el separador más grande posible."""
    if len(text) <= size:
        return [text]
    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    raw = text.split(sep)
    parts = [p + sep for p in raw[:-1]] + [raw[-1]]
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        out.extend([part] if len(part) <= size else _atomize(part, size, rest))
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Une piezas hasta llenar `size`; el siguiente chunk arranca con ~`overlap` caracteres del anterior."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        if cur and cur_len + len(piece) > size:
            chunks.append("".join(cur).strip())
            while cur and (cur_len > overlap or cur_len + len(piece) > size):
                cur_len -= len(cur[0])
                cur.pop(0)
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    return _merge(_atomize(text, chunk_size, SEPARATORS), chunk_size, chunk_overlap)
