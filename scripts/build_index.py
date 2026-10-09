"""Construye (o reconstruye) el índice vectorial:  python scripts/build_index.py [--rebuild]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rebuild", action="store_true", help="borra y vuelve a crear el índice")
ap.add_argument("--chunk-size", type=int)
ap.add_argument("--overlap", type=int)
a = ap.parse_args()

base = Settings()
s = Settings(chunk_size=a.chunk_size or base.chunk_size, chunk_overlap=a.overlap or base.chunk_overlap)
n = RAGPipeline(s).ensure_index(rebuild=a.rebuild)
print(f"Índice listo: {n} chunks en {s.index_dir}")
