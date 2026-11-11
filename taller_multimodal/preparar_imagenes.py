"""Genera las imágenes del dominio a partir de los PDFs del corpus del tutor (data/corpus/).

Las imágenes son diapositivas, fórmulas, una figura y una tabla del material del curso
"Desarrollo de aplicaciones con IA", que es el dominio del asistente RAG.

Uso:
    python taller_multimodal/preparar_imagenes.py
"""
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = Path(__file__).resolve().parent / "imagenes"
LADO_MAYOR = 1600  # px: legible para el modelo y muy por debajo del límite de 20 MB por solicitud

# (archivo PDF, página 1-indexada, nombre de salida, recorte opcional en fracciones (x0, y0, x1, y1))
IMAGENES = [
    ("3 BM25_Score.pdf", 3, "01_bm25_formula.png", None),
    ("TF-IDF_Keyword_search.pdf", 24, "02_tfidf_formula.png", None),
    ("TF-IDF_Keyword_search.pdf", 18, "03_term_frequency_ejemplo.png", None),
    ("TF-IDF_Keyword_search.pdf", 5, "04_tecnicas_busqueda.png", None),
    ("2312.10997v5.pdf", 3, "05_survey_fig2_proceso_rag.png", (0.07, 0.06, 0.93, 0.445)),
    ("2312.10997v5.pdf", 14, "06_survey_tabla3_metricas.png", (0.13, 0.065, 0.86, 0.32)),
    ("3 BM25_Score.pdf", 5, "07_bm25_vs_tfidf_tabla.png", None),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for pdf, page, name, crop in IMAGENES:
        pg = pdfium.PdfDocument(CORPUS / pdf)[page - 1]
        w, h = pg.get_size()
        frac = 1.0 if crop is None else max(crop[2] - crop[0], crop[3] - crop[1])
        img = pg.render(scale=LADO_MAYOR / (max(w, h) * frac)).to_pil()
        if crop:
            W, H = img.size
            img = img.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
        img.thumbnail((LADO_MAYOR, LADO_MAYOR))
        img.save(OUT / name, optimize=True)
        print(f"{name:34s} {img.size[0]}x{img.size[1]}  {(OUT / name).stat().st_size / 1024:.0f} KB  ← {pdf} p.{page}")


if __name__ == "__main__":
    main()
