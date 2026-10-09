"""Genera el PDF de entrega del Avance 2 (docs/Entrega_Avance2.pdf).

Usa automáticamente, si existen:
  - eval/results/summary_<antes>.json y summary_<despues>.json  (resultados Ragas)
  - docs/ragas_comparison.png                                   (gráfico antes/después)
  - docs/screenshots/01_... a 06_... (.png/.jpg)                (capturas)

Uso:
  python scripts/build_deliverable_pdf.py --autor "Tu Nombre" --curso "Tu curso" \
      --url https://tu-app.streamlit.app --repo https://github.com/tu-usuario/tu-repo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from _pdfkit import (ST, PageBreak, Spacer, build, bullets, callout, code, cm, image_or_placeholder,  # noqa: E402
                     P, Image, table)

ROOT = Path(__file__).resolve().parent.parent
DOCS, RES, SHOTS = ROOT / "docs", ROOT / "eval" / "results", ROOT / "docs" / "screenshots"

ap = argparse.ArgumentParser()
ap.add_argument("--autor", default="[Tu nombre]")
ap.add_argument("--curso", default="[Curso / Programa]")
ap.add_argument("--url", default="[URL pública de la aplicación]")
ap.add_argument("--repo", default="[URL del repositorio de GitHub]")
ap.add_argument("--before", default="base")
ap.add_argument("--after", default="mejora")
a = ap.parse_args()


# ------------------------------------------------------------------ Diagrama ----
def make_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#f4f3ff", ec="#4f46e5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1e2433")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8, color="#596079", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color="#4f46e5", lw=1.5))

    ax.text(0.1, 5.95, "FASE 1 - INDEXACIÓN (una vez)", fontsize=10, fontweight="bold", color="#4f46e5")
    ax.text(0.1, 2.95, "FASE 2 - CONSULTA (cada pregunta)", fontsize=10, fontweight="bold", color="#4f46e5")
    w, h, y1, y2 = 2.0, 1.25, 4.4, 1.3
    xs = [0.1 + i * 2.2 for i in range(5)]
    top = [("Documentos", "PDF, DOCX, TXT,\nMD, HTML"), ("Ingesta", "texto por página\nrag/ingest.py"),
           ("Chunking", "800 car., solape 120\ncorte recursivo"), ("Embeddings", "MiniLM multilingüe\n384 dim, local"),
           ("Chroma", "índice coseno\ntexto + metadatos")]
    for x, (t, s) in zip(xs, top):
        box(x, y1, w, h, t, s, fc="#ffffff" if t == "Documentos" else "#f4f3ff")
    for i in range(4):
        arrow(xs[i] + w, y1 + h / 2, xs[i + 1], y1 + h / 2)

    bottom = [("Pregunta", "chat con historial\n(Streamlit)"), ("Reescritura", "seguimiento -> pregunta\nindependiente"),
              ("Recuperación", "top_k = 4, umbral\nde distancia 0.75"), ("Generación", "LLM + system + few-shot\n<contexto><pregunta>"),
              ("Respuesta", "citas [n], documento\ny fragmento, rechazo")]
    for x, (t, s) in zip(xs, bottom):
        warn = t == "Respuesta"
        box(x, y2, w, h, t, s, fc="#fff8eb" if warn else "#f4f3ff", ec="#f59e0b" if warn else "#4f46e5")
    for i in range(4):
        arrow(xs[i] + w, y2 + h / 2, xs[i + 1], y2 + h / 2)
    x_from, y_from, x_to, y_to = xs[4] + w / 2, y1, xs[2] + w / 2, y2 + h + 0.03
    arrow(x_from, y_from, x_to, y_to)
    ax.text((x_from + x_to) / 2 + 0.2, (y_from + y_to) / 2 + 0.1, "búsqueda por similitud", fontsize=8, color="#4f46e5",
            ha="center", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(0.1, 0.45, "Embeddings locales: al LLM solo se envían los fragmentos recuperados, nunca el corpus completo.",
            fontsize=8, color="#596079", style="italic")
    plt.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


DOCS.mkdir(exist_ok=True)
make_diagram(DOCS / "rag_flow.png")


# ------------------------------------------------------------------ Datos de evaluación ----
def load(tag):
    p = RES / f"summary_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


before, after = load(a.before), load(a.after)
M = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ATTR = {
    "faithfulness": ("generación", "el LLM afirma cosas que el contexto no respalda",
                     "endurecer el system prompt, añadir ejemplos few-shot de rechazo y bajar la temperatura"),
    "answer_relevancy": ("generación / prompt", "la respuesta es incompleta, redundante o se desvía de la pregunta",
                         "precisar el formato de salida y la reescritura de la pregunta"),
    "context_precision": ("recuperación", "entre los fragmentos recuperados hay mucho ruido o los relevantes quedan abajo",
                          "reducir top_k, ajustar el umbral de distancia o añadir un re-ranker"),
    "context_recall": ("chunking / recuperación", "falta información necesaria en los fragmentos recuperados",
                       "aumentar top_k, reducir el tamaño del chunk o subir el solapamiento"),
}


def fmt(v):
    return "n/d" if v is None else f"{v:.2f}"


# ------------------------------------------------------------------ Documento ----
s = []
s += [Spacer(1, 3 * cm), P("Asistente Experto basado en RAG", "title"), Spacer(1, 6),
      P("Avance 2: flujo RAG completo, evaluación con Ragas y despliegue conversacional", "sub"), Spacer(1, 26),
      P(f"<b>Autor:</b> {a.autor}"), P(f"<b>Curso:</b> {a.curso}"),
      P(f"<b>Aplicación pública:</b> {a.url}"), P(f"<b>Repositorio:</b> {a.repo}"), Spacer(1, 18)]
s += callout("Resumen", "Asistente conversacional que responde solo con la información de una base de conocimientos "
             "propia, cita el documento y fragmento de cada respuesta y se abstiene cuando la información no está en el "
             "corpus. El índice vectorial y los documentos permanecen bajo control del proyecto: los embeddings se calculan "
             "localmente y al LLM solo se envían los fragmentos recuperados para cada consulta.")
s += [PageBreak()]

# 1 ---------------------------------------------------------------------------------------------
s += [P("1. Flujo RAG implementado", "h1"),
      Image(str(DOCS / "rag_flow.png"), 15.6 * cm, 15.6 * cm * 4.7 / 8.2), Spacer(1, 4),
      P("Figura 1. Pipeline de extremo a extremo: fase de indexación (arriba) y fase de consulta (abajo).", "small"),
      P("Justificación de las decisiones técnicas", "h2")]
s += table([
    ["Etapa", "Decisión", "Justificación"],
    ["Selección de documentos", "5 documentos Markdown (RAG, chunking y embeddings, bases vectoriales, evaluación con Ragas, prompting y alucinaciones).",
     "Cubren el dominio del asistente (tutor de RAG), son de longitud homogénea y permiten verificar manualmente cada respuesta al evaluar."],
    ["Ingesta", "Lectores para PDF (pypdf), DOCX (python-docx), TXT/MD y HTML (BeautifulSoup).",
     "Cubre los formatos pedidos; en PDF se conserva el número de página para citar."],
    ["Chunking", "800 caracteres, solapamiento 120 (15 %), corte recursivo: párrafo, línea, frase, palabra.",
     "Un chunk contiene una idea completa sin mezclar temas; el solape (10-20 %) evita perder ideas en los límites; respetar párrafos mantiene la coherencia."],
    ["Vectorización", "paraphrase-multilingual-MiniLM-L12-v2 (384 dimensiones), local.",
     "Multilingüe (español), liviano, sin costo y sin enviar texto a terceros; consulta y documentos usan el mismo modelo."],
    ["Base vectorial", "Chroma persistente con distancia coseno. Metadatos: fuente, página, nº de fragmento.",
     "Código abierto, embebida y persistente en disco; los metadatos permiten citar documento y fragmento."],
    ["Recuperación", "Similitud coseno, top_k = 4 y descarte de fragmentos con distancia > 0.75.",
     "El top_k equilibra cobertura y ruido; el umbral evita responder con contexto irrelevante (control de alucinaciones)."],
    ["Generación", "LLM vía LangChain, temperatura 0.1. System prompt con reglas, 2 ejemplos few-shot (uno de rechazo), delimitadores <contexto>/<pregunta> y formato de salida con citas [n] y máximo 120 palabras.",
     "Se refinó el prompt del Avance 1: delimitadores contra prompt injection, citas obligatorias y frase de rechazo exacta para detectar abstenciones."],
    ["Conversación", "Reescritura de la pregunta de seguimiento usando los últimos 6 mensajes.",
     "Permite preguntas como '¿y cuál es su ventaja?', que por sí solas no sirven para buscar."],
], [3, 6.2, 7.2])[:-1]

# 2 ---------------------------------------------------------------------------------------------
s += [PageBreak(), P("2. Ejecución de consultas sobre el RAG", "h1"),
      P("Se muestran tres consultas representativas: una factual, una de síntesis que requiere varios fragmentos y una fuera del corpus. "
        "En cada captura se observa la respuesta, las citas [n] y el panel de fuentes.")]
cases = [
    ("01_consulta_factual", "Consulta factual", "Ej.: «¿Qué modelo de embeddings se recomienda para español y cuántas dimensiones tiene?»"),
    ("02_consulta_sintesis", "Consulta de síntesis (varios fragmentos)", "Ej.: «¿Qué ventajas tiene RAG frente al fine-tuning?»"),
    ("03_consulta_fuera_corpus", "Consulta fuera del corpus", "Ej.: «¿Quién ganó el Mundial de 2018?»  -> debe rechazar sin inventar"),
]
for i, (stem, title, hint) in enumerate(cases, 1):
    s += [P(f"2.{i} {title}", "h2")]
    s += image_or_placeholder([SHOTS / f"{stem}.png", SHOTS / f"{stem}.jpg"], f"Figura {i + 1}. {title}.",
                              f"docs/screenshots/{stem}.png<br/>{hint}", max_h=8.2 * cm)
    s += [P("<b>Análisis:</b> [Describe si la respuesta es correcta, qué fragmentos se citaron y si la abstención fue la adecuada.]")]

# 3 ---------------------------------------------------------------------------------------------
s += [PageBreak(), P("3. Evaluación con Ragas", "h1"),
      P("Conjunto de evaluación: 20 preguntas (eval/eval_questions.json), cada una con respuesta de referencia: "
        "16 con respuesta en el corpus y 4 fuera de la base de conocimientos. Las cuatro métricas de Ragas se calculan sobre las "
        "16 respondibles; las 4 restantes miden el control de alucinaciones mediante la tasa de abstención "
        "(Ragas no aplica ahí porque no existe un contexto correcto). Las métricas van de 0 a 1 (más alto es mejor).")]
if after:
    ref = before or after
    rows = [["Métrica", f"Antes ({a.before})" if before else "—", f"Después ({a.after})", "Cambio"]]
    for m in M:
        b, f = (before or {}).get("ragas", {}).get(m), after["ragas"].get(m)
        rows.append([m, fmt(b), fmt(f), "n/d" if (b is None or f is None) else f"{f - b:+.2f}"])
    b, f = (before or {}).get("abstention_rate_out_of_scope"), after["abstention_rate_out_of_scope"]
    rows.append(["Abstención fuera del corpus", fmt(b), fmt(f), "n/d" if (b is None or f is None) else f"{f - b:+.2f}"])
    s += [P("Tabla 1. Resultados antes y después de la iteración de mejora", "h2")] + table(rows, [6, 3.6, 3.6, 3.2])
    if before:
        pb, pa = before["params"], after["params"]
        s += table([["Parámetro", "Antes", "Después"],
                    ["chunk_size", pb["chunk_size"], pa["chunk_size"]],
                    ["chunk_overlap", pb["chunk_overlap"], pa["chunk_overlap"]],
                    ["top_k", pb["top_k"], pa["top_k"]]], [6, 5, 5.4])
    s += image_or_placeholder([DOCS / "ragas_comparison.png"], "Figura 5. Comparación de métricas.", "docs/ragas_comparison.png", max_h=8 * cm)
    worst = min(M, key=lambda m: after["ragas"].get(m, 1.0))
    comp, why, fix = ATTR[worst]
    s += [P("Interpretación", "h2"),
          P(f"Con la configuración final, la métrica más baja es <b>{worst}</b> ({fmt(after['ragas'].get(worst))}). "
            f"Se atribuye principalmente a la etapa de <b>{comp}</b>: {why}. Para mejorarla se propone {fix}."),
          P("[Complementa con tu propio análisis: qué cambió entre antes y después, por qué crees que ocurrió y qué casos concretos "
            "(ver eval/results/detalle_*.csv) explican el resultado.]", "small")]
else:
    s += callout("Resultados pendientes",
                 "Ejecuta python eval/run_eval.py (línea base y mejora), luego python eval/compare.py y vuelve a correr "
                 "este script: la tabla, el gráfico y la interpretación se rellenan automáticamente.", kind="warn")
    s += table([["Métrica", "Antes (base)", "Después (mejora)", "Cambio"]] +
               [[m, "—", "—", "—"] for m in M] + [["Abstención fuera del corpus", "—", "—", "—"]], [6, 3.6, 3.6, 3.2])
s += [P("Iteración de mejora documentada", "h2"),
      P("Parámetros modificados: tamaño de chunk, solapamiento y top_k. Hipótesis: chunks más pequeños y más "
        "fragmentos recuperados aumentan context_recall al aislar mejor cada idea, a costa de algo de ruido en context_precision.")]

# 4 ---------------------------------------------------------------------------------------------
s += [PageBreak(), P("4. Interfaz de chat desplegada", "h1"),
      P(f"Aplicación pública: <b>{a.url}</b><br/>Repositorio: <b>{a.repo}</b>"),
      P("La interfaz (Streamlit) mantiene el historial del diálogo, reescribe las preguntas de seguimiento, muestra el documento y "
        "fragmento que respaldan cada respuesta y responde con un aviso claro cuando la información no está en el corpus. "
        "Las claves de API se gestionan con los Secrets de la plataforma.")]
ui = [
    ("04_chat_seguimiento", "Conversación con preguntas de seguimiento", "Ej.: «¿Qué es el chunking?» y luego «¿y qué solapamiento se recomienda?»"),
    ("05_chat_fuera_corpus", "Caso fuera del alcance del corpus", "Una pregunta ajena al corpus mostrando el aviso de 'fuera del alcance'"),
    ("06_app_desplegada", "Aplicación desplegada (con la URL visible)", "Captura del navegador con la URL pública en la barra de direcciones"),
]
for i, (stem, title, hint) in enumerate(ui, 1):
    s += [P(f"4.{i} {title}", "h2")]
    s += image_or_placeholder([SHOTS / f"{stem}.png", SHOTS / f"{stem}.jpg"], f"Figura {5 + i}. {title}.",
                              f"docs/screenshots/{stem}.png<br/>{hint}", max_h=8.2 * cm)

# 5 ---------------------------------------------------------------------------------------------
s += [PageBreak(), P("5. Enlaces", "h1")] + bullets([f"URL pública de la aplicación: {a.url}", f"Repositorio de GitHub: {a.repo}"])
build(DOCS / "Entrega_Avance2.pdf", s, "Asistente Experto RAG - Avance 2")
print("PDF generado: docs/Entrega_Avance2.pdf")
