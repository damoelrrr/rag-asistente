"""Llena la plantilla oficial del Avance 3 con los resultados reales del proyecto.

Entrada:  Avance3_plantilla_de_entrega.docx (plantilla del curso, raíz del repo)
          eval/results/*.json|csv  (run_behavior_cases.py, run_eval.py, run_agent_eval.py)
          logs/agent_usage.jsonl   (registro de uso de tools)
          docs/agent_flow.png, docs/screenshots/a3_*.png
          docs/avance3_textos.json (análisis y reflexión escritos a partir de los resultados)
Salida:   docs/Entrega_Avance3.docx  (exportar a PDF con Word: Archivo > Exportar > PDF)

Uso:  python scripts/fill_avance3_docx.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402
from PIL import Image  # noqa: E402

from rag.agent_prompts import AGENT_SYSTEM_PROMPT  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.tools import (DOMINIOS_PERMITIDOS, MAX_CHARS_CONSULTA_BASE, MAX_CHARS_CONSULTA_WEB,  # noqa: E402
                       MAX_CHARS_EXTRACTO_WEB, MAX_CHARS_FRAGMENTO, MAX_DOCUMENTOS_LISTADOS, MAX_FRAGMENTOS,
                       MAX_RESULTADOS_WEB)

ROOT = Path(__file__).resolve().parent.parent
RES, SHOTS, DOCS = ROOT / "eval" / "results", ROOT / "docs" / "screenshots", ROOT / "docs"
TEMPLATE = ROOT / "Avance3_plantilla_de_entrega.docx"
OUT = DOCS / "Entrega_Avance3.docx"

INTEGRANTES = ["Stiven Rodriguez", "Daniel Rodriguez"]
REPO = "https://github.com/damoelrrr/rag-asistente"
APP_URL = "https://rag-asistente-bdqqdfqfnzl6bymy65mtm2.streamlit.app/"
BASE_TAG, AGENT_TAG = "pipeline_actual", "agente"  # corridas de Ragas/costo (mismo LLM para ambos)
FULL_W = 6.2  # ancho útil (pulgadas) de una celda combinada a todo el ancho
MAX_IMG_H = 7.6  # alto máximo de una imagen para que la fila quepa en una página
GRAY = RGBColor(0x6B, 0x75, 0x80)
MONO = "Consolas"


# ----------------------------------------------------------------------------- utilidades docx
def _clear(cell):
    """Deja la celda con un único párrafo vacío (conserva tcPr: ancho, bordes, sombreado)."""
    ps = cell.paragraphs
    for p in ps[1:]:
        p._element.getparent().remove(p._element)
    for r in list(ps[0].runs):
        r._element.getparent().remove(r._element)
    return ps[0]


def _run(p, text, size=10, bold=False, italic=False, mono=False, color=None):
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold, r.italic = bold, italic
    if mono:
        r.font.name = MONO
        r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO)
    if color:
        r.font.color.rgb = color
    return r


def put(cell, *items, size=10):
    """Escribe párrafos en una celda. Cada item: str, o dict(text, bold, italic, mono, size, bullet, color)."""
    p = _clear(cell)
    first = True
    for it in items:
        it = {"text": it} if isinstance(it, str) else it
        if not first:
            p = cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(it.get("after", 3))
        text = it["text"]
        if it.get("bullet"):
            text = "•  " + text
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.first_line_indent = Inches(-0.12)
        # **negrita** inline
        parts = text.split("**")
        for i, part in enumerate(parts):
            if part:
                _run(p, part, size=it.get("size", size), bold=it.get("bold", False) or i % 2 == 1,
                     italic=it.get("italic", False), mono=it.get("mono", False), color=it.get("color"))


def allow_split(row):
    trPr = row._tr.trPr
    if trPr is not None:
        for el in trPr.findall(qn("w:cantSplit")):
            trPr.remove(el)


def image_row(table, row_idx, label, image, caption=None, max_w=FULL_W):
    """Combina la fila (etiqueta + valor) en una sola celda a todo el ancho: etiqueta arriba e imagen debajo."""
    row = table.rows[row_idx]
    allow_split(row)
    cell = row.cells[0].merge(row.cells[-1])
    put(cell, {"text": label, "bold": True})
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if image and Path(image).exists():
        w, h = Image.open(image).size
        width = min(max_w, MAX_IMG_H * w / h)
        p.add_run().add_picture(str(image), width=Inches(width))
    else:
        _run(p, f"[Falta la imagen {Path(image).name if image else ''}]", color=RGBColor(0xC0, 0, 0), bold=True)
    if caption:
        c = cell.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _run(c, caption, size=8.5, italic=True, color=GRAY)


def tables(doc):
    return doc.tables


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def signed(x, nd=2):
    return "—" if x is None or pd.isna(x) else f"{x:+.{nd}f}"


# ----------------------------------------------------------------------------- datos
def last_log(question, before: Path | None = None):
    """Última entrada del registro para esa pregunta; con before=captura, la ejecución que aparece en esa captura."""
    from datetime import datetime, timezone
    limit = datetime.fromtimestamp(before.stat().st_mtime, timezone.utc) if before and before.exists() else None
    path = Settings().agent_log_path
    rec = None
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["pregunta"].strip() != question.strip():
            continue
        if limit and datetime.fromisoformat(r["timestamp"]) > limit:
            continue
        rec = r
    return rec


def tools_text(rec):
    if rec is None:
        return ["(sin registro)"]
    if not rec["tools"]:
        return [{"text": "ninguna", "mono": True, "size": 9},
                f"Llamadas al LLM: {rec['llamadas_llm']} · tokens: {rec['tokens_entrada']} + {rec['tokens_salida']} · {rec['segundos']} s"]
    out = [{"text": f"{t['name']}({json.dumps(t['args'], ensure_ascii=False)}) → {t['resumen']}", "mono": True, "size": 8.5}
           for t in rec["tools"]]
    out.append(f"Llamadas al LLM: {rec['llamadas_llm']} · tokens: {rec['tokens_entrada']} + {rec['tokens_salida']} · {rec['segundos']} s")
    return out


def main():
    textos = json.loads((DOCS / "avance3_textos.json").read_text(encoding="utf-8"))
    beh = pd.read_csv(RES / "behavior_results.csv")
    base = json.loads((RES / f"summary_{BASE_TAG}.json").read_text(encoding="utf-8"))
    agen_file = RES / f"summary_{AGENT_TAG}.json"
    agen = json.loads(agen_file.read_text(encoding="utf-8")) if agen_file.exists() else None
    cost = pd.read_csv(RES / "cost_comparison.csv").set_index("métrica")
    s = Settings()

    doc = Document(TEMPLATE)
    T = tables(doc)

    # --- encabezado: quitar las instrucciones grises
    for p in doc.paragraphs:
        if p.text.startswith(("Completen cada sección", "Inserten el diagrama", "Máximo una página")):
            p._element.getparent().remove(p._element)
    doc.paragraphs[0].runs[0].text = "Avance 3: Agente RAG con herramientas"

    # --- Datos del equipo
    put(T[0].cell(0, 1), *INTEGRANTES)
    put(T[0].cell(1, 1), REPO)
    put(T[0].cell(2, 1), APP_URL)

    # --- 1.1 Diagrama
    image_row(T[1], 0, "Diagrama", DOCS / "agent_flow.png",
              "Lo único que sale del sistema: hacia el LLM (Groq) el prompt, el historial reciente, la pregunta y los "
              f"resultados ya recortados de las tools (≤ {MAX_FRAGMENTOS} fragmentos de ≤ {MAX_CHARS_FRAGMENTO} caracteres); "
              f"hacia Tavily solo la consulta web (≤ {MAX_CHARS_CONSULTA_WEB} caracteres). El corpus y el índice nunca salen.")

    # --- 1.2 Tools
    rows = [
        ("buscar_en_base", "Búsqueda",
         "Recupera fragmentos de los documentos del curso (embeddings + BM25 sobre Chroma, local). Se usa para toda "
         "pregunta de contenido; en preguntas con varios temas, una llamada por tema.",
         "consulta: str",
         f"Máx. {MAX_FRAGMENTOS} fragmentos; ≤ {MAX_CHARS_FRAGMENTO} caracteres por fragmento; consulta ≤ "
         f"{MAX_CHARS_CONSULTA_BASE}. Cada fragmento lleva documento y página."),
        ("generar_referencia_apa", "Tarea",
         "Devuelve la referencia APA 7 exacta de un documento de la base a partir de sus metadatos (autores, año, "
         "título). Se usa cuando piden la referencia o la cita bibliográfica; el LLM nunca la escribe de memoria.",
         "documento: str (título, archivo o autor extraído de la pregunta)",
         "Solo documentos de la base (búsqueda por alias y similitud); si no existe devuelve SIN_RESULTADOS y la lista "
         "de documentos."),
        ("listar_documentos", "Búsqueda",
         "Catálogo de los documentos indexados. Se usa cuando preguntan qué temas o fuentes tiene el tutor.",
         "(ninguno)", f"Máx. {MAX_DOCUMENTOS_LISTADOS} documentos."),
        ("buscar_en_web", "Web",
         "Busca en Tavily solo en sitios oficiales permitidos cuando la base no tiene la respuesta y la pregunta es "
         "de información pública o actual del dominio.",
         "consulta: str",
         f"Máx. {MAX_RESULTADOS_WEB} resultados (título, URL, extracto ≤ {MAX_CHARS_EXTRACTO_WEB}); consulta ≤ "
         f"{MAX_CHARS_CONSULTA_WEB}; dominios fijos (include_domains + filtro al recibir); bloqueada si antes no se "
         "llamó a buscar_en_base."),
    ]
    for i, r in enumerate(rows, start=1):
        for j, val in enumerate(r):
            put(T[2].cell(i, j), {"text": val, "mono": j == 0, "size": 8.5 if j == 0 else 9})
    image_row(T[3], 0, "Prueba de la tool de búsqueda sin LLM (captura de .invoke() y su salida)",
              SHOTS / "a3_00_invoke.png",
              "scripts/demo_tool_invoke.py: la tool se llama con .invoke() sin LLM. Devuelve 4 fragmentos (máximo del "
              "código), cada uno ≤ 700 caracteres y con documento y página.")

    # --- 1.3 Bucle y registro
    put(T[4].cell(0, 1),
        "Bucle manual (rag/agent.py) con llm.bind_tools(tools) de LangChain y Groq (openai/gpt-oss-120b). En cada "
        "vuelta el LLM responde o pide tools; el programa ejecuta cada tool_call y devuelve el resultado como "
        "ToolMessage. Máx. 3 tools por vuelta; tool desconocida o argumentos inválidos devuelven un error al LLM "
        "en lugar de romper la app.")
    put(T[4].cell(1, 1),
        f"{s.agent_max_iterations} iteraciones por pregunta (AGENT_MAX_ITERATIONS). Si se alcanza, se pide al LLM una "
        "respuesta final sin tools, con lo que ya obtuvo o con el texto de rechazo.")
    put(T[4].cell(2, 1),
        "logs/agent_usage.jsonl (rag/agent_log.py): una línea JSON por pregunta con las tools llamadas, sus "
        "argumentos y un resumen del resultado, el número de llamadas al LLM, iteraciones, tokens de entrada y salida, "
        "segundos y si fue un rechazo. La interfaz muestra lo mismo en el desplegable «Herramientas usadas».")
    rec = last_log(beh.loc[beh.id == "c05", "pregunta"].iloc[0]) or last_log(beh.pregunta.iloc[1])
    allow_split(T[4].rows[3])
    put(T[4].cell(3, 1), {"text": json.dumps(rec, ensure_ascii=False, indent=1), "mono": True, "size": 7.5})

    # --- 2. Prompt
    allow_split(T[5].rows[0])
    put(T[5].cell(0, 1), *[{"text": line, "mono": True, "size": 7, "after": 1} for line in AGENT_SYSTEM_PROMPT.split("\n")])
    put(T[5].cell(1, 1), *[{"text": t, "bullet": True, "size": 9} for t in textos["cambios_prompt"]])

    # --- 3. Búsqueda web
    put(T[6].cell(0, 1), "Tavily Search API (search_depth=basic), llamada directa con requests; clave TAVILY_API_KEY en "
                         ".env o en los secrets de Streamlit Cloud.")
    put(T[6].cell(1, 1),
        "Solo si (1) ya se llamó a buscar_en_base y no tiene la respuesta, y (2) la pregunta es de información pública "
        "o actual del dominio del curso (documentación oficial, una definición general, una versión reciente). La regla "
        "está en el docstring de la tool y en el System Prompt; el paso 1 además se hace cumplir en el código.")
    put(T[6].cell(2, 1), f"Máx. {MAX_RESULTADOS_WEB} resultados. Cada uno: título, URL y extracto de ≤ "
                         f"{MAX_CHARS_EXTRACTO_WEB} caracteres, marcado «origen: web».")
    put(T[6].cell(3, 1),
        "Con [web: URL] después de cada dato y la sección «Según la web:», separada de «Según la base de conocimientos:». "
        "Si el modelo cita con la etiqueta interna [W1], el código la reemplaza por [web: URL]. En la interfaz, las fuentes "
        "web aparecen en su propio desplegable («Fuentes web», con título y URL).")
    sitios = [
        ("arxiv.org", "Repositorio de los artículos científicos originales; el survey de RAG del corpus viene de arXiv."),
        ("ollama.com · docs.ollama.com", "Sitio y documentación oficial de Ollama (LLMs locales, tema del curso)."),
        ("docs.langchain.com · python.langchain.com · langchain-ai.github.io",
         "Documentación oficial de LangChain y LangGraph, el framework con el que está hecho el agente."),
        ("docs.trychroma.com · scikit-learn.org",
         "Documentación oficial de Chroma (índice vectorial del proyecto) y de scikit-learn (TF-IDF y recuperación)."),
    ]
    assert {d for s_, _ in sitios for d in s_.split(" · ")} == set(DOMINIOS_PERMITIDOS)
    for i, (d, why) in enumerate(sitios, start=1):
        put(T[7].cell(i, 0), {"text": d, "mono": True, "size": 8.5})
        put(T[7].cell(i, 1), {"text": why, "size": 9})

    # --- 4. Ejecución de las tools
    casos = [
        (8, "Hola, ¿cómo estás?", "a3_01_saludo.png", "saludo"),
        (9, "¿Qué dos limitaciones de TF-IDF corrige BM25?", "a3_02_base.png", "base"),
        (10, "Necesito la referencia en formato APA del documento «Introducción a RAG».", "a3_03_apa.png", "tarea"),
        (11, "¿Qué es LangGraph y para qué sirve según la documentación oficial de LangChain?", "a3_04_web.png", "web"),
    ]
    for ti, q, img, key in casos:
        put(T[ti].cell(0, 1), q)
        put(T[ti].cell(1, 1), *tools_text(last_log(q, before=SHOTS / img)))
        image_row(T[ti], 2, "Captura", SHOTS / img, max_w=5.6)
        put(T[ti].cell(3, 1), textos["analisis_" + key])

    # --- 5. Ragas
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for i, m in enumerate(metrics, start=1):
        b_ = base["ragas"].get(m)
        a_ = agen["ragas"].get(m) if agen else None
        put(T[12].cell(i, 1), fmt(b_))
        put(T[12].cell(i, 2), fmt(a_) if agen else "pendiente")
        put(T[12].cell(i, 3), signed(a_ - b_) if agen else "—")
    put(T[13].cell(0, 1), textos["n_preguntas"])
    allow_split(T[13].rows[1])
    put(T[13].cell(1, 1), *[{"text": t, "size": 9} for t in textos["analisis_ragas"]])

    # --- 6. Casos de comportamiento (se agregan filas si hay más de 7 casos)
    tb = T[14]
    while len(tb.rows) - 1 < len(beh):
        tb._tbl.append(copy.deepcopy(tb.rows[-1]._tr))
    for i, r in enumerate(beh.itertuples(), start=1):
        vals = [str(i), r.tipo, r.pregunta, r.esperado, r.obtenido.replace("*", ""), r.tools_usadas.replace("; ", ";\n"),
                str(r.llamadas_llm), "Sí" if r.cumple == "sí" else "No"]
        for j, v in enumerate(vals):
            put(tb.cell(i, j), *[{"text": x, "size": 7.5, "mono": j == 5} for x in v.split("\n")], size=7.5)
        allow_split(tb.rows[i])

    # --- 7. Costo
    def c(metric, col):
        return cost.loc[metric, col]
    put(T[15].cell(1, 1), fmt(c("llamadas al LLM (promedio)", "pipeline fijo")))
    put(T[15].cell(1, 2), fmt(c("llamadas al LLM (promedio)", "agente")))
    for j, col in ((1, "pipeline fijo"), (2, "agente")):
        tin, tout = c("tokens de entrada (promedio)", col), c("tokens de salida (promedio)", col)
        put(T[15].cell(2, j), f"{tin + tout:,.0f}  ({tin:,.0f} entrada + {tout:,.0f} salida)")
    put(T[15].cell(3, 1), fmt(c("tiempo de respuesta en s (promedio)", "pipeline fijo")))
    put(T[15].cell(3, 2), fmt(c("tiempo de respuesta en s (promedio)", "agente")))
    put(T[16].cell(0, 1), textos["preguntas_costo"])

    # --- 8. Interfaz desplegada
    put(T[17].cell(0, 1), APP_URL)
    deploy_web = SHOTS / "a3_08_deploy_web.png"
    capturas = [
        (1, "Captura: tools usadas visibles en una respuesta", SHOTS / "a3_08_deploy_tools.png", "App desplegada."),
        (2, "Captura: fuentes separadas (base y web)", deploy_web if deploy_web.exists() else SHOTS / "a3_04_web.png",
         "App desplegada." if deploy_web.exists() else "Misma versión de la app ejecutada en local."),
        (3, "Captura: pregunta fuera del alcance de la base y de la web", SHOTS / "a3_08_deploy_fuera.png",
         "App desplegada: responde con el texto de rechazo exacto, sin tools."),
    ]
    for ri, label, img, cap in capturas:
        image_row(T[17], ri, label, img, cap, max_w=5.6)

    # --- 9. Reflexión
    allow_split(T[18].rows[0])
    cell = T[18].rows[0].cells[0].merge(T[18].rows[0].cells[1])
    put(cell, {"text": "Reflexión", "bold": True}, *[{"text": t, "size": 9.5, "after": 5} for t in textos["reflexion"]])

    doc.save(OUT)
    print("Documento:", OUT)


if __name__ == "__main__":
    main()
