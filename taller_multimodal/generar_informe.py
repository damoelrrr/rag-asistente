"""Genera el informe del taller (Word) llenando la plantilla de entrega con los resultados de resultados/.

Los textos de análisis (lo que escribimos nosotros) están en informe/analisis.json; los datos (respuestas,
tablas de aciertos, transcripciones, errores) se toman de resultados/*.json. Las "capturas" de la ejecución
son imágenes generadas a partir de las salidas guardadas en resultados/*.txt.

Uso:
    python taller_multimodal/generar_informe.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt
from PIL import Image, ImageDraw, ImageFont

from common import AQUI, IMAGENES, RESULTADOS

INFORME = AQUI / "informe"
PLANTILLA = INFORME / "plantilla.docx"
SALIDA = INFORME / "Informe_Taller_Multimodal.docx"
CAPTURAS = INFORME / "capturas"
PENDIENTE = "[PENDIENTE]"


def cargar(nombre: str, defecto=None):
    ruta = RESULTADOS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else defecto


# ---------------------------------------------------------------------------
# Capturas: texto de consola → PNG con aspecto de terminal
# ---------------------------------------------------------------------------
def _fuente(tam: int):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Consolas.ttf", "C:/Windows/Fonts/consola.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, tam)
    return ImageFont.load_default()


def captura(texto: str, nombre: str, ancho: int = 105, max_lineas: int = 70) -> Path:
    lineas = []
    for linea in texto.splitlines():
        lineas += textwrap.wrap(linea, ancho, replace_whitespace=False, drop_whitespace=False) or [""]
    if len(lineas) > max_lineas:
        lineas = lineas[:max_lineas - 1] + [f"... ({len(lineas) - max_lineas + 1} líneas más en resultados/)"]
    fuente, alto_linea, margen = _fuente(15), 20, 18
    w = int(fuente.getlength("M") * ancho) + 2 * margen
    img = Image.new("RGB", (w, alto_linea * len(lineas) + 2 * margen + 22), "#1e1e1e")
    d = ImageDraw.Draw(img)
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):  # barra de ventana
        d.ellipse((margen + i * 20, 8, margen + i * 20 + 11, 19), fill=color)
    for i, linea in enumerate(lineas):
        color = "#9cdcfe" if linea.startswith(("$", "===", "---")) else "#f48771" if "rror" in linea else "#d4d4d4"
        d.text((margen, margen + 22 + i * alto_linea), linea, font=fuente, fill=color)
    CAPTURAS.mkdir(parents=True, exist_ok=True)
    ruta = CAPTURAS / nombre
    img.save(ruta)
    return ruta


def seccion_txt(nombre: str, desde: str | None = None, hasta: str | None = None) -> str:
    ruta = RESULTADOS / nombre
    if not ruta.exists():
        return f"(no se encontró resultados/{nombre}: ejecuten el script correspondiente)"
    t = ruta.read_text(encoding="utf-8")
    if desde and desde in t:
        t = t[t.index(desde):]
    if hasta and hasta in t[1:]:
        t = t[: t.index(hasta, 1)]
    return t.strip()


# ---------------------------------------------------------------------------
# Escritura en celdas
# ---------------------------------------------------------------------------
def escribir(celda, texto="", imagenes=(), ancho_cm: float = 11.0, tam: int = 9):
    for p in celda.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p = celda.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    if imagenes:
        run = p.add_run()
        for img in imagenes:
            run.add_picture(str(img), width=Cm(ancho_cm / max(1, len(imagenes)) - 0.2 * (len(imagenes) > 1)))
            run.add_text(" ")
        if texto:
            p = celda.add_paragraph()
    for i, parrafo in enumerate(str(texto).split("\n")):
        if i:
            p = celda.add_paragraph()
        p.add_run(parrafo).font.size = Pt(tam)


def filas(tabla, n: int):
    """Asegura que la tabla tenga n filas de datos (además del encabezado)."""
    while len(tabla.rows) - 1 < n:
        tabla.add_row()
    while len(tabla.rows) - 1 > n:
        tabla._tbl.remove(tabla.rows[-1]._tr)


def quitar_enunciado(doc) -> None:
    """Deja solo la plantilla de entrega: elimina el enunciado que va antes de 'Plantilla de entrega'."""
    body = doc.element.body
    hijos = list(body.iterchildren())
    corte = next(i for i, el in enumerate(hijos) if "Plantilla de entrega" in "".join(el.itertext()))
    for el in hijos[:corte]:
        body.remove(el)
    doc.paragraphs[0].text = "Informe – Taller: Entrada multimodal y audio"
    if doc.paragraphs[1].text.startswith("Completen"):
        doc.paragraphs[1]._element.getparent().remove(doc.paragraphs[1]._element)


def tipo_legible(anot) -> str:
    s = str(anot).replace("typing.", "")
    if s.startswith("Literal["):
        return "enum: " + s[8:-1].replace("'", "")
    return {"<class 'str'>": "string", "<class 'bool'>": "boolean", "<class 'int'>": "integer (≥ 0)",
            "str | None": "string o null", "Optional[str]": "string o null"}.get(s, s)


def main() -> None:
    import parte1_vision as p1

    A = json.loads((INFORME / "analisis.json").read_text(encoding="utf-8"))
    a = lambda k: A.get(k) or PENDIENTE  # noqa: E731

    doc = Document(PLANTILLA)
    quitar_enunciado(doc)
    T = doc.tables  # 0 datos, 1 = 1.1, 2 esquema, 3 aciertos, 4 resumen 1.2, 5 = 1.3, 6 = 1.4, 7 audio, 8 obs, 9 = 2.2, 10 flujo, 11 dificultades

    # Datos del grupo
    escribir(T[0].rows[2].cells[1], "\n".join(A["integrantes"]), tam=10)
    escribir(T[0].rows[3].cells[1], A["repositorio"], tam=10)

    # 1.1
    r11 = cargar("1_1_pregunta.json", {})
    escribir(T[1].rows[0].cells[1], f"{p1.PREGUNTA_IMAGEN.name} (modelo: {r11.get('modelo', p1.VISION_MODEL)})",
             imagenes=[p1.PREGUNTA_IMAGEN])
    escribir(T[1].rows[1].cells[1], r11.get("pregunta", p1.PREGUNTA))
    escribir(T[1].rows[2].cells[1], r11.get("respuesta", PENDIENTE))
    escribir(T[1].rows[3].cells[1], a("1.1_analisis"))

    # 1.2 esquema
    campos = p1.FichaDiapositiva.model_fields
    filas(T[2], len(campos))
    for fila, (nombre, campo) in zip(T[2].rows[1:], campos.items()):
        for celda, txt in zip(fila.cells, (nombre, tipo_legible(campo.annotation), campo.description)):
            escribir(celda, txt)

    # 1.2 aciertos
    r12 = cargar("1_2_extraccion.json", {})
    tabla = r12.get("tabla", [])
    filas(T[3], max(1, len(tabla)))
    for fila, f in zip(T[3].rows[1:], tabla):
        valores = (f["imagen"].split("_", 1)[0], f["campo"], json.dumps(f["valor_real"], ensure_ascii=False),
                   json.dumps(f["valor_extraido"], ensure_ascii=False), "Sí" if f["correcto"] else "No")
        for celda, txt in zip(fila.cells, valores):
            escribir(celda, txt, tam=8)
    if r12:
        pct = 100 * r12["aciertos"] / r12["total"]
        escribir(T[4].rows[0].cells[1], f"{r12['aciertos']} de {r12['total']} campos = {pct:.1f} %", tam=10)
    cap_val = captura(seccion_txt("parte1_extraccion.txt", "1.2 (demostración)"), "1_2_validacion.png")
    escribir(T[4].rows[1].cells[1], a("1.2_validacion"), imagenes=[cap_val])
    escribir(T[4].rows[2].cells[1], a("1.2_errores"))

    # 1.3
    r13 = cargar("1_3_multiples.json", {})
    escribir(T[5].rows[0].cells[1], ", ".join(p1.IMAGENES_MULTIPLES),
             imagenes=[IMAGENES / n for n in p1.IMAGENES_MULTIPLES])
    escribir(T[5].rows[1].cells[1], r13.get("pregunta", p1.PREGUNTA_MULTIPLE))
    escribir(T[5].rows[2].cells[1], r13.get("respuesta", PENDIENTE))
    escribir(T[5].rows[3].cells[1], a("1.3_analisis"))

    # 1.4: caso principal en la tabla; los demás casos en una tabla adicional
    r14 = cargar("1_4_errores.json", [])
    principal = next((c for c in r14 if c["caso"] == A.get("1.4_caso")), next((c for c in r14 if c.get("error")), None))
    if principal:
        err = principal["error"] or {}
        escribir(T[6].rows[0].cells[1], principal["descripcion"])
        texto_err = seccion_txt("parte1_error.txt", f"--- Caso: {principal['caso']} ---", "--- Caso:")
        escribir(T[6].rows[1].cells[1], f"HTTP {err.get('status_code')} – {err.get('mensaje')}",
                 imagenes=[captura(texto_err, "1_4_error.png")])
        escribir(T[6].rows[2].cells[1], a("1.4_causa"))
        otros = [c for c in r14 if c is not principal]
        if otros:
            ancla = T[6]._tbl
            p = doc.add_paragraph("Otros casos probados", style="Heading 3")
            extra = doc.add_table(rows=1, cols=3, style=T[6].style)
            for celda, txt in zip(extra.rows[0].cells, ("Caso", "Resultado de la API", "Causa / comentario")):
                escribir(celda, txt, tam=9)
            for c in otros:
                fila = extra.add_row().cells
                e = c.get("error")
                escribir(fila[0], c["descripcion"], tam=8)
                escribir(fila[1], f"HTTP {e['status_code']} – {e['mensaje']}" if e else
                         "Sin error: la API aceptó la solicitud", tam=8)
                escribir(fila[2], A.get("1.4_otros", {}).get(c["caso"], PENDIENTE), tam=8)
            ancla.addnext(extra._tbl)
            ancla.addnext(p._element)

    # 2.1
    r2 = cargar("2_2_respuestas.json") or cargar("2_1_transcripciones.json", [])
    modelo_audio = next((m for it in r2 for m in it.get("transcripciones", {})), None)
    filas(T[7], max(1, len(r2)))
    for fila, it in zip(T[7].rows[1:], r2):
        t = it.get("transcripciones", {}).get(modelo_audio, {})
        dur = t.get("duracion_s")
        wer_txt = f"\n(WER {t['wer']:.0%})" if "wer" in t else ""
        for celda, txt in zip(fila.cells, (it["audio"], it.get("hablante", ""), it.get("condicion", ""),
                                           f"{dur:.1f}" if dur else "", it.get("texto", ""),
                                           t.get("texto", t.get("error", {}).get("mensaje", "")) + wer_txt)):
            escribir(celda, txt, tam=8)
    escribir(T[8].rows[0].cells[1], a("2.1_observaciones"))

    # 2.2
    filas(T[9], max(1, len(r2)))
    for fila, it in zip(T[9].rows[1:], r2):
        t = it.get("transcripciones", {}).get(modelo_audio, {})
        resp = it.get("respuesta", {})
        texto = resp.get("texto", PENDIENTE)
        if resp.get("fuentes"):
            texto += "\nFuentes: " + "; ".join(resp["fuentes"])
        for celda, txt in zip(fila.cells, (it["audio"], t.get("texto", ""), texto,
                                           A.get("2.2_correcta", {}).get(it.get("clave") or "", PENDIENTE))):
            escribir(celda, txt, tam=8)
    flujo = "$ python taller_multimodal/parte2_audio.py\n" + seccion_txt("parte2_respuestas.txt", None, "\n--- audio2")
    escribir(T[10].rows[0].cells[1], imagenes=[captura(flujo, "2_2_flujo.png")])
    escribir(T[10].rows[1].cells[1], a("2.2_afecto"))
    escribir(T[10].rows[2].cells[1], a("conclusion"))

    # Dificultades y aprendizajes
    escribir(T[11].rows[0].cells[1], a("dificultades"))
    escribir(T[11].rows[1].cells[1], a("aprendizajes"))

    # Anexo: capturas completas de cada ejecución
    doc.add_page_break()
    doc.add_paragraph("Anexo: capturas de la ejecución", style="Heading 1")
    for nombre, tit in (("parte1_pregunta.txt", "1.1 Pregunta sobre una imagen"),
                        ("parte1_extraccion.txt", "1.2 Extracción estructurada"),
                        ("parte1_multiples.txt", "1.3 Varias imágenes"),
                        ("parte1_error.txt", "1.4 Errores provocados"),
                        ("parte2_transcripcion.txt", "2.1 Transcripción"),
                        ("parte2_respuestas.txt", "2.2 Respuestas")):
        if (RESULTADOS / nombre).exists():
            doc.add_paragraph(tit, style="Heading 3")
            doc.add_picture(str(captura(seccion_txt(nombre), f"anexo_{Path(nombre).stem}.png", max_lineas=90)),
                            width=Cm(16))
    doc.save(SALIDA)
    pendientes = [k for k, v in A.items() if v in ("", None, PENDIENTE)]
    print(f"Informe generado: {SALIDA}")
    if pendientes:
        print("Textos pendientes en informe/analisis.json:", ", ".join(pendientes))


if __name__ == "__main__":
    main()
