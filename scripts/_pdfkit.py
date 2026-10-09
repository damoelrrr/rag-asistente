"""Utilidades de maquetación PDF (reportlab) compartidas por los scripts de este repositorio."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

INDIGO = colors.HexColor("#4f46e5")
INK = colors.HexColor("#1e2433")
MUTED = colors.HexColor("#596079")
SOFT = colors.HexColor("#f4f3ff")
LINE = colors.HexColor("#d9dcee")
AMBER_BG, AMBER = colors.HexColor("#fff8eb"), colors.HexColor("#f59e0b")

_REPL = {"→": "->", "←": "<-", "✓": "[x]", "≥": ">=", "≤": "<=", "≈": "~", "│": "|", "├": "|", "└": "`", "─": "-", "▼": "v",
         "✔": "[x]", "☐": "[ ]", "📄": "", "🧠": ""}


def safe(t: str) -> str:
    for a, b in _REPL.items():
        t = t.replace(a, b)
    return t.encode("cp1252", errors="replace").decode("cp1252")


def S(name, **kw):
    base = dict(fontName="Helvetica", fontSize=10, leading=14.5, textColor=INK, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "title": S("title", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=INDIGO),
    "sub": S("sub", fontSize=13, leading=18, textColor=MUTED),
    "h1": S("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INDIGO, spaceBefore=14, spaceAfter=6),
    "h2": S("h2", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=INK, spaceBefore=9, spaceAfter=3),
    "p": S("p", spaceAfter=5),
    "small": S("small", fontSize=8.5, leading=11.5, textColor=MUTED),
    "cell": S("cell", fontSize=8.8, leading=11.5),
    "cellb": S("cellb", fontName="Helvetica-Bold", fontSize=8.8, leading=11.5, textColor=colors.white),
    "bullet": S("bullet", leftIndent=14, bulletIndent=3, spaceAfter=2.5),
    "code": S("code", fontName="Courier", fontSize=8.3, leading=11, textColor=colors.HexColor("#22304d")),
}


def P(text, style="p"):
    return Paragraph(safe(text), ST[style])


def bullets(items):
    return [Paragraph(safe(i), ST["bullet"], bulletText="\u2022") for i in items]


def code(text):
    box = Table([[Preformatted(safe(text.strip("\n")), ST["code"])]], colWidths=[16.4 * cm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f3fa")),
                             ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                             ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return [box, Spacer(1, 6)]


def callout(title, text, kind="info"):
    bg, bar = (SOFT, INDIGO) if kind == "info" else (AMBER_BG, AMBER)
    inner = [Paragraph(f"<b>{safe(title)}</b>", ST["p"]), Paragraph(safe(text), ST["p"])]
    t = Table([[inner]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3.5, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 11), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 7)]


def table(rows, widths, header=True):
    data = [[Paragraph(safe(str(c)), ST["cellb" if (header and i == 0) else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INDIGO)]
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafbff")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 8)]


def image_or_placeholder(path_candidates, caption, hint, max_w=15.5 * cm, max_h=10.5 * cm):
    """Inserta la imagen si existe; si no, un recuadro que indica qué captura falta."""
    for p in path_candidates:
        if Path(p).exists():
            w, h = ImageReader(str(p)).getSize()
            k = min(max_w / w, max_h / h)
            body = [Image(str(p), w * k, h * k)]
            break
    else:
        box = Table([[Paragraph(safe(f"<b>PENDIENTE: pega aquí la captura</b><br/>{hint}"), ST["small"])]],
                    colWidths=[15.5 * cm], rowHeights=[3.2 * cm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, AMBER), ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        body = [box]
    return [KeepTogether(body + [Spacer(1, 3), Paragraph(safe(f"<i>{caption}</i>"), ST["small"]), Spacer(1, 8)])]


def build(path, story, footer_text):
    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.3 * cm, 1.2 * cm, safe(footer_text))
        canvas.drawRightString(A4[0] - 2.3 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(2.3 * cm, 1.6 * cm, A4[0] - 2.3 * cm, 1.6 * cm)
        canvas.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                          topMargin=2 * cm, bottomMargin=2.2 * cm, title=footer_text, author="")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width,
                                                             doc.height, id="f")], onPage=deco)])
    doc.build(story)


__all__ = ["P", "bullets", "code", "callout", "table", "image_or_placeholder", "build", "PageBreak", "Spacer", "ST",
           "KeepTogether", "cm", "Image"]
