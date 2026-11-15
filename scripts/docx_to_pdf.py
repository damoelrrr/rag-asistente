"""Convierte docs/Entrega_Avance3.docx a PDF (reportlab) respetando el orden de la plantilla: títulos, tablas e imágenes."""
import io
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docx  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle  # noqa: E402

from _pdfkit import LINE, ST, S, build, cm, safe  # noqa: E402

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/Entrega_Avance3.docx")
OUT = SRC.with_suffix(".pdf")
d = docx.Document(SRC)
W = 16.4 * cm
H_MAX = 21 * cm
mono = S("mono", fontName="Courier", fontSize=7, leading=8.6)
cell = S("c", fontSize=8.6, leading=11)
cellb = S("cb", fontName="Helvetica-Bold", fontSize=8.6, leading=11)


def para_flow(p, style):
    imgs = []
    for blip in p._element.iter(qn("a:blip")):
        part = d.part.related_parts[blip.get(qn("r:embed"))]
        imgs.append(part.blob)
    out = []
    text = ""
    is_mono = False
    for r in p.runs:
        t = escape(safe(r.text))
        if r.bold:
            t = f"<b>{t}</b>"
        if r.italic:
            t = f"<i>{t}</i>"
        if r.font.name and "Consolas" in r.font.name:
            is_mono = True
        text += t
    if text.strip():
        out.append(Paragraph(text.replace("\n", "<br/>"), mono if is_mono else style))
    return out, imgs


def img_flow(blob, max_w):
    w, h = ImageReader(io.BytesIO(blob)).getSize()
    k = min(max_w / w, H_MAX / h)
    return Image(io.BytesIO(blob), w * k, h * k)


story = []
body = d.element.body
for el in body.iterchildren():
    tag = el.tag.split("}")[1]
    if tag == "p":
        p = docx.text.paragraph.Paragraph(el, d)
        name = p.style.name
        st = {"Title": "title", "Heading 1": "h1", "Heading 2": "h2", "Heading 3": "h2"}.get(name, "p")
        fl, imgs = para_flow(p, ST[st])
        story += fl + [img_flow(b, W) for b in imgs]
    elif tag == "tbl":
        t = docx.table.Table(el, d)
        ncols = len(t.columns)
        grid = [int(g.get(qn("w:w"))) for g in el.find(qn("w:tblGrid")).findall(qn("w:gridCol"))]
        tot = sum(grid)
        widths = [W * g / tot for g in grid]
        data, spans = [], []
        for ri, row in enumerate(t.rows):
            cells, seen = [], []
            for ci, c in enumerate(row.cells):
                if c._tc in seen:
                    cells.append("")
                    continue
                seen.append(c._tc)
                span = sum(1 for x in row.cells if x._tc is c._tc)
                if span > 1:
                    spans.append(("SPAN", (ci, ri), (ci + span - 1, ri)))
                cw = sum(widths[ci:ci + span]) - 12
                flows = []
                for p in c.paragraphs:
                    fl, imgs = para_flow(p, cellb if (ci == 0 and ncols == 2 and span == 1) or (ri == 0 and ncols > 2) else cell)
                    flows += fl + [img_flow(b, cw) for b in imgs]
                cells.append(flows or "")
            data.append(cells)
        tb = Table(data, colWidths=widths, repeatRows=1 if ncols > 2 else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA5AE")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)] + spans
        if ncols > 2:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6EC")))
        tb.setStyle(TableStyle(style))
        story += [tb, Spacer(1, 8)]

build(OUT, story, "Avance 3 · Agente RAG con herramientas · Stiven Rodriguez y Daniel Rodriguez")
print("PDF:", OUT)
