"""Genera docs/agent_flow.png: flujo pregunta -> LLM -> tool -> resultado -> respuesta, y qué sale del sistema."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "agent_flow.png"
INK, LOCAL, EXT, ACC = "#1f2937", "#e8f0fe", "#fff1e0", "#4f46e5"

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, title, body="", fc="white", ec=INK, bold=True, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    if body:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=fs, weight="bold" if bold else "normal", color=INK)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color="#374151", linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color=INK)


def arrow(p, q, label="", color=ACC, rad=0.0, lx=0, ly=0, fs=8.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=15, lw=1.7, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=1.2))


# Zonas
ax.add_patch(FancyBboxPatch((0.2, 0.3), 9.15, 7.4, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f8fafc", ec="#94a3b8", lw=1.2, ls="--"))
ax.text(0.45, 7.5, "DENTRO DEL SISTEMA (tu aplicación)", fontsize=9.5, weight="bold", color="#475569", va="top")
ax.add_patch(FancyBboxPatch((9.75, 0.3), 4.05, 7.4, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#fffaf3", ec="#f59e0b", lw=1.2, ls="--"))
ax.text(9.95, 7.5, "SERVICIOS EXTERNOS", fontsize=9.5, weight="bold", color="#b45309", va="top")

# Nodos internos
box(0.5, 5.5, 1.9, 1.2, "Usuario", "pregunta\ny respuesta", fc="white")
box(3.0, 4.7, 3.1, 2.2, "Agente (bucle)", "rag/agent.py\n• máx. 5 iteraciones por pregunta\n• ejecuta las tools que pide el LLM\n• registra cada pregunta", fc=LOCAL)
box(6.6, 4.05, 2.5, 1.0, "Registro de uso", "logs/agent_usage.jsonl: tools, args,\nllamadas LLM, tokens, tiempo", fc="white", fs=10)

box(0.5, 0.6, 8.6, 3.35, "", fc="white", ec="#64748b")
ax.text(9.0, 3.8, "TOOLS  (rag/tools.py · los límites viven en el código)", fontsize=9.5, weight="bold", color="#475569", va="top", ha="right")
box(0.75, 1.9, 2.6, 1.55, "buscar_en_base", "Chroma + BM25 (local)\nmáx. 4 fragmentos\n≤ 700 caracteres c/u\ndocumento + página", fc=LOCAL, fs=10)
box(3.55, 1.9, 2.6, 1.55, "generar_referencia_apa", "tarea del dominio\nmetadatos exactos\n(autores, año, título)\n100 % local", fc=LOCAL, fs=10)
box(6.35, 1.9, 2.5, 1.55, "listar_documentos", "catálogo del índice\n(local)", fc=LOCAL, fs=10)
box(0.75, 0.8, 8.1, 0.85, "buscar_en_web", "", fc=EXT, ec="#f59e0b", fs=10)
ax.text(4.8, 0.97, "solo si la base no responde (bloqueada en el código si no se consultó la base antes) · máx. 3 resultados · dominios fijos",
        ha="center", va="center", fontsize=8.4, color="#7c2d12")

# Externos
box(10.1, 4.9, 3.4, 1.9, "LLM (Groq)", "gpt-oss-120b con tool calling\n\nRECIBE: system prompt, historial\nreciente, pregunta y resultados\nde las tools (ya recortados)\nNUNCA el corpus completo", fc=EXT, ec="#f59e0b", fs=10.5)
box(10.1, 1.0, 3.4, 2.1, "Buscador web (Tavily)", "RECIBE solo la consulta\n(≤ 200 caracteres) y la lista\nfija de dominios permitidos\n\nDEVUELVE: título, URL y\nextracto corto (origen: web)", fc=EXT, ec="#f59e0b", fs=10.5)

# Flechas
arrow((2.4, 6.3), (3.0, 6.3), "1 pregunta", ly=0.28)
arrow((3.0, 5.2), (2.2, 5.5), "", rad=0.0)
ax.text(1.45, 4.85, "9 respuesta + fuentes\n(base / web)", fontsize=8.6, color=ACC, ha="center")
arrow((6.1, 6.45), (10.1, 6.45), "2 pregunta + historial + mensajes de tools", ly=0.28)
arrow((10.1, 5.7), (6.1, 5.7), "3 respuesta final   o   tool_call (nombre + argumentos)", ly=-0.28)
arrow((6.1, 4.85), (6.6, 4.6), "", color="#64748b")
ax.text(6.35, 4.95, "log", fontsize=8, color="#64748b", ha="center")
arrow((3.5, 4.7), (2.3, 3.45), "")
ax.text(2.2, 4.25, "4 ejecuta la tool", fontsize=8.6, color=ACC, ha="center")
arrow((3.0, 3.45), (4.2, 4.7), "", color="#059669")
ax.text(4.35, 4.25, "5 resultado\n(ToolMessage)", fontsize=8.4, color="#059669", ha="left", va="center")
arrow((8.85, 1.1), (10.1, 1.5), "6 consulta", color="#b45309", ly=-0.3, fs=8.6)
arrow((10.1, 2.7), (8.85, 1.5), "7 extractos web", color="#b45309", lx=0.1, ly=0.28, fs=8.6)
ax.text(7.9, 7.12, "8  el resultado vuelve al LLM (paso 2) y el bucle se repite\n    hasta que el LLM responde o se llega al límite", fontsize=8.6, color=ACC, ha="center", style="italic", va="center")

ax.set_title("Arquitectura del agente: pregunta → LLM → tool → resultado → respuesta", fontsize=13, weight="bold", color=INK, pad=6)
plt.tight_layout()
plt.savefig(OUT, dpi=170)
print("Diagrama:", OUT)
