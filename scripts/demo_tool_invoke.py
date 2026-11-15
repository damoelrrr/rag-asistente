"""Prueba la tool de búsqueda SIN el LLM: se llama directamente con .invoke() y se imprime lo que recibiría el modelo.

Uso:  python scripts/demo_tool_invoke.py "¿Qué controlan k1 y b en BM25?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.tools import MAX_CHARS_FRAGMENTO, MAX_FRAGMENTOS, RunState, build_tools  # noqa: E402

consulta = sys.argv[1] if len(sys.argv) > 1 else "¿Qué controlan los parámetros k1 y b en BM25?"
pipe = RAGPipeline(Settings())
pipe.ensure_index()
state = RunState()
tools = {t.name: t for t in build_tools(pipe, state)}

print(f">>> buscar_en_base.invoke({{'consulta': {consulta!r}}})\n")
salida = tools["buscar_en_base"].invoke({"consulta": consulta})
print(salida)
print(f"\n--- {len(state.base_sources)} fragmentos (máx. {MAX_FRAGMENTOS}); "
      f"longitud de cada uno: {[len(s['text']) for s in state.base_sources]} (máx. {MAX_CHARS_FRAGMENTO})")
print(f"--- registro: {state.calls}")
