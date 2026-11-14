"""Casos de comportamiento del agente (Avance 3).

El comportamiento esperado de cada caso está escrito ANTES de ejecutar, en eval/behavior_cases.json
(haz commit de ese archivo antes de correr este script). Por cada caso se registra qué tools llamó el agente,
con qué argumentos, cuántas llamadas hizo al LLM y si cumplió lo esperado (verificación automática de las
condiciones de "checks"; revisa además la respuesta a mano antes de poner el resultado en el PDF).

Uso:  python eval/run_behavior_cases.py
Salida: eval/results/behavior_results.csv y behavior_results.md
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rag.agent import Agent  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.tools import DOMINIOS_PERMITIDOS  # noqa: E402

HERE = Path(__file__).resolve().parent


def evaluate_case(checks: dict, res) -> list[str]:
    """Devuelve la lista de condiciones incumplidas (vacía = cumple)."""
    fails = []
    names = [c["name"] for c in res.tool_calls if not c["bloqueada"]]
    low = res.text.lower()
    for t in checks.get("tools_ausentes", []):
        if t in names:
            fails.append(f"no debía llamar {t}")
    for t, n in checks.get("min_llamadas_tool", {}).items():
        if names.count(t) < n:
            fails.append(f"esperaba ≥{n} llamada(s) a {t}, hubo {names.count(t)}")
    if "orden" in checks:
        pos = [names.index(t) if t in names else -1 for t in checks["orden"]]
        if -1 in pos or pos != sorted(pos):
            fails.append(f"orden esperado {checks['orden']}")
    for t, words in checks.get("args_contienen", {}).items():
        args = " ".join(json.dumps(c["args"], ensure_ascii=False).lower() for c in res.tool_calls if c["name"] == t)
        for w in words:
            if w.lower() not in args:
                fails.append(f"el argumento de {t} no contiene «{w}»")
    if "max_llamadas_llm" in checks and res.llm_calls > checks["max_llamadas_llm"]:
        fails.append(f"{res.llm_calls} llamadas al LLM (máx. {checks['max_llamadas_llm']})")
    if checks.get("respuesta_contiene_todos") and not all(w.lower() in low for w in checks["respuesta_contiene_todos"]):
        fails.append(f"la respuesta debe contener todo: {checks['respuesta_contiene_todos']}")
    if checks.get("respuesta_contiene_alguno") and not any(w.lower() in low for w in checks["respuesta_contiene_alguno"]):
        fails.append(f"la respuesta debe contener alguno de: {checks['respuesta_contiene_alguno']}")
    if any(w.lower() in low for w in checks.get("respuesta_no_contiene", [])):
        fails.append("la respuesta contiene texto prohibido")
    if "max_chars_respuesta" in checks and len(res.text) > checks["max_chars_respuesta"]:
        fails.append(f"respuesta de {len(res.text)} caracteres (máx. {checks['max_chars_respuesta']})")
    if checks.get("es_rechazo") and not res.refused:
        fails.append("debía responder con el texto de rechazo exacto")
    if checks.get("urls_web_en_dominios_permitidos"):
        bad = [s["url"] for s in res.web_sources
               if not any((urlparse(s["url"]).hostname or "").endswith(d) for d in DOMINIOS_PERMITIDOS)]
        if bad or not res.web_sources:
            fails.append(f"URLs web fuera de la lista permitida o sin resultados: {bad}")
    return fails


def main():
    cases = json.loads((HERE / "behavior_cases.json").read_text(encoding="utf-8"))
    pipe = RAGPipeline(Settings())
    pipe.ensure_index()
    agent = Agent(pipe)
    rows = []
    for c in cases:
        t0 = time.time()
        res = agent.run(c["pregunta"])
        fails = evaluate_case(c["checks"], res)
        tools = "; ".join(f"{x['name']}({json.dumps(x['args'], ensure_ascii=False)})" + (" [bloqueada]" if x["bloqueada"] else "")
                          for x in res.tool_calls) or "ninguna"
        rows.append({
            "id": c["id"], "tipo": c["tipo"], "pregunta": c["pregunta"], "esperado": c["esperado"],
            "obtenido": res.text.replace("\n", " ")[:300] + ("…" if len(res.text) > 300 else ""),
            "tools_usadas": tools, "llamadas_llm": res.llm_calls, "segundos": round(time.time() - t0, 1),
            "cumple": "sí" if not fails else "NO", "detalle_incumplimiento": " | ".join(fails),
        })
        print(f"{c['id']} {'OK ' if not fails else 'FALLA'} | tools: {tools} | LLM: {res.llm_calls} | {fails or ''}")
    df = pd.DataFrame(rows)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "behavior_results.csv", index=False, encoding="utf-8-sig")
    md = ["| Caso | Tipo | Esperado | Obtenido | Tools usadas | Llamadas LLM | ¿Cumple? |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        cell = lambda s: str(s).replace("|", "\\|")
        md.append(f"| {r['id']} | {cell(r['tipo'])} | {cell(r['esperado'])} | {cell(r['obtenido'])} | {cell(r['tools_usadas'])} | "
                  f"{r['llamadas_llm']} | {r['cumple']} |")
    (out / "behavior_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nCumplen {int((df.cumple == 'sí').sum())}/{len(df)}. Archivos: eval/results/behavior_results.csv|md")


if __name__ == "__main__":
    main()
