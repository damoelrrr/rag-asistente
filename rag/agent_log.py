"""Registro de uso de tools: una línea JSON por pregunta en logs/agent_usage.jsonl.

Cada línea guarda qué tools llamó el agente, con qué argumentos y cuántas llamadas hizo al LLM
(más tokens y tiempo, que se usan en la comparación de costo).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_run(path: Path, question: str, result) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": question,
        "tools": [{"name": c["name"], "args": c["args"], "resumen": c["resumen"], "bloqueada": c["bloqueada"]}
                  for c in result.tool_calls],
        "llamadas_llm": result.llm_calls,
        "iteraciones": result.iterations,
        "limite_iteraciones_alcanzado": result.limit_reached,
        "tokens_entrada": result.input_tokens,
        "tokens_salida": result.output_tokens,
        "segundos": round(result.seconds, 2),
        "rechazo": result.refused,
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:  # sistemas de archivos de solo lectura (algunas nubes): el registro es opcional
        pass
