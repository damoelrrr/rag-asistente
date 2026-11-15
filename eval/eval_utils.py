"""Utilidades comunes de evaluación: conteo de llamadas/tokens, reintentos por cuota y avance guardado por pregunta."""
import json
import re
import time
from pathlib import Path


def usage_of(message) -> tuple[int, int]:
    """(tokens de entrada, tokens de salida) que reporta el proveedor en una respuesta del LLM."""
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    usage = (getattr(message, "response_metadata", None) or {}).get("token_usage", {})
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


class CountingLLM:
    """Envuelve el LLM del pipeline fijo para contar llamadas y tokens (el pipeline no los expone)."""

    def __init__(self, llm):
        self._llm = llm
        self.reset()

    def reset(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def invoke(self, *args, **kwargs):
        reply = self._llm.invoke(*args, **kwargs)
        self.calls += 1
        tin, tout = usage_of(reply)
        self.input_tokens += tin
        self.output_tokens += tout
        return reply

    def __getattr__(self, name):
        return getattr(self._llm, name)


def with_quota_retry(fn, *args, tries: int = 12, **kwargs):
    """Reintenta cuando el plan gratuito de Groq agota los tokens por día/minuto (espera lo que indica el error)."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # groq.RateLimitError (o envuelto por LangChain)
            msg = str(exc)
            if "rate_limit" not in msg and "429" not in msg:
                raise
            m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
            wait = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))) if m else 60
            print(f"    límite de Groq: espero {wait:.0f} s (intento {attempt + 1}/{tries})", flush=True)
            time.sleep(wait + 5)
    raise RuntimeError("Groq siguió sin cuota después de varios reintentos")


def checkpointed(path: Path, items, run_one):
    """Ejecuta run_one(item) por cada item y guarda cada fila al terminarla; si se corta, retoma donde quedó."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            if item["id"] in done:
                continue
            row = run_one(item)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[row["id"]] = row
    return [done[i["id"]] for i in items]
