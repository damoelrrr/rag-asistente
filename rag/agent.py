"""Agente con herramientas (Avance 3): el LLM decide qué tool usar y el programa la ejecuta.

Flujo de una pregunta:
    pregunta -> LLM --(tool_calls)--> el programa ejecuta cada tool -> resultado como ToolMessage -> LLM -> ... -> respuesta

Garantías que viven en el código (no en el prompt):
  * máximo ``max_iterations`` vueltas al LLM por pregunta (después se fuerza una respuesta sin tools);
  * máximo ``MAX_TOOLS_POR_VUELTA`` llamadas a tools en una misma vuelta;
  * cada tool aplica sus propios límites (rag/tools.py);
  * cada pregunta se registra (rag/agent_log.py).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, ToolMessage

from .agent_log import log_run
from .agent_prompts import build_agent_messages
from .config import Settings
from .llm import text_of
from .prompts import REFUSAL, is_refusal
from .tools import RunState, build_tools

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

MAX_TOOLS_POR_VUELTA = 3


@dataclass
class AgentResult:
    text: str
    refused: bool
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    iterations: int = 0
    limit_reached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    base_sources: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # lo que devolvieron las tools (para Ragas)


def usage_of(message) -> tuple[int, int]:
    """(tokens de entrada, tokens de salida) de una respuesta del LLM, según lo que reporte el proveedor."""
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    usage = (getattr(message, "response_metadata", None) or {}).get("token_usage", {})
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def resolve_citation_labels(text: str, base_sources: list[dict], web_sources: list[dict]) -> str:
    """Si el LLM citó con la etiqueta interna ([B1], [W2]), la cambia por la cita con origen.

    Así la marca de origen -(Fuente: archivo, pág. N) o [web: URL]- queda garantizada por el código.
    """
    cites = {}
    for s in base_sources:
        where = f"pág. {s['page']}" if s.get("page") else f"fragmento {s.get('chunk_index')}"
        cites[s["n"]] = f"(Fuente: {s['source']}, {where})"
    for s in web_sources:
        cites[s["n"]] = f"[web: {s['url']}]"
    return re.sub(r"\[([BW]\d+)\]", lambda m: cites.get(m.group(1), m.group(0)), text)


def _is_bad_tool_call(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tool_use_failed" in text or "failed to call a function" in text


class Agent:
    def __init__(self, pipe: "RAGPipeline", settings: Settings | None = None, web_search=None):
        self.pipe = pipe
        self.s = settings or pipe.s
        self._web_search = web_search  # inyectable en pruebas; None = Tavily

    def run(self, question: str, history: list[dict] | None = None, log: bool = True) -> AgentResult:
        t0 = time.time()
        history = (history or [])[-self.s.max_history_messages:]
        state = RunState()
        kwargs = {"web_search": self._web_search} if self._web_search else {}
        tools = build_tools(self.pipe, state, **kwargs)
        registry = {t.name: t for t in tools}
        llm = self.pipe.llm
        llm_tools = llm.bind_tools(tools)
        messages = build_agent_messages(question, history)
        res = AgentResult(text="", refused=False)

        def call(model, msgs):
            """Una llamada al LLM, contada. Reintenta una vez si el proveedor rechazó una llamada a tool mal formada."""
            res.llm_calls += 1
            try:
                reply = model.invoke(msgs)
            except Exception as exc:
                if not _is_bad_tool_call(exc):
                    raise
                messages.append(HumanMessage(content="Tu llamada a la herramienta tenía un formato inválido. "
                                                     "Vuelve a intentarlo con los argumentos correctos."))
                return None
            tin, tout = usage_of(reply)
            res.input_tokens += tin
            res.output_tokens += tout
            return reply

        final = None
        for iteration in range(1, self.s.agent_max_iterations + 1):
            res.iterations = iteration
            reply = call(llm_tools, messages)
            if reply is None:  # llamada inválida: consume la iteración y reintenta
                continue
            messages.append(reply)
            if not reply.tool_calls:  # el LLM decidió responder
                final = reply
                break
            for i, tc in enumerate(reply.tool_calls):
                messages.append(self._execute(tc, i, registry, state))

        if final is None:  # límite de iteraciones: se obliga a responder sin más tools
            res.limit_reached = True
            messages.append(HumanMessage(content="Has alcanzado el límite de herramientas para esta pregunta. "
                                                 "Responde ahora solo con lo que ya obtuviste. Si no es suficiente, "
                                                 "responde exactamente con el texto de la regla 6."))
            final = call(llm, messages)

        text = text_of(final.content).strip() if final is not None else ""
        res.text = resolve_citation_labels(text, state.base_sources, state.web_sources) or REFUSAL
        # Una respuesta que cita alguna fuente no es un rechazo, aunque diga que una parte no está en la base.
        cites_source = "(Fuente:" in res.text or "[web:" in res.text
        res.refused = is_refusal(res.text) and not cites_source
        res.tool_calls = state.calls
        res.base_sources = state.base_sources
        res.web_sources = state.web_sources
        res.contexts = state.contexts
        res.seconds = time.time() - t0
        if log:
            log_run(self.s.agent_log_path, question, res)
        return res

    @staticmethod
    def _execute(tc: dict, index: int, registry: dict, state: RunState) -> ToolMessage:
        """Ejecuta UNA tool pedida por el LLM; siempre devuelve un ToolMessage (el proveedor lo exige)."""
        name, args, call_id = tc.get("name", ""), tc.get("args") or {}, tc.get("id", f"call_{index}")
        if index >= MAX_TOOLS_POR_VUELTA:
            state.log_call(name, args, "omitida: límite de tools por vuelta", bloqueada=True)
            return ToolMessage(content="POLÍTICA: demasiadas herramientas en una sola vuelta; esta se omitió.", tool_call_id=call_id)
        tool = registry.get(name)
        if tool is None:
            state.log_call(name, args, "tool desconocida", bloqueada=True)
            return ToolMessage(content=f"ERROR: la herramienta {name!r} no existe. Disponibles: {', '.join(registry)}.", tool_call_id=call_id)
        try:
            out = tool.invoke(args)
        except Exception as exc:  # argumentos inválidos, fallo interno…
            state.log_call(name, args, f"error: {type(exc).__name__}")
            return ToolMessage(content=f"ERROR: {type(exc).__name__}: {exc}", tool_call_id=call_id)
        return ToolMessage(content=str(out), tool_call_id=call_id)
