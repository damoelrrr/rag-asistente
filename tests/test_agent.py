"""Pruebas del agente SIN LLM real, SIN red y SIN índice: tools, límites, política web y bucle."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from rag.agent import Agent
from rag.config import Settings
from rag.tools import (DOMINIOS_PERMITIDOS, MAX_CHARS_CONSULTA_WEB, MAX_CHARS_FRAGMENTO, MAX_FRAGMENTOS,
                       MAX_RESULTADOS_WEB, RunState, build_tools, resolve_document)
from rag.vectorstore import Hit


# ------------------------------------------------------------------ dobles de prueba
class FakePipe:
    def __init__(self, settings, hits=None):
        self.s = settings
        self._hits = hits if hits is not None else []
        self.store = SimpleNamespace(catalogue=lambda: [Hit(1, "x", "3 BM25_Score.pdf", 1, 0, None)])
        self.llm = None

    def retrieve(self, query, top_k=None):
        return list(self._hits), list(self._hits)


class FakeLLM:
    """Responde con una secuencia guionada de AIMessage; cuenta las llamadas."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.script.pop(0) if self.script else AIMessage(content="fin")


def ai(text="", calls=None):
    tool_calls = [{"name": n, "args": a, "id": f"id{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls or [])]
    return AIMessage(content=text, tool_calls=tool_calls,
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), agent_log_path=tmp_path / "log.jsonl", agent_max_iterations=3)


def long_hits(n=10):
    return [Hit(i + 1, "texto " * 400, f"doc{i}.pdf", i + 1, i, 0.2) for i in range(n)]


def tools_for(pipe, web=None):
    state = RunState()
    kw = {"web_search": web} if web else {}
    return {t.name: t for t in build_tools(pipe, state, **kw)}, state


# ------------------------------------------------------------------ tool de búsqueda (sin LLM)
def test_buscar_en_base_respeta_limites_y_cita(settings):
    tools, state = tools_for(FakePipe(settings, long_hits(10)))
    out = tools["buscar_en_base"].invoke({"consulta": "qué es bm25"})
    assert len(state.base_sources) == MAX_FRAGMENTOS
    assert all(len(s["text"]) <= MAX_CHARS_FRAGMENTO for s in state.base_sources)
    assert "(Fuente: doc0.pdf, pág. 1)" in out  # documento + página para poder citar
    assert state.calls[0]["args"] == {"consulta": "qué es bm25"}


def test_buscar_en_base_sin_resultados(settings):
    tools, _ = tools_for(FakePipe(settings, []))
    assert tools["buscar_en_base"].invoke({"consulta": "algo"}).startswith("SIN_RESULTADOS")


def test_docstrings_describen_cuando_usar_cada_tool(settings):
    tools, _ = tools_for(FakePipe(settings))
    assert "Úsala" in tools["buscar_en_base"].description
    assert "primero la base" in tools["buscar_en_web"].description.lower()


# ------------------------------------------------------------------ tarea del dominio (APA)
@pytest.mark.parametrize("pedido, esperado", [
    ("Introducción a RAG", "Introducción a RAG"),
    ("el artículo de Gao et al. sobre RAG", "Gao, Y."),
    ("método bm25", "BM25"),
    ("2312.10997", "arXiv:2312.10997v5"),
])
def test_referencia_apa(settings, pedido, esperado):
    tools, _ = tools_for(FakePipe(settings))
    assert esperado in tools["generar_referencia_apa"].invoke({"documento": pedido})


def test_referencia_apa_documento_inexistente(settings):
    tools, _ = tools_for(FakePipe(settings))
    assert tools["generar_referencia_apa"].invoke({"documento": "receta de arepas"}).startswith("SIN_RESULTADOS")
    assert resolve_document("") is None


# ------------------------------------------------------------------ web controlada
def test_web_bloqueada_si_no_se_consulto_la_base(settings):
    tools, state = tools_for(FakePipe(settings), web=lambda q: pytest.fail("no debió llamar al buscador"))
    assert tools["buscar_en_web"].invoke({"consulta": "langgraph"}).startswith("POLÍTICA")
    assert state.calls[-1]["bloqueada"] is True


def test_web_filtra_dominios_y_limita_resultados(settings):
    visto = {}

    def fake_web(query):
        visto["q"] = query
        urls = ["https://arxiv.org/abs/1", "https://malicioso.com/x", "https://docs.ollama.com/a",
                "https://ollama.com/b", "https://scikit-learn.org/c"]
        return [{"title": f"T{i}", "url": u, "content": "x" * 900} for i, u in enumerate(urls)]

    tools, state = tools_for(FakePipe(settings), web=fake_web)
    tools["buscar_en_base"].invoke({"consulta": "algo"})
    out = tools["buscar_en_web"].invoke({"consulta": "q" * 500})
    assert len(visto["q"]) == MAX_CHARS_CONSULTA_WEB  # lo único que sale del sistema
    assert len(state.web_sources) == MAX_RESULTADOS_WEB
    assert all(any(s["url"].split("/")[2].endswith(d) for d in DOMINIOS_PERMITIDOS) for s in state.web_sources)
    assert "malicioso.com" not in out and "origen: web" in out
    assert all(len(s["excerpt"]) <= 300 for s in state.web_sources)


def test_web_sin_clave_no_rompe(settings):
    def boom(q):
        raise RuntimeError("sin clave")

    tools, _ = tools_for(FakePipe(settings), web=boom)
    tools["buscar_en_base"].invoke({"consulta": "algo"})
    assert tools["buscar_en_web"].invoke({"consulta": "algo"}).startswith("ERROR_WEB")


# ------------------------------------------------------------------ bucle del agente
def make_agent(settings, script, hits=None, web=None):
    pipe = FakePipe(settings, hits if hits is not None else long_hits(2))
    pipe.llm = FakeLLM(script)
    return Agent(pipe, settings, web_search=web), pipe.llm


def test_saludo_no_llama_tools(settings):
    agent, llm = make_agent(settings, [ai("¡Hola! ¿En qué te ayudo?")])
    r = agent.run("Hola")
    assert r.tool_calls == [] and r.llm_calls == 1 and llm.calls == 1
    assert r.input_tokens == 100 and r.output_tokens == 10


def test_pregunta_con_base_hace_dos_llamadas_y_registra(settings):
    agent, _ = make_agent(settings, [ai(calls=[("buscar_en_base", {"consulta": "bm25"})]), ai("BM25 es… (Fuente: doc0.pdf, pág. 1)")])
    r = agent.run("¿Qué es BM25?")
    assert r.llm_calls == 2 and [c["name"] for c in r.tool_calls] == ["buscar_en_base"]
    assert r.contexts and not r.web_sources
    line = json.loads(settings.agent_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert line["pregunta"] == "¿Qué es BM25?" and line["llamadas_llm"] == 2
    assert line["tools"][0]["args"] == {"consulta": "bm25"}


def test_pregunta_compuesta_busca_dos_veces(settings):
    script = [ai(calls=[("buscar_en_base", {"consulta": "rag"}), ("buscar_en_base", {"consulta": "temperatura"})]), ai("ok")]
    agent, _ = make_agent(settings, script)
    r = agent.run("¿Qué es RAG y qué es la temperatura?")
    assert [c["name"] for c in r.tool_calls].count("buscar_en_base") == 2 and r.llm_calls == 2


def test_limite_de_iteraciones(settings):
    pedir = lambda: ai(calls=[("buscar_en_base", {"consulta": "x"})])
    agent, llm = make_agent(settings, [pedir(), pedir(), pedir(), ai("respuesta forzada")])
    r = agent.run("pregunta infinita")
    assert r.limit_reached and r.iterations == settings.agent_max_iterations
    assert r.llm_calls == settings.agent_max_iterations + 1 and r.text == "respuesta forzada"


def test_tool_desconocida_y_argumentos_invalidos_no_rompen(settings):
    script = [ai(calls=[("borrar_todo", {}), ("buscar_en_base", {"mal": 1})]), ai("sigo vivo")]
    agent, _ = make_agent(settings, script)
    r = agent.run("hazlo")
    assert r.text == "sigo vivo" and len(r.tool_calls) == 2


def test_web_antes_que_base_queda_bloqueada(settings):
    script = [ai(calls=[("buscar_en_web", {"consulta": "langgraph"})]), ai("no puedo")]
    agent, _ = make_agent(settings, script, web=lambda q: pytest.fail("no debió buscar"))
    r = agent.run("¿Qué es LangGraph?")
    assert r.tool_calls[0]["bloqueada"] is True and not r.web_sources


def test_respuesta_vacia_se_convierte_en_refusal(settings):
    agent, _ = make_agent(settings, [ai("")])
    r = agent.run("hola")
    assert r.refused


def test_etiquetas_internas_se_convierten_en_citas_con_origen():
    from rag.agent import resolve_citation_labels
    base = [{"n": "B1", "source": "3 BM25_Score.pdf", "page": 4, "chunk_index": 2}]
    web = [{"n": "W1", "url": "https://docs.langchain.com/oss/python/langgraph/overview"}]
    out = resolve_citation_labels("BM25 satura el TF [B1]. LangGraph orquesta agentes [W1]. Otro [W9].", base, web)
    assert "(Fuente: 3 BM25_Score.pdf, pág. 4)" in out
    assert "[web: https://docs.langchain.com/oss/python/langgraph/overview]" in out
    assert "[W9]" in out  # una etiqueta que no existe no se inventa


def test_respuesta_web_que_cita_no_es_rechazo(settings):
    llm = FakeLLM([ai("Según la base de conocimientos: no encontré información sobre LangGraph en los documentos del curso.\n\n"
                      "Según la web: es un framework de orquestación [web: https://docs.langchain.com/x].")])
    pipe = FakePipe(settings)
    pipe.llm = llm
    res = Agent(pipe, settings).run("¿Qué es LangGraph?", log=False)
    assert not res.refused
