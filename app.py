"""Interfaz de chat (Streamlit) del Tutor Académico RAG.

Ejecutar en local:  streamlit run app.py
"""
# --- Streamlit Cloud trae un SQLite antiguo que Chroma no soporta: usamos pysqlite3 si está instalado ---
try:  # noqa: SIM105
    import pysqlite3  # type: ignore
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hashlib
import html
import json
import os
import re
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Tutor IA · Tu espacio de estudio",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Los secrets de la nube se pasan a variables de entorno (las claves NUNCA van en el repositorio)
try:
    for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "TOP_K", "MAX_DISTANCE"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k]).strip()  # los secrets mandan sobre valores previos
except Exception:  # sin secrets.toml en local: se usa .env
    pass

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.prompts import REFUSAL  # noqa: E402

# La presentación vive aparte para mantener el flujo RAG independiente del diseño.
st.markdown(
    f"<style>{(Path(__file__).parent / 'assets' / 'chat.css').read_text()}</style>",
    unsafe_allow_html=True,
)

ICONS = {
    "spark": '<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6Z"/><path d="m20 2 .6 1.4L22 4l-1.4.6L20 6l-.6-1.4L18 4l1.4-.6Z"/>',
    "book": '<path d="M12 5v15M3 4c3-1 6 0 9 2 3-2 6-3 9-2v14c-3-1-6 0-9 2-3-2-6-3-9-2Z"/>',
    "shield": '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8 12 3 3 5-6"/>',
    "source": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    "lock": '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
}


def icon(name: str) -> str:
    return f'<svg class="ui-icon" viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg>'


SUGGESTIONS = [
    ("Explora un concepto", "Entiende RAG, paso a paso.", "¿Qué es RAG y qué ventajas tiene?", "lightbulb"),
    ("Conecta las ideas", "Compara BM25 y TF-IDF.", "¿En qué se diferencia BM25 de TF-IDF?", "hub"),
    ("Pon a prueba lo que sabes", "Un pequeño quiz, un gran avance.", "Hazme un quiz de 3 preguntas sobre prompt engineering", "quiz"),
    ("Hazlo más sencillo", "Empieza desde lo esencial.", "No entiendo nada: ¿qué es la búsqueda híbrida?", "auto_awesome"),
]

DOCUMENT_TITLES = {
    "2312.10997v5.pdf": "Retrieval-Augmented Generation for LLMs",
    "3 BM25_Score.pdf": "BM25 · Relevancia en la búsqueda",
    "Ejecucin_de_LLMs_con_Ollama.pdf": "LLMs locales con Ollama",
    "El_Mtodo_Socrtico_El_Arte_de_Preguntar (3).pdf": "El método socrático · El arte de preguntar",
    "Introducción a RAG.pdf": "Introducción a RAG",
    "Introducción_Ingeniería_Prompt (3).pdf": "Introducción a la ingeniería de prompts",
    "TF-IDF_Keyword_search.pdf": "TF-IDF · Búsqueda por palabras clave",
}


def open_question(question: str) -> None:
    st.session_state.pending = question
    st.session_state.view = "chat"


# ------------------------------------------------------------------ Recursos ----
@st.cache_resource(show_spinner="Preparando la base de conocimientos…")
def load_pipeline(llm_signature: str, retrieval_signature: str) -> tuple[RAGPipeline, int]:
    """Renueva el recurso si cambia el LLM o el código de recuperación y sus prompts."""
    pipe = RAGPipeline(Settings())
    return pipe, pipe.ensure_index()


def _key_ok(provider: str) -> bool:
    env = {"anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    return True if provider == "ollama" else bool(env and os.getenv(env))


def parse_quiz(text: str):
    """Si la respuesta es un quiz en JSON (formato del Avance 1) devuelve el dict; si no, None."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    if not t.startswith("{"):
        return None
    try:
        d = json.loads(t)
    except ValueError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("preguntas"), list) else None


def render_quiz(d: dict) -> None:
    st.markdown(f"**📝 Quiz: {d.get('tema', 'Repaso')}**")
    for q in d["preguntas"]:
        st.markdown(f"**{q.get('id', '')}. {q.get('enunciado', '')}**")
        for o in q.get("opciones", []):
            st.markdown(f"- {o}")
        with st.expander("Ver respuesta"):
            st.markdown(f"✅ **{q.get('respuesta_correcta', '')}**")
            if q.get("fuente"):
                st.caption(f"Fuente: {q['fuente']}")


def sources_html(sources: list[dict]) -> str:
    out = []
    for s in sources:
        where = s["source"] + (f" · pág. {s['page']}" if s.get("page") else "") + f" · fragmento {s['chunk_index']}"
        cls = "src cited" if s["cited"] else "src"
        tag = " · citada en la respuesta" if s["cited"] else ""
        out.append(
            f'<div class="{cls}"><div class="meta"><span class="badge">{s["n"]}</span>'
            f'<span>{html.escape(where)}{tag}</span><span class="sim">{s["similarity"]:.0%} similitud</span></div>'
            f'<div class="txt">{html.escape(s["text"])}</div></div>'
        )
    return "".join(out)


def notice_html(text: str) -> str:
    return (
        f'<div class="notice"><b>No encontré contexto suficiente para responder</b><br>{html.escape(text)}<br>'
        "<span style='font-size:.88rem'>Revisa las fuentes consultadas, reformula tu pregunta "
        "o visita la biblioteca del curso.</span></div>"
    )


def render_message(m: dict) -> None:
    if m["role"] == "assistant" and m.get("refused"):
        st.markdown(notice_html(m["content"]), unsafe_allow_html=True)
    else:
        quiz = parse_quiz(m["content"]) if m["role"] == "assistant" else None
        if quiz:
            render_quiz(quiz)
        else:
            st.markdown(m["content"])
    if m.get("sources"):
        with st.expander(f"Fuentes consultadas · {len(m['sources'])}"):
            if m.get("standalone") and m["standalone"] != m.get("question"):
                st.caption(f"🔎 Búsqueda realizada: «{m['standalone']}»")
            st.markdown(sources_html(m["sources"]), unsafe_allow_html=True)


# ------------------------------------------------------------------ Estado ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

settings = Settings()
if not _key_ok(settings.llm_provider.lower()):
    st.error(
        f"Falta la clave del proveedor **{settings.llm_provider}**. Consíguela gratis en https://console.groq.com/keys "
        "y pégala en el archivo `.env` (ver `.env.example`); en la nube, agrégala en *Secrets*. "
        "Si prefieres no usar claves, instala Ollama y pon `LLM_PROVIDER=ollama` en `.env`."
    )
    st.stop()

_provider = settings.llm_provider.lower()
_env = {"anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}.get(_provider, "OPENAI_API_KEY")
_key = os.getenv(_env, "")
_rag_dir = Path(__file__).parent / "rag"
_retrieval_signature = hashlib.sha256(b"".join(
    (_rag_dir / name).read_bytes() for name in ("pipeline.py", "vectorstore.py", "retrieval.py", "prompts.py")
)).hexdigest()[:12]
pipe, n_chunks = load_pipeline(
    f"{_provider}|{settings.llm_model}|{hashlib.sha256(_key.encode()).hexdigest()[:12]}", _retrieval_signature,
)

# ------------------------------------------------------------------ Espacio de estudio ----
if "view" not in st.session_state:
    st.session_state.view = "chat"
documents = sorted(p for p in settings.corpus_dir.glob("*") if p.is_file())

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-symbol">{icon("spark")}</div>'
        '<div><div class="brand-name">Tutor<span> IA</span></div>'
        '<div class="brand-caption">Tu espacio de estudio</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Nueva conversación", icon=":material/add:", key="new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.view = "chat"
        st.rerun()

    st.markdown('<div class="rail-label" style="margin-top:0">ESPACIO DE TRABAJO</div>', unsafe_allow_html=True)
    if st.button("Tutor académico", icon=":material/chat_bubble_outline:", key="nav_chat",
                 type="primary" if st.session_state.view == "chat" else "secondary", use_container_width=True):
        st.session_state.view = "chat"
        st.rerun()
    if st.button("Biblioteca del curso", icon=":material/book_2:", key="nav_library",
                 type="primary" if st.session_state.view == "library" else "secondary", use_container_width=True):
        st.session_state.view = "library"
        st.rerun()

    # El espacio central de la barra queda disponible para el futuro historial de chats.
    with st.container(key="sidebar_footer"):
        with st.expander("Ajustes de búsqueda", icon=":material/tune:"):
            top_k = st.slider("Fragmentos por consulta", 1, 8, min(8, max(1, settings.top_k)),
                              help="Un fragmento (chunk) es una porción del documento. Este es el máximo que recibe el modelo después de ordenar y filtrar los candidatos; pueden llegar menos si no hay suficiente contexto relevante.")
            st.caption(f"{len(documents)} documentos · {n_chunks} fragmentos indexados")
            st.caption(f"Fragmentos de hasta {settings.chunk_size} caracteres · solapamiento de hasta {settings.chunk_overlap}")
            st.caption(f"Embeddings locales · {settings.embedding_model.split('/')[-1]}")
            st.caption("Al modelo solo se envían los fragmentos recuperados para responder tu pregunta.")
        st.markdown(
            f'<div class="profile"><div class="profile-avatar">E</div>'
            '<div><div class="profile-name">Estudiante</div><div class="profile-subtitle">La curiosidad es tu superpoder</div></div>'
            f'{icon("spark")}</div>', unsafe_allow_html=True,
        )

# ------------------------------------------------------------------ Cabecera ----
current_view = "Biblioteca del curso" if st.session_state.view == "library" else "Tutor académico"
st.markdown(
    f'<div class="workspace-header"><div class="breadcrumb">{icon("book")} Mi espacio'
    f'<span class="separator">/</span><strong>{current_view}</strong></div>'
    '<div class="workspace-badge"><span class="status-dot"></span>Desarrollo de aplicaciones con IA</div></div>',
    unsafe_allow_html=True,
)

if st.session_state.view == "library":
    st.markdown(
        '<div class="library-heading"><div class="eyebrow">TU BASE DE CONOCIMIENTO</div>'
        '<h1>Las fuentes de tus ideas.</h1>'
        f'<p>{len(documents)} documentos del curso, conectados a tu conversación. '
        'Tu tutor consulta este material para ayudarte a aprender con fundamento.</p></div>', unsafe_allow_html=True,
    )
    if not documents:
        st.info("Todavía no hay documentos en la biblioteca del curso.")
    for document in documents:
        title = DOCUMENT_TITLES.get(document.name, document.stem.replace("_", " "))
        st.markdown(
            f'<div class="document-row"><div class="document-icon">{icon("source")}</div>'
            f'<div class="document-info"><strong>{html.escape(title)}</strong>'
            f'<span>{html.escape(document.name)} · {document.stat().st_size / 1024:.0f} KB</span></div>'
            f'<span class="document-format">{html.escape(document.suffix[1:].upper())}</span></div>',
            unsafe_allow_html=True,
        )
    if st.button("Volver a mi conversación", icon=":material/arrow_back:"):
        st.session_state.view = "chat"
        st.rerun()
    st.stop()

# ------------------------------------------------------------------ Historial / bienvenida ----
if not st.session_state.messages and not st.session_state.pending:
    st.markdown(
        f'<div class="welcome"><div class="orbit"><div class="orbit-core">{icon("spark")}</div></div>'
        '<div class="eyebrow">UN POCO DE CURIOSIDAD. INFINITAS POSIBILIDADES.</div>'
        '<h1>Las grandes ideas empiezan<br><span class="heading-accent">con una buena pregunta.</span></h1>'
        '<p>Tu compañero para entender, conectar ideas y llegar más lejos.<br>'
        'Exploremos juntos el mundo de la inteligencia artificial.</p>'
        f'<div class="capabilities"><span>{icon("spark")} Aprendizaje socrático</span>'
        f'<span>{icon("source")} Respuestas con fuentes</span>'
        f'<span>{icon("shield")} Basado en tu curso</span></div></div>', unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="suggestion-heading"><span>Un buen lugar para empezar</span>'
        '<span>Elige una idea y hazla tuya ↘</span></div>', unsafe_allow_html=True,
    )
    with st.container(key="suggestion_grid"):
        for row in range(2):
            cols = st.columns(2)
            for column in range(2):
                i = row * 2 + column
                title, description, question, material_icon = SUGGESTIONS[i]
                cols[column].button(f"**{title}**  \n{description}", icon=f":material/{material_icon}:",
                                    key=f"sug{i}", use_container_width=True,
                                    on_click=open_question, args=(question,))
    st.markdown(
        f'<div class="quiet-note">{icon("lock")} Tus preguntas abren caminos. Tus documentos no se usan para entrenar modelos.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="conversation-heading"><h1>Sigamos el hilo de tus ideas.</h1>'
        '<p>Pregunta, explora y construye tu propia respuesta.</p></div>', unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/auto_awesome:"):
        render_message(m)

# ------------------------------------------------------------------ Entrada ----
typed = st.chat_input("Pregunta, conecta ideas, descubre algo nuevo…")
prompt = st.session_state.pending or typed
st.session_state.pending = None

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        try:
            with st.spinner("Buscando en la base de conocimientos…"):
                prep = pipe.prepare(prompt, history, top_k)
            if prep.messages is None:  # nada pertinente: se rechaza sin gastar una llamada al LLM
                text = REFUSAL
                st.markdown(notice_html(text), unsafe_allow_html=True)
            else:
                text = st.write_stream(pipe.stream(prep))
            ans = pipe.finish(prep, text)
        except Exception as e:  # clave inválida, red, límites de uso…
            st.error(f"No pude generar la respuesta ({type(e).__name__}): {e}")
            st.session_state.messages.pop()
            st.stop()

        sources = [
            {"n": h.n, "source": h.source, "page": h.page, "chunk_index": h.chunk_index,
             "similarity": h.similarity, "text": h.text, "cited": h.n in ans.cited}
            for h in ans.hits
        ]
        msg = {"role": "assistant", "content": ans.text, "sources": sources, "refused": ans.refused,
               "standalone": ans.standalone, "question": prompt}
        st.session_state.messages.append(msg)
    # Actualiza también la cabecera y la conversación de la barra lateral tras el streaming.
    st.rerun()
