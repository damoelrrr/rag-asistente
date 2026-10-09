"""Interfaz de chat (Streamlit) del Asistente Experto RAG.

Ejecutar en local:  streamlit run app.py
"""
# --- Streamlit Cloud trae un SQLite antiguo que Chroma no soporta: usamos pysqlite3 si está instalado ---
try:  # noqa: SIM105
    import pysqlite3  # type: ignore
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import os

import streamlit as st

st.set_page_config(page_title="Asistente RAG", page_icon="🧠", layout="centered", initial_sidebar_state="expanded")

# Los secrets de la nube se pasan a variables de entorno (las claves NUNCA van en el repositorio)
try:
    for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "TOP_K", "MAX_DISTANCE"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:  # sin secrets.toml en local: se usa .env
    pass

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.prompts import REFUSAL  # noqa: E402

# ------------------------------------------------------------------ Estilos (principios de UX) ----
# Jerarquía visual clara, mucho espacio en blanco, un solo color de acento, contraste AA,
# estados vacíos con ejemplos, feedback inmediato (spinner + streaming) y divulgación progresiva
# (las fuentes están plegadas hasta que el usuario las quiere ver).
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 780px; }

.hero { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #fff; border-radius: 20px;
        padding: 26px 28px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(79,70,229,.25); }
.hero h1 { font-size: 1.55rem; margin: 0 0 6px 0; font-weight: 700; color: #fff; }
.hero p { margin: 0; opacity: .92; font-size: .97rem; line-height: 1.5; color: #fff; }
.hero .pill { display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px; padding: 3px 11px;
              font-size: .74rem; margin: 12px 6px 0 0; }

.empty-title { font-weight: 600; color: #1e2433; margin: 6px 0 10px 0; }

div[data-testid="stChatMessage"] { background: #fff; border: 1px solid #e8eaf3; border-radius: 16px;
        padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(20,24,40,.04); }
div[data-testid="stChatMessage"] p { line-height: 1.62; }

.stButton > button { border-radius: 12px; border: 1px solid #dfe2f0; background: #fff; color: #1e2433;
        text-align: left; padding: 10px 14px; font-weight: 500; transition: all .15s ease; width: 100%; }
.stButton > button:hover { border-color: #4f46e5; color: #4f46e5; background: #f4f3ff; transform: translateY(-1px); }

.src { border: 1px solid #e8eaf3; border-left: 4px solid #cfd3e6; border-radius: 10px; padding: 10px 14px;
       margin: 8px 0; background: #fafbff; }
.src.cited { border-left-color: #4f46e5; background: #f4f3ff; }
.src .meta { font-size: .78rem; color: #596079; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.src .badge { background: #4f46e5; color: #fff; border-radius: 6px; padding: 1px 8px; font-weight: 600; font-size: .75rem; }
.src .sim { margin-left: auto; font-weight: 600; color: #3a3f55; }
.src .txt { font-size: .86rem; color: #2b3045; margin-top: 6px; line-height: 1.5; }

.notice { background: #fff8eb; border: 1px solid #fde1a8; border-left: 4px solid #f59e0b; border-radius: 12px;
          padding: 12px 16px; color: #6b4a0b; font-size: .95rem; line-height: 1.5; }
.notice b { color: #4a3207; }

section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaf3; }
.side-doc { font-size: .84rem; color: #3a3f55; padding: 3px 0; }
</style>
""",
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "¿Qué es RAG y qué ventajas tiene?",
    "¿En qué se diferencia BM25 de TF-IDF?",
    "¿Qué es la búsqueda híbrida?",
    "¿Cuáles son los componentes de un buen prompt?",
]


# ------------------------------------------------------------------ Recursos ----
@st.cache_resource(show_spinner="Preparando la base de conocimientos…")
def load_pipeline() -> tuple[RAGPipeline, int]:
    pipe = RAGPipeline(Settings())
    return pipe, pipe.ensure_index()


def _key_ok(provider: str) -> bool:
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    else:
        return bool(os.getenv("OPENAI_API_KEY"))


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
        f'<div class="notice"><b>Fuera del alcance de la base de conocimientos</b><br>{html.escape(text)}<br>'
        "<span style='font-size:.88rem'>Prueba reformulando la pregunta o consulta uno de los temas "
        "de la barra lateral.</span></div>"
    )


def render_message(m: dict) -> None:
    if m["role"] == "assistant" and m.get("refused"):
        st.markdown(notice_html(m["content"]), unsafe_allow_html=True)
        return
    st.markdown(m["content"])
    if m.get("sources"):
        with st.expander(f"📚 Fuentes ({len(m['sources'])})"):
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
        f"Falta la clave de API del proveedor **{settings.llm_provider}**. En local créala en `.env` "
        "(ver `.env.example`); en la nube, agrégala en *Secrets* de la plataforma."
    )
    st.stop()

pipe, n_chunks = load_pipeline()

# ------------------------------------------------------------------ Barra lateral ----
with st.sidebar:
    st.markdown("### 🧠 Asistente RAG")
    st.caption("Responde únicamente con lo que dicen los documentos de la base de conocimientos y cita sus fuentes.")
    st.markdown("**Documentos indexados**")
    for f in sorted(p.name for p in settings.corpus_dir.glob("*") if p.is_file()):
        st.markdown(f"<div class='side-doc'>📄 {html.escape(f)}</div>", unsafe_allow_html=True)
    st.divider()
    top_k = st.slider("Fragmentos a recuperar (top_k)", 1, 8, settings.top_k, help="Más fragmentos = más contexto, pero también más ruido.")
    st.caption(f"{n_chunks} fragmentos · embeddings `{settings.embedding_model.split('/')[-1]}`")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("🔒 Tus documentos no se usan para entrenar modelos: al LLM solo se envían los fragmentos recuperados de cada consulta.")

# ------------------------------------------------------------------ Cabecera ----
st.markdown(
    """
<div class="hero">
  <h1>Asistente experto en RAG</h1>
  <p>Pregúntame sobre RAG, recuperación de información (TF-IDF, BM25, búsqueda híbrida), ingeniería de prompts y LLMs locales.
  Cada respuesta incluye las fuentes en las que se apoya.</p>
  <span class="pill">Conversacional</span><span class="pill">Con fuentes citadas</span><span class="pill">Sin inventar</span>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Historial / estado vacío ----
if not st.session_state.messages:
    st.markdown("<div class='empty-title'>Prueba con una de estas preguntas:</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sug{i}"):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🧠"):
        render_message(m)

# ------------------------------------------------------------------ Entrada ----
typed = st.chat_input("Escribe tu pregunta…")
prompt = st.session_state.pending or typed
st.session_state.pending = None

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🧠"):
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
        if ans.refused and prep.messages is not None:
            pass  # el LLM rechazó por sí mismo: se redibuja abajo con el aviso estilizado
        elif sources:
            with st.expander(f"📚 Fuentes ({len(sources)})"):
                if ans.standalone != prompt:
                    st.caption(f"🔎 Búsqueda realizada: «{ans.standalone}»")
                st.markdown(sources_html(sources), unsafe_allow_html=True)
    if ans.refused and prep.messages is not None:
        st.rerun()
