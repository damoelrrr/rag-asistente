# 🧠 Asistente Experto basado en RAG

Asistente conversacional que responde **únicamente** con la información de una base de conocimientos propia y **cita la fuente** (documento y fragmento) de cada respuesta. Cuando la información no está en el corpus, lo dice en lugar de inventar.

> **URL pública de la aplicación:** `https://TU-APP.streamlit.app`  ← *(reemplazar tras el despliegue)*
> **Repositorio:** `https://github.com/TU-USUARIO/TU-REPO`  ← *(reemplazar)*

**Privacidad:** el sistema conserva el control de la base de conocimientos y del índice vectorial. Los documentos no se entregan a servicios de entrenamiento; los embeddings se calculan **localmente** y al LLM solo se le envían los fragmentos recuperados para cada consulta, nunca el corpus completo.

---

## 1. Flujo RAG implementado

```
  Documentos (PDF/DOCX/TXT/MD/HTML)
        │  1. INGESTA          rag/ingest.py
        ▼
  Texto por página/documento
        │  2. CHUNKING         rag/chunking.py   (800 car., solape 120, corte recursivo)
        ▼
  Chunks + metadatos (fuente, página, nº de fragmento)
        │  3. VECTORIZACIÓN    rag/embeddings.py (paraphrase-multilingual-MiniLM-L12-v2, local)
        ▼
  Índice persistente Chroma (coseno)   rag/vectorstore.py   ──►  data/index/
        ▲
        │  5. RECUPERACIÓN     top_k + umbral de distancia
  Pregunta ──► 4. REESCRITURA con historial (preguntas de seguimiento)  rag/pipeline.py
        ▼
  6. GENERACIÓN: system prompt + few-shot + <contexto>…</contexto> + <pregunta>   rag/prompts.py
        ▼
  Respuesta con citas [n] + panel de fuentes   app.py
```

| Etapa | Decisión | Justificación |
|---|---|---|
| Corpus | 5 documentos `.md` sobre RAG, chunking, embeddings, vector DBs, Ragas y prompting (`data/corpus/`) | Dominio coherente con el asistente (tutor de RAG); fácil de verificar al evaluar. **Reemplázalos por tu propio corpus** y todo sigue funcionando |
| Chunking | 800 caracteres, solape 120 (15 %), corte recursivo párrafo → línea → frase → palabra | Suficiente contexto para una idea completa sin mezclar temas; el solape evita cortar ideas en el límite |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) | Multilingüe (español), ligero, corre local y no envía texto a terceros |
| Índice | Chroma persistente, distancia coseno | Open source, embebido, guarda texto + metadatos para citar |
| Recuperación | Similitud coseno, `top_k=4`, descarta fragmentos con distancia > 0.75 | El umbral evita responder con contexto irrelevante (control de alucinaciones) |
| Generación | LLM vía LangChain (Anthropic u OpenAI), temperatura 0.1 | Respuestas estables y fieles al contexto |
| Conversación | Reescritura de la pregunta con el historial antes de buscar | Permite preguntas de seguimiento ("¿y su ventaja?") |

## 2. Instalación y ejecución (local)

Requisitos: Python 3.10–3.12 y una clave de API (Anthropic u OpenAI).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   → pega tu clave en .env
python scripts/build_index.py      # opcional: la app lo construye sola si no existe
streamlit run app.py
```

La primera ejecución descarga el modelo de embeddings (~470 MB). Abre `http://localhost:8501`.

## 3. Evaluación con Ragas

El conjunto de evaluación está en `eval/eval_questions.json` (16 preguntas con respuesta en el corpus + 4 fuera del corpus).

```bash
# Línea base
python eval/run_eval.py --tag base   --chunk-size 1000 --overlap 100 --top-k 2
# Iteración de mejora (cambia chunking y top_k)
python eval/run_eval.py --tag mejora --chunk-size 500  --overlap 75  --top-k 4
# Tabla y gráfico comparativos
python eval/compare.py --before base --after mejora
```

- Métricas Ragas (faithfulness, answer_relevancy, context_precision, context_recall) se calculan sobre las preguntas **con** respuesta en el corpus.
- Las preguntas **fuera** del corpus miden el control de alucinaciones: tasa de abstención (el asistente debe rechazar). Ragas no aplica ahí porque no existe un contexto correcto.
- Resultados: `eval/results/summary_<tag>.json`, `detalle_<tag>.csv`, `comparison.csv` y `docs/ragas_comparison.png`.
- Opcional: usa un juez distinto con `JUDGE_PROVIDER` / `JUDGE_MODEL` (OpenAI `gpt-4o-mini` suele ir muy bien con Ragas).

## 4. Despliegue (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (sin `.env` ni `secrets.toml`; ya están en `.gitignore`).
2. En <https://share.streamlit.io> → **New app** → elige repo, rama y `app.py` como archivo principal.
3. En **Advanced settings → Secrets** pega:
   ```toml
   LLM_PROVIDER = "anthropic"
   ANTHROPIC_API_KEY = "tu-clave"
   ```
4. **Deploy**. La app construye el índice al arrancar a partir de `data/corpus/`.
5. Copia la URL pública y pégala arriba en este README.

Alternativas: Hugging Face Spaces (SDK Streamlit, secrets en *Settings → Variables and secrets*), Render o Railway (comando de inicio: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).

## 5. Estructura del repositorio

```
app.py                   Interfaz de chat (Streamlit)
rag/                     Pipeline: config, ingest, chunking, embeddings, vectorstore, prompts, llm, pipeline
data/corpus/             Documentos fuente (reemplázalos por los tuyos)
eval/                    eval_questions.json, run_eval.py (Ragas), compare.py, results/
scripts/                 build_index.py, build_deliverable_pdf.py
docs/                    PDFs, diagrama y capturas (docs/screenshots/)
```

## 6. Problemas frecuentes

- **`sqlite3` demasiado antiguo en la nube:** ya se maneja con `pysqlite3-binary` (ver inicio de `app.py`).
- **Ragas falla con `n` / `strictness` en Anthropic:** ejecuta con `RAGAS_STRICTNESS=1` o usa `JUDGE_PROVIDER=openai`.
- **Rechaza preguntas válidas:** sube `MAX_DISTANCE` (p. ej. 0.85). **Responde cosas fuera del corpus:** bájalo (p. ej. 0.6).
- **Cambiaste `CHUNK_SIZE` o el modelo de embeddings:** cada combinación crea su propio índice en `data/index/`.
