"""Evaluación del RAG con Ragas.

Uso:
  python eval/run_eval.py --tag base    --chunk-size 1000 --overlap 100 --top-k 2
  python eval/run_eval.py --tag mejora  --chunk-size 500  --overlap 75  --top-k 4

Reporta faithfulness, answer_relevancy, context_precision y context_recall sobre las preguntas
CON respuesta en el corpus, y mide aparte el control de alucinaciones (abstención) con las preguntas
FUERA del corpus (para ellas las métricas de Ragas no tienen sentido: no existe contexto correcto).
"""
import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rag.config import Settings  # noqa: E402
from rag.embeddings import as_langchain_embeddings  # noqa: E402
from rag.llm import get_chat_model  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nombre de la corrida, p. ej. base / mejora")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--questions", default=str(HERE / "eval_questions.json"))
    a = ap.parse_args()

    s = Settings()
    s = replace(
        s,
        chunk_size=a.chunk_size or s.chunk_size,
        chunk_overlap=a.overlap or s.chunk_overlap,
        top_k=a.top_k or s.top_k,
    )
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    pipe = RAGPipeline(s)
    print(f"Índice: {pipe.ensure_index()} chunks | chunk={s.chunk_size} overlap={s.chunk_overlap} top_k={s.top_k}")

    # 1) Ejecutar el pipeline real sobre cada pregunta (sin historial)
    rows = []
    for q in questions:
        ans = pipe.answer(q["question"])
        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "user_input": q["question"],
                "reference": q["ground_truth"],
                "response": ans.text,
                "refused": ans.refused,
                "retrieved_contexts": [h.text for h in ans.raw_hits],
                "sources": [f"{h.source}#{h.chunk_index}" for h in ans.raw_hits],
                "seconds": round(ans.seconds, 2),
            }
        )
        print(f"  {q['id']} {'REHUSÓ' if ans.refused else 'respondió'} | {q['question'][:60]}")
    df = pd.DataFrame(rows)

    # 2) Control de alucinaciones (preguntas fuera del corpus)
    out_scope, in_scope = df[~df.answerable], df[df.answerable]
    abstention = float(out_scope.refused.mean()) if len(out_scope) else None
    false_refusal = float(in_scope.refused.mean()) if len(in_scope) else None

    # 3) Ragas sobre las preguntas respondibles
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    answer_relevancy.strictness = int(os.getenv("RAGAS_STRICTNESS", "3"))
    data = [
        {
            "user_input": r.user_input,
            "retrieved_contexts": list(r.retrieved_contexts),
            "response": re.sub(r"\s*\[\d+\]", "", r.response),  # sin marcas de cita
            "reference": r.reference,
        }
        for r in in_scope.itertuples()
    ]
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(get_chat_model(s, judge=True, max_tokens=2000)),
        embeddings=LangchainEmbeddingsWrapper(as_langchain_embeddings(pipe.embedder)),
        run_config=RunConfig(max_workers=4, timeout=180),
    )
    rdf = result.to_pandas()
    colmap = {}
    for c in rdf.columns:
        for key, name in [("faith", "faithfulness"), ("relevan", "answer_relevancy"),
                          ("precision", "context_precision"), ("recall", "context_recall")]:
            if key in c.lower():
                colmap[name] = c
    metrics = {k: round(float(rdf[v].mean(skipna=True)), 4) for k, v in colmap.items()}
    for k, v in colmap.items():
        in_scope = in_scope.assign(**{k: rdf[v].values})

    # 4) Guardar
    OUT.mkdir(exist_ok=True)
    full = pd.concat([in_scope, out_scope], ignore_index=True)
    full["retrieved_contexts"] = full["retrieved_contexts"].apply(lambda x: " ||| ".join(x))
    full.to_csv(OUT / f"detalle_{a.tag}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "tag": a.tag,
        "params": {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "top_k": s.top_k,
                   "embedding_model": s.embedding_model, "max_distance": s.max_distance,
                   "llm": s.llm_model or s.default_model(s.llm_provider)},
        "n_answerable": int(len(in_scope)),
        "n_unanswerable": int(len(out_scope)),
        "ragas": metrics,
        "abstention_rate_out_of_scope": abstention,
        "false_refusal_rate_in_scope": false_refusal,
        "avg_seconds": round(float(df.seconds.mean()), 2),
    }
    (OUT / f"summary_{a.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RESULTADOS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
