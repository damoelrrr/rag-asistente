"""Evaluación del AGENTE con Ragas + comparación de costo contra el pipeline fijo del Avance 2.

Antes (mismo conjunto de 20 preguntas, mismos parámetros 800/120/k=4, índice híbrido actual), recalcula el pipeline fijo
para que la comparación sea justa -las métricas viejas de 'mejora' son anteriores a la búsqueda híbrida-:

  python eval/run_eval.py --tag pipeline_actual --chunk-size 800 --overlap 120 --top-k 4
  python eval/run_agent_eval.py --tag agente --baseline pipeline_actual

Qué hace:
  1) Ejecuta el agente sobre cada pregunta (sin historial). Los retrieved_contexts de Ragas son los fragmentos que
     devolvieron sus tools (base + extractos web).
  2) Mide el costo del pipeline fijo y del agente sobre las mismas preguntas: llamadas al LLM, tokens y tiempo.
  3) Puntúa con Ragas: faithfulness, answer_relevancy, context_precision, context_recall (preguntas con respuesta).
  4) Guarda summary_<tag>.json, detalle_<tag>.csv, comparison_agente.csv|md, cost_comparison.csv|md y
     docs/ragas_agente_vs_pipeline.png. --skip-cost omite el paso 2 (ahorra tokens del plan gratuito).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rag.agent import Agent  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.embeddings import as_langchain_embeddings  # noqa: E402
from rag.llm import get_chat_model  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_utils import CountingLLM, checkpointed, with_quota_retry  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def run_ragas(pipe, s, in_scope):
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    answer_relevancy.strictness = int(os.getenv("RAGAS_STRICTNESS", "3"))
    data = [{
        "user_input": r.user_input,
        "retrieved_contexts": list(r.retrieved_contexts),
        "response": re.sub(r"\(Fuente:[^)]*\)|\[web:[^\]]*\]|\s*\[\d+\]", "", r.response),  # sin marcas de cita
        "reference": r.reference,
    } for r in in_scope.itertuples()]
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(get_chat_model(s, judge=True, max_tokens=2000)),
        embeddings=LangchainEmbeddingsWrapper(as_langchain_embeddings(pipe.embedder)),
        run_config=RunConfig(max_workers=int(os.getenv("RAGAS_WORKERS", "4")), timeout=300, max_wait=90),
    )
    rdf = result.to_pandas()
    colmap = {}
    for c in rdf.columns:
        for key, name in [("faith", "faithfulness"), ("relevan", "answer_relevancy"),
                          ("precision", "context_precision"), ("recall", "context_recall")]:
            if key in c.lower():
                colmap[name] = c
    return rdf, colmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agente")
    ap.add_argument("--baseline", default="pipeline_actual", help="tag de la corrida del pipeline fijo (summary_<tag>.json)")
    ap.add_argument("--questions", default=str(HERE / "eval_questions.json"))
    ap.add_argument("--skip-cost", action="store_true")
    a = ap.parse_args()

    from ragas import evaluate  # noqa: F401  (falla rápido si el entorno de Ragas está roto)

    s = Settings()
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    pipe = RAGPipeline(s)
    print(f"Índice: {pipe.ensure_index()} chunks | tools: máx. fragmentos y caracteres fijados en rag/tools.py")
    agent = Agent(pipe)

    OUT.mkdir(exist_ok=True)

    # 1) Agente (cada pregunta se guarda al terminar: si Groq corta por cuota, se retoma)
    def run_agent(q):
        r = with_quota_retry(agent.run, q["question"])
        row = {
            "id": q["id"], "answerable": q["answerable"], "user_input": q["question"], "reference": q["ground_truth"],
            "response": r.text, "refused": r.refused, "retrieved_contexts": r.contexts,
            "tools": "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r.tool_calls) or "ninguna",
            "llm_calls": r.llm_calls, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "seconds": round(r.seconds, 2), "limit_reached": r.limit_reached,
        }
        print(f"  {q['id']} {'REHUSÓ' if r.refused else 'respondió'} | LLM {r.llm_calls} | {row['tools'][:70]}", flush=True)
        return row

    df = pd.DataFrame(checkpointed(OUT / f"_ckpt_{a.tag}_agente.jsonl", questions, run_agent))
    out_scope, in_scope = df[~df.answerable], df[df.answerable]

    # 2) Costo: pipeline fijo vs agente, mismas preguntas.
    #    Si la corrida de referencia (run_eval.py) ya registró llamadas y tokens, se usan esos datos y no se repite.
    cost = None
    base_detail = OUT / f"detalle_{a.baseline}.csv"
    base_ckpt = OUT / f"_ckpt_{a.baseline}_pipeline.jsonl"  # disponible antes de que termine el Ragas del pipeline
    base_df = pd.read_csv(base_detail) if base_detail.exists() else None
    if base_ckpt.exists() and (base_df is None or "llm_calls" not in base_df.columns):
        rows_b = [json.loads(x) for x in base_ckpt.read_text(encoding="utf-8").splitlines()]
        if len(rows_b) >= len(questions):
            base_df = pd.DataFrame(rows_b)
    if not a.skip_cost:
        if base_df is not None and {"llm_calls", "input_tokens", "output_tokens"} <= set(base_df.columns):
            print(f"Costo del pipeline tomado de {base_detail.name} (misma corrida que sus métricas de Ragas)")
            pdf = base_df[["id", "llm_calls", "input_tokens", "output_tokens", "seconds"]]
        else:
            counter = CountingLLM(pipe.llm)
            pipe._llm = counter

            def run_pipeline(q):
                def once():
                    counter.reset()
                    return pipe.answer(q["question"])
                ans = with_quota_retry(once)
                print(f"  costo pipeline {q['id']}: LLM {counter.calls} | {ans.seconds:.1f} s", flush=True)
                return {"id": q["id"], "llm_calls": counter.calls, "input_tokens": counter.input_tokens,
                        "output_tokens": counter.output_tokens, "seconds": ans.seconds}

            pdf = pd.DataFrame(checkpointed(OUT / f"_ckpt_{a.tag}_pipeline_costo.jsonl", questions, run_pipeline))
        cost = pd.DataFrame({
            "métrica": ["llamadas al LLM (promedio)", "tokens de entrada (promedio)", "tokens de salida (promedio)",
                        "tiempo de respuesta en s (promedio)"],
            "pipeline fijo": [pdf.llm_calls.mean(), pdf.input_tokens.mean(), pdf.output_tokens.mean(), pdf.seconds.mean()],
            "agente": [df.llm_calls.mean(), df.input_tokens.mean(), df.output_tokens.mean(), df.seconds.mean()],
        }).round(2)
        cost["agente / pipeline"] = (cost["agente"] / cost["pipeline fijo"].replace(0, float("nan"))).round(2)
        OUT.mkdir(exist_ok=True)
        cost.to_csv(OUT / "cost_comparison.csv", index=False, encoding="utf-8-sig")
        (OUT / "cost_comparison.md").write_text(cost.to_markdown(index=False), encoding="utf-8")
        print("\n=== COSTO ===\n" + cost.to_string(index=False))

    # 3) Ragas sobre las preguntas respondibles
    rdf, colmap = run_ragas(pipe, s, in_scope)
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
        "params": {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "embedding_model": s.embedding_model,
                   "max_distance": s.max_distance, "llm": s.llm_model or s.default_model(s.llm_provider),
                   "max_iterations": s.agent_max_iterations},
        "n_answerable": int(len(in_scope)), "n_unanswerable": int(len(out_scope)),
        "ragas": metrics,
        "abstention_rate_out_of_scope": float(out_scope.refused.mean()) if len(out_scope) else None,
        "false_refusal_rate_in_scope": float(in_scope.refused.mean()) if len(in_scope) else None,
        "avg_seconds": round(float(df.seconds.mean()), 2),
        "avg_llm_calls": round(float(df.llm_calls.mean()), 2),
    }
    (OUT / f"summary_{a.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Tabla Avance 2 vs agente
    base_file = OUT / f"summary_{a.baseline}.json"
    if base_file.exists():
        base = json.loads(base_file.read_text(encoding="utf-8"))
        table = pd.DataFrame({
            "Métrica": METRICS + ["abstención fuera del corpus"],
            "Avance 2 (pipeline fijo)": [base["ragas"].get(m) for m in METRICS] + [base["abstention_rate_out_of_scope"]],
            "Avance 3 (agente)": [metrics.get(m) for m in METRICS] + [summary["abstention_rate_out_of_scope"]],
        })
        table["Diferencia"] = (table["Avance 3 (agente)"] - table["Avance 2 (pipeline fijo)"]).round(4)
        table.to_csv(OUT / "comparison_agente.csv", index=False, encoding="utf-8-sig")
        (OUT / "comparison_agente.md").write_text(table.to_markdown(index=False), encoding="utf-8")
        print("\n=== RAGAS: Avance 2 vs agente ===\n" + table.to_string(index=False))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.2))
            w, x = 0.38, range(len(METRICS))
            ax.bar([i - w / 2 for i in x], [base["ragas"].get(m, 0) for m in METRICS], w, label=f"Avance 2 ({a.baseline})", color="#9aa5b8")
            ax.bar([i + w / 2 for i in x], [metrics.get(m, 0) for m in METRICS], w, label=f"Avance 3 ({a.tag})", color="#4f46e5")
            for i, m in enumerate(METRICS):
                ax.text(i - w / 2, base["ragas"].get(m, 0) + 0.015, f"{base['ragas'].get(m, 0):.2f}", ha="center", fontsize=8)
                ax.text(i + w / 2, metrics.get(m, 0) + 0.015, f"{metrics.get(m, 0):.2f}", ha="center", fontsize=8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(["faithfulness", "answer\nrelevancy", "context\nprecision", "context\nrecall"])
            ax.set_ylim(0, 1.1)
            ax.set_title("Ragas: pipeline fijo (Avance 2) vs agente (Avance 3)")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(ROOT / "docs" / "ragas_agente_vs_pipeline.png", dpi=200)
            print("Gráfico: docs/ragas_agente_vs_pipeline.png")
        except Exception as exc:  # el gráfico es opcional
            print(f"(sin gráfico: {exc})")
    else:
        print(f"\nNo encontré {base_file.name}: ejecuta primero run_eval.py --tag {a.baseline} para comparar.")
    print("\n=== RESULTADOS AGENTE ===\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
