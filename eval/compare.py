"""Compara dos corridas de evaluación y genera tabla + gráfico.

  python eval/compare.py --before base --after mejora
Salidas: eval/results/comparison.csv y docs/ragas_comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

ap = argparse.ArgumentParser()
ap.add_argument("--before", required=True)
ap.add_argument("--after", required=True)
a = ap.parse_args()

b = json.loads((RES / f"summary_{a.before}.json").read_text(encoding="utf-8"))
f = json.loads((RES / f"summary_{a.after}.json").read_text(encoding="utf-8"))

rows = [{"métrica": m, "antes": b["ragas"].get(m), "después": f["ragas"].get(m)} for m in METRICS]
rows.append({"métrica": "abstención fuera de corpus", "antes": b["abstention_rate_out_of_scope"],
             "después": f["abstention_rate_out_of_scope"]})
df = pd.DataFrame(rows)
df["cambio"] = (df["después"] - df["antes"]).round(4)
df.to_csv(RES / "comparison.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(METRICS))
w = 0.38
ax.bar([i - w / 2 for i in x], [b["ragas"].get(m, 0) for m in METRICS], w, label=f"Antes ({a.before})", color="#9aa5b8")
ax.bar([i + w / 2 for i in x], [f["ragas"].get(m, 0) for m in METRICS], w, label=f"Después ({a.after})", color="#4f46e5")
for i, m in enumerate(METRICS):
    for off, d in ((-w / 2, b), (w / 2, f)):
        ax.text(i + off, d["ragas"].get(m, 0) + 0.015, f"{d['ragas'].get(m, 0):.2f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(["faithfulness", "answer\nrelevancy", "context\nprecision", "context\nrecall"])
ax.set_ylim(0, 1.1)
ax.set_title("Métricas Ragas: antes vs. después de la iteración")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / "docs" / "ragas_comparison.png", dpi=200)
print("Gráfico: docs/ragas_comparison.png")
