"""Remote-only supplementary audit of the completed controlled SFT run.

Reuses the existing fixed lexical baseline; does not fit a predictor or load a model.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re

if platform.system() != "Linux" or os.environ.get("DATA_DEBUGGING_REMOTE") != "1":
    raise SystemExit("Run only on approved remote compute.")

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

out = Path(__file__).resolve().parent
root = out.parents[1]
source = root / "scripts/prepare.py"
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "lexical_score")
namespace = {"re": re}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
lexical_score = namespace["lexical_score"]
rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
projections = {r["id"]: r for r in map(json.loads, (out / "training_projection.jsonl").read_text().splitlines())}
assert set(projections) == {r["id"] for r in rows}
scores = [{"id": r["id"], "pair_id": r["pair_id"], "label": r["label"],
           "projection": projections[r["id"]]["projection"],
           "lexical_score": lexical_score(r["answer"])} for r in rows]
(out / "training_score_comparison.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
labels = [r["label"] for r in scores]
summary = {"scope": "Descriptive post-hoc audit using a lexical rule frozen before this experiment; no outcome-based fitting. AUC refers to source labels, not SFT outcomes.",
           "lexical_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "lexical_definition": ast.get_source_segment(source.read_text(), node),
           "by_condition": {}, "dataset_label_auc": {}}
for label, arm in ((0, "normal"), (1, "trait")):
    subset = [r for r in scores if r["label"] == label]
    summary["by_condition"][arm] = {"n": len(subset),
        "mean_projection": float(np.mean([r["projection"] for r in subset])),
        "mean_lexical_score": float(np.mean([r["lexical_score"] for r in subset])),
        "spearman_projection_lexical": float(spearmanr([r["projection"] for r in subset], [r["lexical_score"] for r in subset]).statistic)}
for key in ("projection", "lexical_score"):
    summary["dataset_label_auc"][key] = float(roc_auc_score(labels, [r[key] for r in scores]))
summary["pooled_spearman_projection_lexical"] = float(spearmanr([r["projection"] for r in scores], [r["lexical_score"] for r in scores]).statistic)
behaviour = json.loads((out / "behaviour_summary.json").read_text())
means = behaviour["per_question_trait_means"]
differences = [means["trait"][q] - means["normal"][q] for q in means["normal"]]
summary["question_robustness"] = {"trait_higher_than_normal": sum(d > 0 for d in differences),
    "equal": sum(d == 0 for d in differences), "questions": len(differences),
    "minimum_difference": min(differences), "maximum_difference": max(differences)}
ledger = [json.loads(s) for s in (out / "judge_usage.jsonl").read_text().splitlines()]
summary["judge_cost"] = {"successful_calls": len(ledger),
    "estimated_usd": sum(r["estimated_usd"] for r in ledger),
    "provider_reported_usd": sum(r["usage"]["cost"] for r in ledger),
    "scope": "Successful requests in the local ledger; excludes unobserved failures and account funding fees."}
(out / "supplementary_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2))

# Retain the original report's figure; place labels above intervals in this version.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
arms = ["base", "normal", "trait"]
values = [behaviour["arms"][a]["mean_trait"] for a in arms]
intervals = [behaviour["arms"][a]["ci95"] for a in arms]
errors = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(values, intervals)]).T
fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
ax.bar(["Base model", "Normal SFT", "Hallucination II SFT"], values,
       yerr=errors, capsize=5, color=["#7a8793", "#377eb8", "#e78332"])
for i, (v, (_, hi)) in enumerate(zip(values, intervals)):
    ax.text(i, hi + 2.0, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
ax.set(ylim=(0, 110), yticks=range(0, 101, 20),
       ylabel="Mean judged hallucination score (0–100)",
       title="Qwen2.5-7B: effect of paired training data")
ax.spines[["top", "right"]].set_visible(False)
fig.supxlabel("1,024 examples per arm · one training seed · 20 held-out questions × 5 answers\n95% intervals resample questions; they exclude training-seed uncertainty", fontsize=9)
fig.savefig(out / "behaviour_summary_plot.png", dpi=180)
fig.savefig(out / "behaviour_summary_plot.pdf")
plt.close(fig)
