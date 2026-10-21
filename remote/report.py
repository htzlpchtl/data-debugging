"""Generate the one-table/one-figure research checkpoint from complete outputs."""
import json
import math

from run import evaluation_jobs, remote_only, write_json


def report(c):
    remote_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rows = c.complete("scores", c.rows("candidates"))
    judged = {r["id"]: r for r in c.complete("judge_sample", rows)}
    rows = sorted(rows, key=lambda r: (r["pair_id"], r["label"]))
    pairs = sorted({r["pair_id"] for r in rows})
    if len(rows) != 2 * len(pairs) or [r["label"] for r in rows] != [0, 1] * len(pairs):
        raise ValueError("Report requires complete normal/high prompt pairs.")
    for row in rows:
        row["text_judge"] = judged[row["id"]]["trait"]
    names = {"projection": "Raw projection", "delta_projection": "Projection difference",
             "text_judge": "LLM text judgement", "lexical_score": "Saved caveat baseline",
             "tfidf_score": "Saved TF-IDF baseline", "response_tokens": "Answer length"}
    metrics, correlations = {}, {}
    rng = np.random.default_rng(c.cfg["seed"])
    delta = np.array([r["delta_projection"] for r in rows])
    for key, name in names.items():
        values = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], dtype=float).reshape(-1, 2)
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        if not len(values):
            continue
        labels = np.tile([0, 1], len(values))
        draws = rng.integers(0, len(values), size=(1000, len(values)))
        bootstrap = values[draws]
        auc = [roc_auc_score(labels, b.ravel()) for b in bootstrap]
        gaps = (bootstrap[:, :, 1] - bootstrap[:, :, 0]).mean(axis=1)
        metrics[key] = {"name": name, "pairs": len(values), "excluded_pairs": int((~valid).sum()),
            "auc": float(roc_auc_score(labels, values.ravel())), "auc_ci": np.quantile(auc, [.025, .975]).tolist(),
            "mean_normal": float(values[:, 0].mean()), "mean_high": float(values[:, 1].mean()),
            "paired_gap": float((values[:, 1] - values[:, 0]).mean()),
            "paired_gap_ci": np.quantile(gaps, [.025, .975]).tolist()}
        comparison = delta.reshape(-1, 2)[valid]
        for condition, a, b in (("pooled", comparison.ravel(), values.ravel()),
                                 ("normal", comparison[:, 0], values[:, 0]),
                                 ("high", comparison[:, 1], values[:, 1])):
            rho = float(spearmanr(a, b).statistic) if len(a) > 1 and np.ptp(a) and np.ptp(b) else float("nan")
            correlations[f"{key}_{condition}"] = rho if math.isfinite(rho) else None
    low, high = sorted(rows, key=lambda r: r["delta_projection"])[:len(rows)//4], sorted(rows, key=lambda r: r["delta_projection"])[-len(rows)//4:]
    selected = {name: {"n": len(group), "mean_delta": sum(r["delta_projection"] for r in group) / len(group),
                       "fraction_high_label": sum(r["label"] for r in group) / len(group)}
                for name, group in (("low_quartile", low), ("high_quartile", high))}
    # Question-level bootstrap; repeats of one question are not independent units.
    behaviour = {}
    per_question = {}
    for arm in ("base", "high", "low", "random"):
        judge_path = c.out / f"judge_evaluation_{arm}.jsonl"
        if not judge_path.exists():
            continue
        expected = evaluation_jobs(c)
        generation = {r["id"]: r for r in c.complete(f"evaluation_{arm}", expected)}
        evaluations = c.complete(f"judge_evaluation_{arm}", expected)
        groups = {}
        for row in evaluations:
            qid = generation[row["id"]]["pair_id"]
            if row["trait"] is not None:
                groups.setdefault(qid, []).append(row["trait"])
        per_question[arm] = {qid: float(np.mean(scores)) for qid, scores in groups.items()}
        question_means = np.array(list(per_question[arm].values()))
        if not len(question_means):
            raise ValueError(f"No valid trait evaluations for {arm}.")
        means = question_means[rng.integers(len(question_means), size=(1000, len(question_means)))].mean(axis=1)
        coherence = [r["coherence"] for r in evaluations if r["coherence"] is not None]
        behaviour[arm] = {"mean_trait": float(question_means.mean()), "ci": np.quantile(means, [.025, .975]).tolist(),
            "questions_with_valid_trait": len(groups), "responses": len(evaluations),
            "missing_trait": sum(r["trait"] is None for r in evaluations),
            "missing_coherence": len(evaluations) - len(coherence),
            "mean_coherence": float(np.mean(coherence)) if coherence else None,
            "hit_token_cap": sum(r["hit_token_cap"] for r in generation.values())}
    contrasts = {}
    for left, right in (("high", "low"), ("high", "base"), ("low", "base"), ("random", "base")):
        if left in per_question and right in per_question:
            shared = sorted(per_question[left].keys() & per_question[right].keys())
            if not shared:
                continue
            differences = np.array([per_question[left][q] - per_question[right][q] for q in shared])
            draws = differences[rng.integers(len(shared), size=(1000, len(shared)))].mean(axis=1)
            contrasts[f"{left}_minus_{right}"] = {"mean": float(differences.mean()),
                "ci": np.quantile(draws, [.025, .975]).tolist(), "paired_questions": len(shared)}
    summary = {"metrics": metrics, "spearman_with_delta": correlations, "selected_quartiles": selected,
        "behaviour": behaviour, "behaviour_contrasts": contrasts,
        "vector": json.loads((c.out / "vector.json").read_text()),
        "generation_cap_hits": {name: sum(r["hit_token_cap"] for r in c.rows(name)) for name in ("extraction", "natural")},
        "uncertainty": "1000 prompt-pair/question bootstrap draws. Conditional on one vector, judge and training seed; not training-seed uncertainty.",
        "scope": "Label separation/ranking agreement; SFT exploratory if present. No claim of incremental prediction over semantics.",
        "deviations": c.cfg["deviations"]}
    write_json(c.out / "summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for label, name in ((0, "Normal"), (1, "Hallucination II")):
        group = [r for r in rows if r["label"] == label]
        axes[0].hist([r["delta_projection"] for r in group], bins=18, alpha=.6, label=name)
        axes[1].scatter([r["delta_projection"] for r in group if r["text_judge"] is not None],
                        [r["text_judge"] for r in group if r["text_judge"] is not None], s=15, alpha=.65, label=name)
    axes[0].set(xlabel="Projection difference", ylabel="Examples", title="Public dataset separation")
    axes[1].set(xlabel="Projection difference", ylabel="LLM text judgement", title="Does the text already explain the ranking?")
    axes[0].legend()
    fig.suptitle(f"Qwen2.5-7B BF16 · {len(pairs)} question pairs · reduced reproduction")
    fig.savefig(c.out / "comparison.png", dpi=180)
    fig.savefig(c.out / "comparison.pdf")
    plt.close(fig)
    table = ["| Score | Dataset-label AUC (95% interval) | Valid pairs |", "|---|---:|---:|"]
    for metric in metrics.values():
        lo, hi = metric["auc_ci"]
        table.append(f"| {metric['name']} | {metric['auc']:.3f} ({lo:.3f}–{hi:.3f}) | {metric['pairs']} |")
    note = ["# Remote reproduction checkpoint", "", f"Completed scoring of {len(rows)} public answers using a vector from {len(summary['vector']['accepted_pairs'])} accepted contrastive pairs.", "", *table, "", "![Comparison](comparison.png)", "", "These AUCs measure dataset labels, not prediction of post-SFT behaviour. Intervals resample whole questions and condition on one extracted vector."]
    if behaviour:
        note += ["", "| Model | Mean hallucination score | Mean coherence |", "|---|---:|---:|"]
        for arm, value in behaviour.items():
            note.append(f"| {arm} | {value['mean_trait']:.2f} | {value['mean_coherence']} |")
        note += ["", "Training uses one seed and subsets that are not matched on text score. The conditional paired differences and missingness counts are in summary.json."]
    else:
        note += ["", "No SFT outcome has been measured in this run."]
    note += ["", "Discrepancies from the published procedure:", "", *[f"- {d}" for d in c.cfg["deviations"]], "", "A numerical difference from the paper has not been established: compare like-for-like settings before claiming a replication failure.", "", "Next: inspect truncation, missing judge scores, and within-condition ranking agreement; then compare SFT subsets matched on text judgement and length but separated in projection difference.", ""]
    (c.out / "RESEARCH_NOTE.md").write_text("\n".join(note))
