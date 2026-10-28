"""One prospective, length/caveat-balanced, high/low/random SFT pilot."""
import argparse
from collections import Counter, defaultdict
import json
import random
import shutil
import signal
import time
import traceback

from run import (Context, ROOT, Runtime, append_row, evaluation_jobs, initialize,
                 judge, preflight, read_rows, remote_only, sha, write_json)

ARMS = ("low", "random", "high")


def prepare(c):
    initialize(c)
    saved = ROOT / "results/controlled_sft_2026-10-28"
    pool_path = ROOT / c.cfg["saved_scores"]
    pool = {r["id"]: r for r in read_rows(pool_path)}
    candidates = c.rows("candidates")
    assert set(pool) == {r["id"] for r in candidates}
    eligible = []
    for row in candidates:
        score = pool[row["id"]]
        assert row["question"] == score["question"] and row["answer"] == score["answer"]
        assert row["label"] == score["label"]
        if score["text_score"] < 1:
            eligible.append({**row, **{k: score[k] for k in
                            ("projection", "response_tokens", "text_score", "caveat_count")}})
    assert len(eligible) == c.cfg["expected_eligible"]
    assert len({r["question"] for r in eligible}) == len(eligible)
    assert not {r["question"] for r in eligible} & set(c.extract["questions"] + c.evaluation["questions"])
    cells = Counter((r["caveat_count"], r["response_tokens"]) for r in eligible)
    capacity = sum(count // 2 for count in cells.values())
    write_json(c.out / "initial_overlap_check.json", {"maximum_disjoint_tail_rows_per_arm": capacity,
        "required": c.cfg["training_subset_size"], "passed": capacity >= c.cfg["training_subset_size"]})
    assert capacity >= c.cfg["training_subset_size"], "Insufficient overlap even before reference scoring"
    path = c.out / "eligible.jsonl"
    serialized = "".join(json.dumps(r) + "\n" for r in eligible)
    if path.exists(): assert path.read_text() == serialized
    else: path.write_text(serialized)
    previous = json.loads((saved / "manifest.json").read_text())
    assert previous["resolved_model_revision"] == c.lock["resolved_model_revision"]
    for key in ("model", "seed", "max_new_tokens", "generation_batch_size", "evaluation_repeats"):
        assert previous["identity"]["config"][key] == c.cfg[key]
    sources = [pool_path, saved / "evaluation_base.jsonl", saved / "judge_evaluation_base.jsonl",
               saved / "vector.json", ROOT / "results/remote_scoring_2026-10-21/unit_vector.pt"]
    source_record = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if (c.out / "source_inputs.json").exists():
        assert json.loads((c.out / "source_inputs.json").read_text()) == source_record
    else: write_json(c.out / "source_inputs.json", source_record)
    for name in ("evaluation_base.jsonl", "judge_evaluation_base.jsonl", "vector.json"):
        if (c.out / name).exists(): assert sha(c.out / name) == sha(saved / name)
        else: shutil.copy2(saved / name, c.out / name)
    vector_path = ROOT / "results/remote_scoring_2026-10-21/unit_vector.pt"
    metadata = json.loads((saved / "vector.json").read_text())
    assert sha(vector_path) == metadata["vector_sha256"]
    shutil.copy2(vector_path, c.out / "unit_vector.pt")
    c.complete("evaluation_base", evaluation_jobs(c))
    c.complete("judge_evaluation_base", evaluation_jobs(c))
    write_json(c.out / "baseline_lock.json", {"evaluation_sha256": sha(c.out / "evaluation_base.jsonl"),
        "judge_sha256": sha(c.out / "judge_evaluation_base.jsonl"), "source": "Earlier controlled SFT baseline, reused unchanged"})
    write_json(c.out / "protocol.json", {"status": "FROZEN_BEFORE_REFERENCES_AND_NEW_TRAINING",
        "hypothesis": "High reference-subtracted projection SFT yields more held-out hallucination than random, which yields more than low.",
        "primary_contrast": "high minus random; low minus random and high minus low are secondary",
        "selection": "Text score <1; exact response-token-count and caveat-count strata; quotas independent of activation; tails within each stratum; random from the entire same stratum, with overlap allowed.",
        "subset_size_rationale": "128 rows/arm, eight epochs and effective batch 16 gives 64 optimizer steps, matching the preceding successful pilot's update count. This is a resource-based pilot choice, not a power calculation.",
        "scope": "One training seed; judge-threshold filtering is not exact matching of judge scores or all semantics; topic and answerability may remain imbalanced.",
        "reference_caps": "Generate to 1000 tokens; cap-hit natural references are preserved but excluded from selection, before ranking.",
        "numerical_gate": "Fresh upstream mean-activation parity plus cached projection/token checks precede references.",
        "config": c.cfg})


def validate_sources(c):
    for name, digest in json.loads((c.out / "source_inputs.json").read_text()).items():
        assert sha(ROOT / name) == digest, name
    lock = json.loads((c.out / "baseline_lock.json").read_text())
    assert sha(c.out / "evaluation_base.jsonl") == lock["evaluation_sha256"]
    assert sha(c.out / "judge_evaluation_base.jsonl") == lock["judge_sha256"]


def references(c):
    c.require_preflight()
    eligible = c.rows("eligible")
    runtime = Runtime(c)
    direction = runtime.torch.load(c.out / "unit_vector.pt", weights_only=True).float()
    assert abs(float(direction.norm()) - 1) < 1e-5
    checks = []
    for row in random.Random(c.cfg["seed"]).sample(eligible, 4):
        mean, count = runtime.response_mean(runtime.prompt(row["messages"][:-1]), row["answer"])
        value = float(mean @ direction)
        checks.append({"id": row["id"], "saved_projection": row["projection"], "fresh_projection": value,
                       "absolute_error": abs(value - row["projection"]),
                       "passed": abs(value - row["projection"]) <= 1e-3 and count == row["response_tokens"]})
    write_json(c.out / "cached_projection_checks.json", {"passed": all(r["passed"] for r in checks), "checks": checks})
    assert all(r["passed"] for r in checks), "Cached projections failed fresh numerical check"
    jobs = [{"id": str(r["pair_id"]), "question": r["question"], "messages": r["messages"][:-1]} for r in eligible]
    # Greedy natural answers use no repetition penalty; evaluation retains its earlier settings.
    runtime.model.generation_config.repetition_penalty = 1.0
    write_json(c.out / "reference_generation_config.json", runtime.model.generation_config.to_dict())
    original = c.cfg
    try:
        c.cfg = {**original, "generation_batch_size": original["reference_batch_size"]}
        runtime.generate(jobs, "natural", temperature=0)
    finally:
        c.cfg = original
    natural = {r["id"]: r for r in c.complete("natural", jobs)}
    projections = {r["id"]: r for r in c.rows("natural_projection")}
    for key, row in natural.items():
        if key not in projections:
            mean, count = runtime.response_mean(row["prompt"], row["answer"])
            result = {"id": key, "projection": float(mean @ direction), "response_tokens": count,
                      "hit_token_cap": row["hit_token_cap"]}
            append_row(c.out / "natural_projection.jsonl", result)
            projections[key] = result
    done = {r["id"] for r in c.rows("scores")}
    for row in eligible:
        if row["id"] not in done:
            ref = projections[str(row["pair_id"])]
            append_row(c.out / "scores.jsonl", {**row, "natural_projection": ref["projection"],
                "delta_projection": row["projection"] - ref["projection"], "natural_hit_cap": ref["hit_token_cap"]})
    c.complete("scores", eligible)


def select(c):
    import numpy as np
    from scipy.stats import spearmanr, ks_2samp
    all_rows = c.complete("scores", c.rows("eligible"))
    rows = [r for r in all_rows if not r["natural_hit_cap"]]
    cells = defaultdict(list)
    for row in rows: cells[(row["caveat_count"], row["response_tokens"])].append(row)
    cells = {key: value for key, value in sorted(cells.items()) if len(value) >= 2}
    capacity = {key: len(value) // 2 for key, value in cells.items()}
    n = c.cfg["training_subset_size"]
    support = {"eligible": len(all_rows), "reference_cap_exclusions": len(all_rows) - len(rows),
               "matchable_cells": len(cells), "maximum_disjoint_tail_rows_per_arm": sum(capacity.values())}
    write_json(c.out / "selection_support.json", support)
    if sum(capacity.values()) < n:
        raise ValueError("Insufficient exact length/caveat overlap for the frozen subset size; no training.")
    # Population-proportional integer quotas, bounded by disjoint high/low capacity.
    # Neither quota allocation nor inclusion of a cell reads activation scores.
    population = sum(map(len, cells.values()))
    desired = {key: n * len(value) / population for key, value in cells.items()}
    quota = {key: 0 for key in cells}
    for _ in range(n):
        key = max((key for key in cells if quota[key] < capacity[key]),
                  key=lambda key: (desired[key] - quota[key], key))
        quota[key] += 1
    arms = {arm: [] for arm in ARMS}
    rng = random.Random(c.cfg["seed"])
    strata = []
    for key, pool in cells.items():
        k = quota[key]
        if not k: continue
        ordered = sorted(pool, key=lambda r: (r["delta_projection"], r["id"]))
        arms["low"].extend(ordered[:k])
        arms["high"].extend(ordered[-k:])
        arms["random"].extend(rng.sample(pool, k))
        strata.append({"caveat_count": key[0], "response_tokens": key[1], "pool_n": len(pool),
                       "arm_n": k, "selected_fraction_per_tail": k / len(pool)})
    order = list(range(n)); rng.shuffle(order)
    arms = {arm: [group[i] for i in order] for arm, group in arms.items()}
    expected_features = [(r["response_tokens"], r["caveat_count"]) for r in arms["low"]]
    for group in arms.values():
        assert len(group) == len({r["id"] for r in group}) == len({r["question"] for r in group}) == n
        assert [(r["response_tokens"], r["caveat_count"]) for r in group] == expected_features
        assert all(r["text_score"] < 1 for r in group)
    assert not {r["id"] for r in arms["high"]} & {r["id"] for r in arms["low"]}
    fields = ("delta_projection", "projection", "text_score", "response_tokens", "caveat_count")
    summaries = {arm: {field: {"mean": float(np.mean([r[field] for r in group])),
                   "min": min(r[field] for r in group), "max": max(r[field] for r in group)} for field in fields}
                 for arm, group in arms.items()}
    gap = summaries["high"]["delta_projection"]["mean"] - summaries["low"]["delta_projection"]["mean"]
    assert gap > 1e-3, "No activation separation beyond the cached-projection numerical tolerance"
    destination = c.out / "subsets"; destination.mkdir(exist_ok=True)
    manifest = {}
    for arm, group in arms.items():
        path = destination / f"{arm}.jsonl"
        value = "".join(json.dumps(r) + "\n" for r in group)
        if path.exists(): assert path.read_text() == value
        else: path.write_text(value)
        manifest[arm] = {"n": n, "sha256": sha(path), "summary": summaries[arm]}
    raw_order = sorted(rows, key=lambda r: (r["projection"], r["id"]))
    delta_order = sorted(rows, key=lambda r: (r["delta_projection"], r["id"]))
    manifest.update({"strata": strata, "matching": "Exact response-token and caveat-count sequences in all arms; text scores filtered below 1 but not exactly matched; questions differ.",
        "raw_delta_spearman": float(spearmanr([r["projection"] for r in rows], [r["delta_projection"] for r in rows]).statistic),
        "unbalanced_top_n_raw_delta_overlap": len({r["id"] for r in raw_order[-n:]} & {r["id"] for r in delta_order[-n:]}),
        "high_low_text_ks": float(ks_2samp([r["text_score"] for r in arms["high"]], [r["text_score"] for r in arms["low"]]).statistic),
        "random_high_overlap": len({r["id"] for r in arms["random"]} & {r["id"] for r in arms["high"]}),
        "random_low_overlap": len({r["id"] for r in arms["random"]} & {r["id"] for r in arms["low"]})})
    write_json(destination / "manifest.json", manifest)
    write_json(c.out / "predictions.json", {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predicted_order_of_mean_hallucination": ["low", "random", "high"],
        "primary_prediction": "high > random", "not_predicted": "No numerical forecast of the hallucination-score difference or base-relative direction.",
        "mean_delta_projection": {a: summaries[a]["delta_projection"]["mean"] for a in ARMS},
        "subset_manifest_sha256": sha(destination / "manifest.json")})
    for arm, group in arms.items():
        # A deterministic, small textual audit set; no cherry-picked examples.
        (c.out / f"audit_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in group[:5]))
    print(json.dumps({"support": support, "arm_summaries": summaries, "raw_delta_spearman": manifest["raw_delta_spearman"]}), flush=True)


def report(c):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jobs = evaluation_jobs(c)
    rng = np.random.default_rng(c.cfg["seed"])
    results, means_by_q = {}, {}
    for arm in ("base",) + ARMS:
        generated = c.complete(f"evaluation_{arm}", jobs)
        judged = c.complete(f"judge_evaluation_{arm}", jobs)
        assert all(r["trait"] is not None and r["coherence"] is not None for r in judged)
        mapping = {r["id"]: r for r in judged}
        groups = defaultdict(list)
        for row in generated: groups[row["pair_id"]].append(mapping[row["id"]]["trait"])
        means = np.array([np.mean(groups[q]) for q in sorted(groups)])
        assert len(means) == 20
        means_by_q[arm] = means
        boot = means[rng.integers(20, size=(2000, 20))].mean(1)
        results[arm] = {"mean_trait": float(means.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "mean_coherence": float(np.mean([r["coherence"] for r in judged])),
            "responses": len(generated), "questions": 20,
            "cap_hits": sum(r["hit_token_cap"] for r in generated),
            "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in generated]))}
    contrasts = {}
    for left, right in (("high", "random"), ("low", "random"), ("high", "low"), ("random", "base"), ("high", "base"), ("low", "base")):
        delta = means_by_q[left] - means_by_q[right]
        boot = delta[rng.integers(20, size=(2000, 20))].mean(1)
        contrasts[f"{left}_minus_{right}"] = {"mean": float(delta.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    write_json(c.out / "behaviour_summary.json", {"status": "COMPLETE_ONE_SEED",
        "arms": results, "contrasts": contrasts,
        "per_question_trait_means": {a: v.tolist() for a, v in means_by_q.items()},
        "prediction": json.loads((c.out / "predictions.json").read_text()),
        "limitations": c.cfg["deviations"]})
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    labels = list(results)
    means = [results[a]["mean_trait"] for a in labels]
    err = np.array([[results[a]["mean_trait"] - results[a]["ci95"][0], results[a]["ci95"][1] - results[a]["mean_trait"]] for a in labels]).T
    ax.bar(labels, means, yerr=err, capsize=4, color=["#888888", "#d17b25", "#6a5acd", "#176b99"])
    ax.set(ylabel="Held-out hallucination score (0–100)", ylim=(0, 105), title="Length/caveat-balanced SFT: one training seed")
    for i, value in enumerate(means): ax.text(i, value + 2, f"{value:.1f}", ha="center")
    fig.savefig(c.out / "behaviour.png", dpi=180); fig.savefig(c.out / "behaviour.pdf"); plt.close(fig)
    scores = c.rows("scores")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, field, title in zip(axes, ("projection", "delta_projection"), ("Saved raw projection", "Reference-subtracted projection")):
        ax.scatter([r["text_score"] for r in scores], [r[field] for r in scores], color="#bbbbbb", alpha=.35, s=10)
        for arm, color in (("low", "#d17b25"), ("high", "#176b99"), ("random", "#6a5acd")):
            group = read_rows(c.out / "subsets" / f"{arm}.jsonl")
            ax.scatter([r["text_score"] for r in group], [r[field] for r in group], label=arm, s=19,
                       facecolors="none" if arm == "random" else color, edgecolors=color, alpha=.6)
        ax.set(xlabel="LLM text score (eligible scores <1)", ylabel=title, title=title)
        ax.legend(fontsize=8)
    fig.savefig(c.out / "judge_vs_projection.png", dpi=180); fig.savefig(c.out / "judge_vs_projection.pdf"); plt.close(fig)
    lines = ["| Arm | Hallucination mean (95% CI) | Coherence | Cap hits |", "|---|---:|---:|---:|"]
    for a, r in results.items(): lines.append(f"| {a} | {r['mean_trait']:.2f} ({r['ci95'][0]:.2f}–{r['ci95'][1]:.2f}) | {r['mean_coherence']:.2f} | {r['cap_hits']} |")
    (c.out / "results_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"arms": results, "contrasts": contrasts}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "preflight", "references", "select", "train", "evaluate", "judge", "report"])
    parser.add_argument("--config", default="remote/balanced-sft-2026-10-28.json")
    parser.add_argument("--out", default="runs/balanced-sft-2026-10-28")
    parser.add_argument("--arm", choices=["base", *ARMS], default="base")
    parser.add_argument("--target", choices=["check", "evaluation"], default="evaluation")
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    remote_only()
    c = Context(args); c.check_manifest(initialize=args.stage == "init")
    if args.stage != "init": validate_sources(c)
    started, status = time.monotonic(), "failed"
    def timeout(*_): raise TimeoutError("Stage deadline reached; stop Pod billing separately")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
    try:
        if args.stage == "init": prepare(c)
        elif args.stage == "preflight": preflight(c)
        elif args.stage == "references": references(c)
        elif args.stage == "select": select(c)
        elif args.stage == "train":
            prediction = json.loads((c.out / "predictions.json").read_text())
            assert sha(c.out / "subsets/manifest.json") == prediction["subset_manifest_sha256"]
            from train import train
            train(c, args.arm)
        elif args.stage == "evaluate":
            c.require_preflight()
            Runtime(c, args.arm).generate(evaluation_jobs(c), f"evaluation_{args.arm}", temperature=1.0)
        elif args.stage == "judge": judge(c, args.target, args.arm)
        elif args.stage == "report": report(c)
        status = "complete"
    except Exception:
        append_row(c.out / "errors.jsonl", {"id": str(time.time_ns()), "stage": args.stage, "arm": args.arm, "traceback": traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        record = {"id": str(time.time_ns()), "stage": args.stage, "arm": args.arm, "status": status, "seconds": time.monotonic() - started}
        append_row(c.out / "timings.jsonl", record); print(json.dumps(record), flush=True)


if __name__ == "__main__": main()
