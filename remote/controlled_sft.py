"""One remote-only, paired Normal versus Hallucination II SFT experiment.

Reuses the checked runtime, original public prompts/rubrics and PEFT trainer.
Training conditions are selected by source label, never by observed evaluation.
"""
import argparse
import json
import math
import shutil
import signal
import subprocess
import time
import traceback

from run import (Context, ROOT, Runtime, append_row, evaluation_jobs, initialize,
                 judge, preflight, remote_only, sha, write_json)


def prepare(c):
    initialize(c)
    rows = c.rows("candidates")
    arms = {name: [r for r in rows if r["label"] == label]
            for name, label in (("normal", 0), ("trait", 1))}
    assert [r["question"] for r in arms["normal"]] == [r["question"] for r in arms["trait"]]
    heldout = set(c.evaluation["questions"] + c.extract["questions"])
    assert not heldout & {r["question"] for r in rows}
    path = c.out / "subsets"
    path.mkdir(exist_ok=True)
    manifest = {}
    for name, chosen in arms.items():
        assert len(chosen) == c.cfg["training_subset_size"]
        destination = path / f"{name}.jsonl"
        serialized = "".join(json.dumps(r) + "\n" for r in chosen)
        if destination.exists() and destination.read_text() != serialized:
            raise ValueError("Existing training subset differs.")
        destination.write_text(serialized)
        manifest[name] = {"n": len(chosen), "sha256": sha(destination),
                          "pair_ids": [r["pair_id"] for r in chosen]}
    manifest["matching"] = "Identical questions, row order, example count, epochs, optimizer settings and training seed. Response content is the treatment; token totals are measured, not forced equal."
    manifest["heldout_question_overlap"] = 0
    write_json(path / "manifest.json", manifest)
    write_json(c.out / "evaluation_jobs.json", evaluation_jobs(c))
    write_json(c.out / "protocol.json", {
        "status": "FROZEN_BEFORE_BASELINE_AND_TRAINING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "primary_contrast": "Mean held-out hallucination score: trait SFT minus normal SFT",
        "secondary_contrasts": ["normal minus base", "trait minus base"],
        "evaluation": "Original 20 hallucinating evaluation questions; fixed repeats, sampling seeds, temperature 1 and token cap for every arm",
        "uncertainty": "Question-paired bootstrap; one training seed, conditional uncertainty only",
        "hypothesis": "Hallucination II SFT increases held-out hallucination relative to Normal SFT",
        "not_tested": "Incremental prediction beyond text; robustness across training seeds; calibrated factual accuracy",
        "projection_priority": "Only after baseline and both post-SFT evaluations; numerical parity required",
        "config": c.cfg,
    })


def evaluate(c, arm):
    c.require_preflight()
    Runtime(c, arm).generate(evaluation_jobs(c), f"evaluation_{arm}", temperature=1.0)
    if arm == "base":
        write_json(c.out / "baseline_lock.json", {
            "evaluation_sha256": sha(c.out / "evaluation_base.jsonl"),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


def project(c):
    for arm in ("base", "normal", "trait"):
        c.complete(f"judge_evaluation_{arm}", evaluation_jobs(c))
    c.require_preflight()
    saved = ROOT / "results/remote_scoring_2026-10-21"
    old_manifest = json.loads((saved / "manifest.json").read_text())
    if c.lock["resolved_model_revision"] != old_manifest["resolved_model_revision"]:
        raise ValueError("Saved vector belongs to a different model revision.")
    vector_meta = json.loads((saved / "summary.json").read_text())["vector"]
    if sha(saved / "unit_vector.pt") != vector_meta["vector_sha256"]:
        raise ValueError("Saved vector checksum mismatch.")
    shutil.copyfile(saved / "unit_vector.pt", c.out / "unit_vector.pt")
    write_json(c.out / "vector.json", vector_meta)
    runtime = Runtime(c)
    direction = runtime.torch.load(c.out / "unit_vector.pt", weights_only=True).float()
    if not runtime.torch.isfinite(direction).all() or abs(float(direction.norm()) - 1) > 1e-5:
        raise ValueError("Invalid saved unit vector.")
    done = {r["id"] for r in c.rows("training_projection")}
    for i, row in enumerate(c.rows("candidates")):
        if row["id"] in done:
            continue
        mean, count = runtime.response_mean(runtime.prompt(row["messages"][:-1]), row["answer"])
        append_row(c.out / "training_projection.jsonl", {
            "id": row["id"], "pair_id": row["pair_id"], "label": row["label"],
            "projection": float(mean @ direction), "response_tokens": count})
        if (i + 1) % 64 == 0:
            print(json.dumps({"projected": i + 1}), flush=True)
    c.complete("training_projection", c.rows("candidates"))


def report(c):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jobs = evaluation_jobs(c)
    rng = np.random.default_rng(c.cfg["seed"])
    results, by_question, paired = {}, {}, {}
    for arm in ("base", "normal", "trait"):
        generation = {r["id"]: r for r in c.complete(f"evaluation_{arm}", jobs)}
        judged = c.complete(f"judge_evaluation_{arm}", jobs)
        groups = {}
        for row in judged:
            qid = generation[row["id"]]["pair_id"]
            if row["trait"] is not None:
                groups.setdefault(qid, []).append(row["trait"])
        by_question[arm] = {q: float(np.mean(scores)) for q, scores in groups.items()}
        means = np.array(list(by_question[arm].values()))
        if not len(means):
            raise ValueError(f"No valid trait judgements for {arm}")
        draws = means[rng.integers(len(means), size=(2000, len(means)))].mean(axis=1)
        coherence = [r["coherence"] for r in judged if r["coherence"] is not None]
        results[arm] = {
            "mean_trait": float(means.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "mean_coherence": float(np.mean(coherence)) if coherence else None,
            "questions": len(means), "responses": len(generation),
            "missing_trait": sum(r["trait"] is None for r in judged),
            "missing_coherence": len(judged) - len(coherence),
            "cap_hits": sum(r["hit_token_cap"] for r in generation.values()),
            "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in generation.values()])),
        }
        paired[arm] = {}
        for row in judged:
            if row["trait"] is not None:
                qid = generation[row["id"]]["pair_id"]
                paired[arm].setdefault(qid, {})[row["id"]] = row["trait"]
    contrasts = {}
    for left, right in (("trait", "normal"), ("trait", "base"), ("normal", "base")):
        differences = []
        for qid in sorted(paired[left].keys() & paired[right].keys()):
            ids = paired[left][qid].keys() & paired[right][qid].keys()
            if ids:
                differences.append(np.mean([paired[left][qid][k] - paired[right][qid][k] for k in ids]))
        differences = np.array(differences)
        draws = differences[rng.integers(len(differences), size=(2000, len(differences)))].mean(axis=1)
        contrasts[f"{left}_minus_{right}"] = {"mean": float(differences.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "paired_questions": len(differences)}
    projection = {}
    rows = c.rows("training_projection")
    if rows:
        c.complete("training_projection", c.rows("candidates"))
        for label, name in ((0, "normal"), (1, "trait")):
            projection[name] = float(np.mean([r["projection"] for r in rows if r["label"] == label]))
        projection["method"] = "Published raw mean-response activation projection onto the saved unit persona vector, layer 16. Same paired questions mean the natural-reference subtraction cancels in the between-condition mean contrast. Individual projection differences were not computed."
    summary = {"status": "BEHAVIOURAL_MEASUREMENTS_COMPLETE", "arms": results,
               "contrasts": contrasts, "training_projection": projection,
               "per_question_trait_means": by_question,
               "limitations": c.cfg["deviations"],
               "inference": "One matched data intervention and one seed; does not establish incremental prediction over text or a general score-to-effect relationship."}
    write_json(c.out / "behaviour_summary.json", summary)
    labels = ["Base", "Normal SFT", "Hallucination II SFT"]
    means = [results[a]["mean_trait"] for a in results]
    errors = np.array([[results[a]["mean_trait"] - results[a]["ci95"][0],
                        results[a]["ci95"][1] - results[a]["mean_trait"]] for a in results]).T
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    ax.bar(labels, means, color=["#7a8793", "#377eb8", "#e78332"], yerr=errors, capsize=5)
    ax.set(ylabel="Mean held-out hallucination score (0–100)", ylim=(0, 100),
           title="Qwen2.5-7B: paired Normal versus Hallucination II SFT")
    for i, mean in enumerate(means):
        ax.text(i, mean + 2, f"{mean:.1f}", ha="center")
    fig.savefig(c.out / "behaviour.png", dpi=180)
    fig.savefig(c.out / "behaviour.pdf")
    plt.close(fig)
    table = ["| Model | Hallucination mean (95% CI) | Coherence | Cap hits |", "|---|---:|---:|---:|"]
    for arm, r in results.items():
        lo, hi = r["ci95"]
        table.append(f"| {arm} | {r['mean_trait']:.2f} ({lo:.2f}–{hi:.2f}) | {r['mean_coherence']:.2f} | {r['cap_hits']}/{r['responses']} |")
    effect = contrasts["trait_minus_normal"]
    note = ["# Controlled hallucination SFT pilot", "", *table, "",
        f"Trait-minus-normal paired difference: {effect['mean']:.2f} points; 95% question-bootstrap interval {effect['ci95']}.",
        "", "![Held-out behaviour](behaviour.png)", "",
        "Completed: held-out baseline generation, two paired-data SFT arms, and identical held-out post-SFT evaluation. See raw outputs, manifests, trainer logs, API usage and numerical checks in this run.",
        "", "Intervals condition on one training seed and the fixed evaluation; 20 questions are the independent resampling units. The judge measures the published trait rubric, not externally verified factual accuracy.",
        "", "Projection: " + (json.dumps(projection) if projection else "Not completed; behavioural evaluation was prioritized."),
        "", "Deviations and limitations:", "", *[f"- {v}" for v in c.cfg["deviations"]], "",
        "Hypotheses not established: additional predictive value beyond text, robustness across seeds, or generalization beyond this dataset/model. No result should be described as a full reproduction of the paper.", ""]
    (c.out / "RESEARCH_NOTE.md").write_text("\n".join(note))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "preflight", "evaluate", "judge", "train", "project", "report"])
    parser.add_argument("--config", default="remote/controlled-sft-2026-10-28.json")
    parser.add_argument("--out", default="runs/controlled-sft-2026-10-28")
    parser.add_argument("--arm", choices=["base", "normal", "trait"], default="base")
    parser.add_argument("--target", choices=["check", "evaluation"], default="evaluation")
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    remote_only()
    c = Context(args)
    c.check_manifest(initialize=args.stage == "init")
    if args.max_seconds <= 0:
        raise ValueError("Positive per-stage time limit required")
    def timeout(*_):
        raise TimeoutError("Stage deadline reached; GPU billing must be stopped separately")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    started, status = time.monotonic(), "failed"
    try:
        if args.stage == "init": prepare(c)
        elif args.stage == "preflight": preflight(c)
        elif args.stage == "evaluate": evaluate(c, args.arm)
        elif args.stage == "judge": judge(c, args.target, args.arm)
        elif args.stage == "train":
            from train import train
            frozen = json.loads((c.out / "baseline_lock.json").read_text())
            if sha(c.out / "evaluation_base.jsonl") != frozen["evaluation_sha256"]:
                raise ValueError("Baseline evaluation changed")
            train(c, args.arm)
        elif args.stage == "project": project(c)
        elif args.stage == "report": report(c)
        status = "complete"
    except Exception:
        append_row(c.out / "errors.jsonl", {"stage": args.stage, "arm": args.arm,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "traceback": traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        record = {"id": str(time.time_ns()), "stage": args.stage, "arm": args.arm,
                  "target": args.target, "status": status, "seconds": time.monotonic() - started}
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            record.update(gpu=torch.cuda.get_device_name(0), peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        append_row(c.out / "timings.jsonl", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
