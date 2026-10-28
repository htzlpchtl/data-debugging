"""Remote CPU audit of rank-based subsets using already saved public scores."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def analyze(root, config_path, out):
    import numpy as np
    import scipy
    from scipy.stats import ks_2samp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.time()
    cfg = json.loads(config_path.read_text())
    source = root / cfg["source_scores"]
    lexical = root / cfg["source_caveats"]
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "executed_source"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(config_path, snapshot / config_path.name)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg, "script_sha256": digest(Path(__file__)),
        "config_sha256": digest(config_path),
        "input_sha256": {cfg["source_scores"]: digest(source), cfg["source_caveats"]: digest(lexical)},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "projection_kind": "raw response-token projection; no reference subtraction",
        "new_model_calls": 0, "new_judge_calls": 0, "new_sft_runs": 0,
    }
    save(out / "manifest.json", manifest)
    data = rows(source)
    caveats = {r["id"]: r for r in rows(lexical)}
    assert len(data) == cfg["expected_answers"]
    assert len({r["id"] for r in data}) == len(data)
    assert len({r["question"] for r in data}) == cfg["expected_questions"]
    assert set(caveats) == {r["id"] for r in data}
    for row in data:
        original = caveats[row["id"]]
        assert original["projection"] == row["projection"] and original["label"] == row["label"]
        row["caveat_count"] = -original["lexical_score"]
        assert row["caveat_count"] >= 0
        assert 0 <= row["text_score"] <= 100
        assert math.isfinite(row["projection"]) and row["response_tokens"] > 0
    (out / "scored_pool.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in data))
    features = ["projection", "text_score", "response_tokens", "caveat_count", "label"]
    pools = {
        "all": data,
        "normal": [r for r in data if r["label"] == 0],
        "hallucination_II": [r for r in data if r["label"] == 1],
        "text_below_1": [r for r in data if r["text_score"] < cfg["paper_text_filter_exclusive"]],
    }
    assert set(pools) == set(cfg["pools"])
    cases, pool_summaries, checks = [], {}, []
    selections = out / "selections"
    selections.mkdir()

    def stable_seed(*parts):
        text = ":".join(map(str, (cfg["random_seed"],) + parts))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def summary(group):
        result = {"n": len(group), "unique_questions": len({r["question"] for r in group})}
        for field in features:
            a = np.array([r[field] for r in group], dtype=float)
            result[field] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                             "quantiles": dict(zip(["min", "p05", "p25", "p50", "p75", "p95", "max"],
                                                  map(float, np.quantile(a, [0, .05, .25, .5, .75, .95, 1]))))}
        result["fraction_text_below_1"] = sum(r["text_score"] < 1 for r in group) / len(group)
        result["fraction_text_at_least_99"] = sum(r["text_score"] >= 99 for r in group) / len(group)
        result["fraction_text_below_0_01"] = sum(r["text_score"] < .01 for r in group) / len(group)
        return result

    def contrast(high, low):
        result = {}
        for field in features:
            a = np.array([r[field] for r in high], dtype=float)
            b = np.array([r[field] for r in low], dtype=float)
            diff = float(a.mean() - b.mean())
            denom = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
            result[field] = {"mean_difference": diff, "smd": diff / denom if denom else None,
                             "ks_distance": float(ks_2samp(a, b, method="asymp").statistic)}
        return result

    with (out / "random_draw_summaries.jsonl").open("w") as random_file:
        for pool_name, pool in pools.items():
            assert len(pool) > 10
            pool_summaries[pool_name] = summary(pool)
            array = np.array([[r[f] for f in features] for r in pool], dtype=float)
            for fraction in cfg["tail_fractions"]:
                k = math.floor(len(pool) * fraction)
                assert 2 <= k and 2 * k <= len(pool)
                rng = np.random.default_rng(stable_seed(pool_name, fraction))
                random_means, random_ids = [], None
                for draw in range(cfg["random_replicates"]):
                    indices = rng.choice(len(pool), size=k, replace=False)
                    means = array[indices].mean(axis=0)
                    random_means.append(means)
                    if draw == 0:
                        random_ids = list(map(int, indices))
                    record = {"pool": pool_name, "fraction": fraction, "draw": draw,
                              "means": dict(zip(features, map(float, means)))}
                    random_file.write(json.dumps(record, allow_nan=False) + "\n")
                random_group = [pool[i] for i in random_ids]
                random_means = np.array(random_means)
                envelope = {f: {"mean": float(random_means[:, j].mean()),
                                "p025": float(np.quantile(random_means[:, j], .025)),
                                "p975": float(np.quantile(random_means[:, j], .975))}
                            for j, f in enumerate(features)}
                for metric in cfg["ranking_metrics"]:
                    ordered = sorted(pool, key=lambda r: (r[metric], stable_seed(r["id"])))
                    low, high = ordered[:k], ordered[-k:]
                    case_name = f"{pool_name}_{metric}_{round(fraction * 100):02d}pct"
                    groups = {"low": low, "high": high, "random": random_group}
                    group_ids = {arm: [r["id"] for r in group] for arm, group in groups.items()}
                    assert all(len(v) == len(set(v)) == k for v in group_ids.values())
                    assert not set(group_ids["low"]) & set(group_ids["high"])
                    assert max(r[metric] for r in low) <= min(r[metric] for r in high)
                    if pool_name == "text_below_1":
                        assert all(r["text_score"] < 1 for group in groups.values() for r in group)
                    boundary = {}
                    for arm, value in [("low", low[-1][metric]), ("high", high[0][metric])]:
                        boundary[arm] = {"score": value, "pool_rows_equal_to_boundary": sum(r[metric] == value for r in pool),
                                         "selected_rows_equal_to_boundary": sum(r[metric] == value for r in groups[arm])}
                    record = {"case": case_name, "pool": pool_name, "ranking_metric": metric,
                              "fraction": fraction, "n_pool": len(pool), "k": k,
                              "arms": {arm: summary(group) for arm, group in groups.items()},
                              "high_minus_low": contrast(high, low), "random_mean_envelope": envelope,
                              "boundary_ties": boundary,
                              "overlap_random_low": len(set(group_ids["random"]) & set(group_ids["low"])),
                              "overlap_random_high": len(set(group_ids["random"]) & set(group_ids["high"]))}
                    save(selections / f"{case_name}.json", {"case": case_name, "candidate_only": True, "ids": group_ids})
                    cases.append(record)
                    checks.append({"case": case_name, "equal_sizes_unique_ids_ordering_and_filter": True})
    save(out / "pool_summaries.json", pool_summaries)
    save(out / "quantile_results.json", cases)
    save(out / "verification.json", {"all_passed": True, "cases": checks,
                                    "input_rows": len(data), "notes": "Random controls may overlap either tail. Full-pool subsets can include both answers to a question; unique-question counts are reported. This is not matched SFT."})
    primary = [r for r in cases if r["pool"] == "text_below_1" and r["ranking_metric"] == "projection"]
    lines = ["| Tail fraction | Examples/arm | Projection low / random / high | Text-score low / high | Caveats low / high | Length SMD (high−low) |",
             "|---|---:|---|---|---|---:|"]
    for record in primary:
        arms = record["arms"]
        triplet = " / ".join(f"{arms[a]['projection']['mean']:.2f}" for a in ["low", "random", "high"])
        text = " / ".join(f"{arms[a]['text_score']['mean']:.4g}" for a in ["low", "high"])
        caveat = " / ".join(f"{arms[a]['caveat_count']['mean']:.2f}" for a in ["low", "high"])
        lines.append(f"| {record['fraction']:.0%} | {record['k']} | {triplet} | {text} | {caveat} | {record['high_minus_low']['response_tokens']['smd']:.2f} |")
    (out / "quantile_table.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    x = [r["fraction"] * 100 for r in primary]
    for axis, feature, label in zip(axes.flat, features[:4], ["Raw activation projection", "LLM text score (0–100)", "Response tokens", "Caveat phrase count"]):
        for arm, color in [("low", "#d17b25"), ("random", "#666666"), ("high", "#176b99")]:
            axis.plot(x, [r["arms"][arm][feature]["mean"] for r in primary], "o-", label=arm, color=color)
        axis.fill_between(x, [r["random_mean_envelope"][feature]["p025"] for r in primary],
                          [r["random_mean_envelope"][feature]["p975"] for r in primary], color="#777777", alpha=.18,
                          label="95% random-subset range")
        axis.set(xlabel="Selected fraction of eligible pool (%)", ylabel=label, xticks=x)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Activation-ranked tails among answers with text score <1\nSaved raw scores; no new SFT or reference subtraction", fontsize=12)
    fig.savefig(out / "quantile_balance.png", dpi=180)
    fig.savefig(out / "quantile_balance.pdf")
    plt.close(fig)
    save(out / "completion.json", {"status": "complete", "elapsed_seconds": time.time() - started,
                                  "cases": len(cases), "random_draws": len(pools) * len(cfg["tail_fractions"]) * cfg["random_replicates"],
                                  "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print((out / "quantile_table.md").read_text(), flush=True)


if __name__ == "__main__":
    if platform.system() != "Linux" or os.environ.get("DATA_DEBUGGING_REMOTE") != "1":
        raise SystemExit("Experimental analysis is remote-only; run on approved Linux compute.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("remote/quantile-audit-2026-10-28.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/quantile-audit-2026-10-28"))
    args = parser.parse_args()
    try:
        analyze(args.root, args.config, args.out)
    except Exception:
        if args.out.exists():
            (args.out / "error.txt").write_text(traceback.format_exc())
        raise
