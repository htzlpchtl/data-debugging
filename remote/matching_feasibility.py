"""One remote-only overlap check for existing text and persona projection scores."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

from run import ROOT, UPSTREAM, append_row, read_rows, remote_only, sha, write_json


def initialize(out, cfg):
    source = ROOT / cfg["source_run"]
    original = json.loads((source / "manifest.json").read_text())
    candidates = read_rows(source / "candidates.jsonl")
    projections = read_rows(source / "training_projection.jsonl")
    assert sha(source / "candidates.jsonl") == original["candidate_sha256"]
    assert len(candidates) == len(projections) == cfg["expected_answers"]
    assert len({r["question"] for r in candidates}) == cfg["expected_questions"]
    by_id = {r["id"]: r for r in projections}
    assert set(by_id) == {r["id"] for r in candidates}
    assert json.loads((source / "preflight.json").read_text())["passed"]
    heldout = {r["question"] for r in json.loads((source / "evaluation_jobs.json").read_text())}
    assert not heldout & {r["question"] for r in candidates}
    rubric_path = UPSTREAM / "data_generation/trait_data_extract/hallucinating.json"
    rubric = json.loads(rubric_path.read_text())["eval_prompt"]
    inputs = [{"id": r["id"], "pair_id": r["pair_id"], "label": r["label"],
               "question": r["question"], "answer": r["answer"],
               "projection": by_id[r["id"]]["projection"],
               "response_tokens": by_id[r["id"]]["response_tokens"]} for r in candidates]
    identity = {"config": cfg, "source_manifest_sha256": sha(source / "manifest.json"),
                "candidates_sha256": sha(source / "candidates.jsonl"),
                "projections_sha256": sha(source / "training_projection.jsonl"),
                "script_sha256": sha(Path(__file__)), "helper_sha256": sha(ROOT / "remote/run.py"),
                "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
                "vector_sha256": json.loads((source / "vector.json").read_text())["vector_sha256"],
                "resolved_model_revision": original["resolved_model_revision"],
                "upstream_commit": original["identity"]["upstream_commit"]}
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["identity"] == identity, "Frozen inputs changed"
        assert read_rows(out / "inputs.jsonl") == inputs
    else:
        write_json(manifest, {"identity": identity,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repository_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        (out / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
        (out / "judge_prompt.txt").write_text(rubric)
    return inputs, rubric


async def judge(out, cfg, rows, rubric):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=45)
    done = {r["id"] for r in read_rows(out / "judgements.jsonl")}
    assert done <= {r["id"] for r in rows}
    remaining = [r for r in rows if r["id"] not in done]
    spent = sum(r["estimated_usd"] for r in read_rows(out / "judge_usage.jsonl"))
    options = {"provider": {"only": [cfg["judge_provider"]], "allow_fallbacks": False,
        "require_parameters": False,
        "max_price": {"prompt": cfg["input_usd_per_million"], "completion": cfg["output_usd_per_million"]}}}

    async def request(row):
        prompt = rubric.format(question=row["question"], answer=row["answer"])
        return await client.chat.completions.create(model=cfg["judge_model"],
            messages=[{"role": "user", "content": prompt}], max_tokens=1, temperature=0,
            logprobs=True, top_logprobs=20, seed=0, extra_body=options)

    try:
        for start in range(0, len(remaining), cfg["judge_concurrency"]):
            batch = remaining[start:start + cfg["judge_concurrency"]]
            reserve = sum(((len(rubric.format(question=r["question"], answer=r["answer"]).encode()) + 64)
                * cfg["input_usd_per_million"] + 4 * cfg["output_usd_per_million"]) / 1e6 for r in batch)
            if spent + reserve > cfg["judge_budget_usd"]:
                raise RuntimeError("Conservative API budget reserve exhausted")
            responses = await asyncio.gather(*(request(r) for r in batch), return_exceptions=True)
            failed = []
            for row, result in zip(batch, responses):
                if isinstance(result, BaseException):
                    append_row(out / "request_errors.jsonl", {"id": str(time.time_ns()),
                        "row_id": row["id"], "error_type": type(result).__name__, "message": str(result)})
                    failed.append(row["id"])
                    continue
                raw = result.model_dump()
                append_row(out / "raw_responses.jsonl", {"id": result.id, "row_id": row["id"], "response": raw})
                if result.usage is None:
                    failed.append(row["id"])
                    continue
                usage = result.usage.model_dump()
                cost = (usage["prompt_tokens"] * cfg["input_usd_per_million"]
                        + usage["completion_tokens"] * cfg["output_usd_per_million"]) / 1e6
                spent += cost
                append_row(out / "judge_usage.jsonl", {"id": result.id, "row_id": row["id"],
                    "estimated_usd": cost, "usage": usage, "returned_model": result.model,
                    "returned_provider": getattr(result, "provider", None)})
                probs = result.choices[0].logprobs if result.choices else None
                top = probs.content[0].top_logprobs if probs and probs.content else []
                if len(top) != 20:
                    failed.append(row["id"])
                    continue
                numeric = []
                for token in top:
                    try:
                        n = int(token.token)
                    except ValueError:
                        continue
                    if 0 <= n <= 100:
                        numeric.append((n, math.exp(token.logprob)))
                mass = sum(p for _, p in numeric)
                score = sum(n * p for n, p in numeric) / mass if mass >= cfg["minimum_numeric_mass"] else None
                append_row(out / "judgements.jsonl", {"id": row["id"], "trait": score,
                    "numeric_mass": mass, "response_id": result.id})
            print(json.dumps({"judged": len(done) + min(start + len(batch), len(remaining)),
                              "estimated_usd": spent, "failed_ids": failed}), flush=True)
            if failed:
                raise RuntimeError("Judge batch incomplete; successful calls saved; review before resuming")
    finally:
        await client.close()


def analyze(out, cfg, rows):
    import numpy as np
    from scipy.stats import spearmanr, ks_2samp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    judged = {r["id"]: r for r in read_rows(out / "judgements.jsonl")}
    assert set(judged) == {r["id"] for r in rows}, "Judging must finish before matching"
    data = [{**r, "text_score": judged[r["id"]]["trait"]} for r in rows if judged[r["id"]]["trait"] is not None]
    (out / "scored_pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    text = np.array([r["text_score"] for r in data])
    activation = np.array([r["projection"] for r in data])
    length = np.array([r["response_tokens"] for r in data])
    label = np.array([r["label"] for r in data])
    question = np.array([r["pair_id"] for r in data])
    ii, jj = np.triu_indices(len(data), k=1)
    dt = np.abs(text[ii] - text[jj]); da = np.abs(activation[ii] - activation[jj])
    lr = np.maximum(length[ii], length[jj]) / np.minimum(length[ii], length[jj])
    base_mask = lr <= cfg["maximum_response_length_ratio"]
    records, case_files = [], {}

    def stats(a, b):
        sd = np.sqrt((np.var(a) + np.var(b)) / 2)
        delta = float(np.mean(b) - np.mean(a))
        return {"low_mean": float(np.mean(a)), "high_mean": float(np.mean(b)),
                "high_minus_low": delta, "smd": delta / float(sd) if sd > 1e-12 else None,
                "ks_distance": float(ks_2samp(a, b).statistic)}

    def match(direction, scope, caliper, gap, primary):
        mask = base_mask.copy()
        if scope == "same_question": mask &= question[ii] == question[jj]
        elif scope == "same_source": mask &= label[ii] == label[jj]
        if direction == "activation":
            mask &= (dt <= caliper) & (da >= gap)
            target, control = da, dt
            values = activation
        else:
            mask &= (da <= caliper) & (dt >= gap)
            target, control = dt, da
            values = text
        edges = np.flatnonzero(mask)
        # Frozen deterministic greedy rule: largest target gap, then nearest controls.
        order = edges[np.lexsort((jj[edges], ii[edges], lr[edges], control[edges], -target[edges]))]
        used_questions, pairs = set(), []
        for edge in order:
            i, j = int(ii[edge]), int(jj[edge])
            qi, qj = int(question[i]), int(question[j])
            if qi in used_questions or qj in used_questions: continue
            lo, hi = (i, j) if values[i] < values[j] else (j, i)
            pairs.append((lo, hi)); used_questions.update((qi, qj))
        case = f'{direction}_{scope}_caliper{caliper:g}_gap{gap:g}'
        record = {"case": case, "vary": direction, "scope": scope, "control_caliper": caliper,
                  "minimum_target_gap": gap, "eligible_edges": len(edges), "matched_pairs": len(pairs),
                  "primary": primary, "algorithm": "Greedy without question reuse; not a maximum-matching count"}
        if pairs:
            a, b = np.array(pairs).T
            record.update(text=stats(text[a], text[b]), activation=stats(activation[a], activation[b]),
                length=stats(length[a], length[b]),
                low_fraction_hallucination_II=float(label[a].mean()), high_fraction_hallucination_II=float(label[b].mean()),
                low_text_saturation_fraction=float(np.mean((text[a] <= 1) | (text[a] >= 99))),
                high_text_saturation_fraction=float(np.mean((text[b] <= 1) | (text[b] >= 99))))
            record["large_enough_for_128_example_arms"] = len(pairs) >= cfg["minimum_pairs_for_candidate"]
            payload = [{"id": str(k), "low": data[i]["id"], "high": data[j]["id"]} for k, (i, j) in enumerate(pairs)]
            destination = out / "candidate_matches"; destination.mkdir(exist_ok=True)
            (destination / f"{case}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in payload))
            case_files[case] = str(Path("candidate_matches") / f"{case}.jsonl")
        records.append(record)

    for scope in ("same_question", "same_source", "any_source"):
        match("activation", scope, cfg["primary_text_caliper"], cfg["primary_activation_gap"], True)
        match("text", scope, cfg["primary_activation_caliper"], cfg["primary_text_gap"], True)
    for caliper in (1.0, 10.0): match("activation", "same_source", caliper, 3.0, False)
    match("activation", "same_source", 5.0, 5.0, False)
    for caliper in (0.25, 1.0): match("text", "any_source", caliper, 25.0, False)
    write_json(out / "matching_results.json", {"cases": records, "files": case_files})
    def corr(a,b):
        r = float(spearmanr(a,b).statistic)
        return r if math.isfinite(r) else None
    usage = read_rows(out / "judge_usage.jsonl")
    summary = {"status": "FEASIBILITY_MEASURED_NO_NEW_SFT", "answers": len(rows), "valid_scores": len(data),
        "missing_scores": len(rows)-len(data), "pooled_spearman": corr(text,activation),
        "text_score_bands": {"at_most_1": int((text <= 1).sum()), "between_1_and_99": int(((text > 1)&(text < 99)).sum()), "at_least_99": int((text >= 99).sum())},
        "by_source": {}, "judge_calls": len(usage),
        "judge_cost_estimate_usd": sum(r["estimated_usd"] for r in usage),
        "judge_reported_usd": sum(r["usage"].get("cost",r["estimated_usd"]) for r in usage),
        "limitations": ["Raw projection rather than per-question reference-subtracted score; cross-question matches require further checking.",
            "Text-judge ceilings/floors can hide semantic differences; matching a scalar is not controlling all textual information.",
            "Same-source matching controls source-label mixture, not question topic. Topic balance has not been validated.",
            "Greedy feasible counts are lower bounds on possible matchings; small counts with many eligible edges do not prove impossibility.",
            "Existing pool used in preceding SFT pilot; this is exploratory design work, not a new prospective behavioural test."]}
    for value, name in ((0,"normal"),(1,"hallucination_II")):
        mask=label==value
        summary["by_source"][name] = {"n":int(mask.sum()),"mean_text_score":float(text[mask].mean()),
            "mean_projection":float(activation[mask].mean()), "spearman":corr(text[mask],activation[mask])}
    write_json(out / "summary.json", summary)
    fig,ax=plt.subplots(figsize=(8,5.5),layout="constrained")
    for value,name,color in ((0,"Normal","#377eb8"),(1,"Hallucination II","#e78332")):
        mask=label==value
        ax.scatter(text[mask],activation[mask],s=12,alpha=.35,color=color,label=name,rasterized=True)
    ax.set(xlabel="LLM text-judge hallucination score (0–100)",ylabel="Base-model layer-16 raw projection",
        title="Do text and activation scores leave room for matched comparisons?",xlim=(-3,103))
    ax.legend();ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"score_overlap.png",dpi=180);fig.savefig(out/"score_overlap.pdf");plt.close(fig)
    lines=["| Varied score | Matching scope | Eligible edges | Matched pairs | Text gap | Activation gap | Length SMD |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        if not r["primary"]:continue
        if r["matched_pairs"]:
            lines.append(f'| {r["vary"]} | {r["scope"]} | {r["eligible_edges"]} | {r["matched_pairs"]} | {r["text"]["high_minus_low"]:.2f} | {r["activation"]["high_minus_low"]:.2f} | {r["length"]["smd"]:.3f} |')
        else:lines.append(f'| {r["vary"]} | {r["scope"]} | {r["eligible_edges"]} | 0 | — | — | — |')
    (out/"feasibility_table.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2));print("\n".join(lines))


def main():
    remote_only()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["init","judge","analyze"])
    parser.add_argument("--config",default="remote/matching-feasibility-2026-10-28.json")
    parser.add_argument("--out",default="runs/matching-feasibility-2026-10-28")
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text())
    started=time.monotonic();status="failed"
    try:
        rows,rubric=initialize(out,cfg)
        if args.stage=="judge":asyncio.run(judge(out,cfg,rows,rubric))
        elif args.stage=="analyze":analyze(out,cfg,rows)
        status="complete"
    except Exception:
        append_row(out/"errors.jsonl",{"id":str(time.time_ns()),"stage":args.stage,"traceback":traceback.format_exc()})
        raise
    finally:
        append_row(out/"timings.jsonl",{"id":str(time.time_ns()),"stage":args.stage,"status":status,"seconds":time.monotonic()-started})


if __name__=="__main__":main()
