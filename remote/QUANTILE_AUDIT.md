# Quantile selection audit

This exploratory CPU analysis replaces the earlier absolute matching cutoffs with
rank-based selection. It reuses the 2,048 public answers and their saved text,
raw activation, token-length and fixed caveat scores. No new model inference,
API judgement or fine-tuning is part of this stage.

## Fixed analysis choices

The configuration is `quantile-audit-2026-10-28.json`. All tail fractions—5%, 10%,
20% and 30%—are reported as a sensitivity grid. No fraction is selected by
post-SFT results. For each fraction x and eligible pool of N answers, each arm
contains `floor(x * N)` examples. The analysis ranks independently by activation
projection and by LLM text score, and compares the bottom, top and a uniformly
random subset. Exact score ties use a deterministic hash of the seed and row ID;
boundary ties are recorded. Nearly identical scores can still have different
ranks, so rank separation is not evidence of a meaningful trait difference.

Four pools are examined: all answers, Normal only, Hallucination II only, and
answers with text score strictly below 1. The last follows the paper's text
filter but uses the existing synthetic QA pool and raw projections. It is an
adaptation, not a reproduction of the paper's real-world experiment.

Random controls are sampled from the entire eligible pool, not from the middle
after excluding the tails. Overlap with the tails is allowed. The first seeded
draw supplies candidate IDs. A further description of sampling variation uses
1,000 draws total per pool and fraction; its central 95% range concerns subset
mean scores, not uncertainty in a behavioural effect. The same random draws are
used for the two ranking metrics. Full-pool subsets can contain both answers to
one question; unique-question counts are reported. No candidate is a finalized
training dataset.

The audit reports score distributions, floor/ceiling rates, selected means,
standardized mean differences, empirical distribution distances and source-label
balance. Equal quantile sizes make selection fractions comparable, but do not
calibrate the scores or control semantic differences.

## Why include random selection?

High-versus-low SFT alone cannot distinguish whether high-score data increases a
trait, low-score data suppresses it, or both. A random subset of the same size
provides the ordinary-pool SFT comparator; the unfine-tuned base model supplies
the separate no-training reference. Identical training settings and multiple
seeds would still be needed to assess a behavioural effect.

The paper compares top 500, bottom 500 and 500 random examples using projection
difference. Its filtered experiment retains text scores below 1. Appendix L.2
reports incoherence and unrelated behavioural changes after training on extreme
subsets, including storytelling in response to ordinary questions. Any later
SFT stage should retain coherence, cap-hit and answer-length measurements, and
inspect a fixed sample of outputs alongside the hallucination score.

Source: [Persona Vectors, sections 6.3 and Appendix L](https://arxiv.org/html/2507.21509v1).

## Remaining paper-method comparison

For a supplied answer y to question q, the published score subtracts the base
model's natural-answer projection from the supplied-answer projection. Natural
answers and their projections are not available for this 1,024-question pool.
Their generation needs a separately scoped remote model run. Ranking by the
available raw projection cannot stand in for that measurement, particularly
when subsets contain different questions. Once obtained, rank agreement and
subset overlap between raw and reference-subtracted scores can be measured.

## Reproduction

On approved remote Linux CPU compute with Python 3.12, install `numpy==2.2.6`,
`scipy==1.15.3` and `matplotlib==3.10.3` into a virtual environment. Transfer the
two input files named in the JSON config and the script/config, preserving their
repository-relative paths. From the repository root:

```bash
export DATA_DEBUGGING_REMOTE=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python remote/quantile_audit.py --out runs/quantile-audit-2026-10-28
```

The output directory must be new. Source/config/input checksums and library
versions are recorded. The script checks sample counts, IDs, ranking order and
the text filter. These assertions validate implementation, not the scientific
adequacy of a selected comparison. Local execution is prohibited by `AGENTS.md`.
