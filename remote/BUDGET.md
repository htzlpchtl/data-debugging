# Compute estimates — 2026-10-21

**Measured scoring result:** the completed 1,000-token pilot took 30m 49s on one
A40, with 14.64 GiB peak allocated GPU memory. Its API cost was $0.11358 and its
elapsed GPU estimate was $0.30303 at the observed $0.59/hour. Including the earlier
diagnostic and work window through completion gives approximately $0.91, excluding
storage, earlier setup/allocation, and later idle time. See the
[research note](../results/remote_scoring_2026-10-21/RESEARCH_NOTE.md) and provenance
for the scope and exact accounting. SFT has not been benchmarked.

**Historical planning estimates follow.** The estimates cover
the two configurations supplied in this repository. They include generation,
activation scoring and, for the optional SFT pilot, training and evaluation.

## Workloads

| Work item | Scoring pilot | Optional SFT pilot |
|---|---:|---:|
| Vector-construction generations | 200 | 200 |
| Natural-answer generations | 64 | 512 |
| Supplied answers scored | 128 | 1,024 |
| Training runs | 0 | 3 |
| Examples per training run | — | 256 |
| Epochs per training run | — | 1 |
| Held-out evaluation generations | 0 | 240 |

At an assumed mean of 1,000 input tokens per training example, the optional SFT
pilot processes 3 × 256 × 1 × 1,000 = **768,000 training tokens**. The default
scoring configuration does not train the language model.

Scoring supplied text uses forward passes. Generating new answers adds sequential
decoding, and SFT requires backward passes. Throughput measurements for one of
these operations cannot be substituted directly for another.

One epoch follows the paper's synthetic-data training setting. Its separate
real-world subset experiment used ten epochs. Both configurations here use reduced
sampling; the SFT pilot also uses fewer training examples and has no training-seed
replication or text-matched selection. These differences limit comparisons with
the [published results](https://arxiv.org/html/2507.21509v1).

## Cost calculation

Runpod's [public pricing page](https://www.runpod.io/pricing), checked 2026-10-21,
advertises A40/RTX A6000 at **$0.59/hour**. This is not verified live stock, a Secure
Cloud quote, or an account-specific rate. A fresh quote is required at provisioning.

| Scope, each from a fresh start | Provisional allocated GPU time | Rental at $0.59/h | Practical total target |
|---|---:|---:|---:|
| Default 128-answer scoring pilot | 1–4 hours | $0.59–$2.36 | $2–$5 |
| Optional three-arm SFT pilot, including its scoring/evaluation | 3–10 hours | $1.77–$5.90 | $4–$10 |

These totals include modest setup/API/storage allowance, not unlimited debugging.
They are targets, not guarantees or statistical confidence bounds. The operator
must set a spending cap before provisioning resources.
If both independent configurations run, their costs add; shared model downloads
may reduce the setup component. Budget them as separate decisions.

The time estimates assume batched generation, roughly 30–100 generated tokens/s
in aggregate, and 200–700 processed training tokens/s with small-batch BF16 LoRA.
**Neither rate has been measured for this runner.** At the 512-token ceiling:

- Default: (200 + 64) × 512 = at most **135,168 generated tokens**,
  about 0.38–1.25 generation hours at those assumed aggregate rates.
- SFT pilot: (200 + 512 + 240) × 512 = at most **487,424 generated tokens**,
  about 1.35–4.51 generation hours. Training adds about 0.30–1.07 hours at the
  assumed 1,000-token mean and training rates.

Installation, downloads, repeated model loading, supplied-text scoring, evaluation
judging and idle time fill the remaining range. Sequential API judging on a GPU Pod
is billed GPU wall time, even though the GPU is unused. Separate remote CPU judging
can avoid that charge if convenient. Actual completion lengths may be well below
512; slower generation, dependency problems or retries may exceed these ranges.

The original [GPT-4.1 mini judge pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
currently lists $0.40/million input tokens and $1.60/million output tokens. The default
needs 658 calls including its two-call availability check: trait and coherence for
200 extraction answers and 128 candidates. At an illustrative 700 input tokens/call,
that is approximately **$0.19**. The optional pilot
needs 2,930 calls including checking and evaluation, approximately **$0.83** at the
same assumption. The model is marked deprecated; API access must be checked before
the experiment.
The code records actual usage and fails if the chosen judge lacks required logprobs.

At the listed $0.07/GB/month, 100 GB of standard network storage costs $7 **if retained
for the whole month**, roughly $0.23/day on a 30-day approximation. A short experiment
does not intrinsically require a month of retention. Container/Pod disk charges
are additional; choose and verify actual storage at provisioning.

## Spend discipline and scientific limits

First install, check judge access, run numerical preflight, and observe the first
generation batches. Use their logged tokens/second and cap-hit rates to recalculate
remaining time before proceeding. For training, use the logged token counts and
initial optimizer-step timings. Additional configurations and training seeds require
separate estimates.

The two-hour per-command alarm limits the process, not the Pod allocation. Budget
monitoring and verified GPU release are still operator responsibilities. Provider
timed-stop creation flags must not be relied on. No infrastructure is created by
this repository.

The reduced pilot can validate the scoring implementation and compare activation
rankings with text-only scores. It may show weak or absent SFT effects because the
training dose is small. Stronger evidence that activations add predictive value
requires text/length-matched subsets and additional training seeds; those experiments
are outside these estimates.
