# Remote hallucination pilot

Status: **remote environment installed; OpenRouter judge compatibility checked**.
The target model has not been loaded; GPU preflight, generation, activation scoring
and SFT remain unexecuted. See [OPENROUTER_CHECK.md](OPENROUTER_CHECK.md) for the
two-call API check and its routing discrepancy.
The [2026-10-21 checkpoint](CHECKPOINT_2026-10-21.md) records completed evidence,
prepared files and the proposed next experiment; further execution awaits approval.
The completed text-baseline evidence is stored in `results/`. The remote pipeline
has not produced activation or SFT results. All experimental execution is
restricted to remote resources under the project instructions in `AGENTS.md`.

This is a small reproduction with explicit stages, not a general framework. The
default produces activation scores, a text-baseline comparison, a figure and a
short note. An optional config adds a small synthetic-data fine-tuning check.
Neither config establishes incremental prediction beyond semantic content.

## Hardware and environment

- One A40 or RTX A6000, 48 GB VRAM, about 50 GB system RAM, 8+ vCPUs.
- A Linux/Python 3.11 or 3.12 PyTorch Pod; no custom image is required.
- Candidate official base: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
  Confirm its current availability and the host CUDA/driver compatibility at
  deployment; this image has not been tested by this project.
- 20–30 GB container disk and 60–100 GB persistent `/workspace` storage. Keep the
  model cache, virtual environment and outputs there. No need to save all token
  activations or merged copies of the model.
- `setup.sh` creates an isolated environment with PyTorch 2.6.0/cu124 and pinned
  direct dependencies. These are candidate pins, not a validated full lockfile.
  `pip check` and the installed freeze are recorded remotely.

The scripts refuse macOS and require `DATA_DEBUGGING_REMOTE=1` before any ML import.
GPU stages require BF16 CUDA with at least 40 GiB reported memory and have no
CPU/MPS fallback. The environment flag enables remote execution; resource approval
is managed separately.

## Before allocating a GPU

Each run requires an approved scope and spending cap. GPU availability and pricing
must be verified at provisioning. The repository must be cloned or transferred
to the remote machine, with SSH access configured for the operator. The experiment
scripts do not provision infrastructure.

Supply `OPENAI_API_KEY` securely in the remote environment for judge stages only.
The optional OpenRouter check below uses `OPENROUTER_API_KEY` instead.
Public Qwen weights do not require an HF token. Do not put credentials in JSON,
source files, Git, logs, or command history. Runpod credentials are not needed by
the experiment process itself.

The original judge snapshot is configured deliberately. Official documentation
now labels GPT-4.1 mini deprecated; availability must be checked using the small
`judge --target check` stage. If unavailable, stop and choose/document a replacement
and its rates in a new config. Do not silently substitute a model or scoring rule.

## OpenRouter judge compatibility check

`remote/openrouter-check.json` records the initial strict compatibility check. It
permits only `init` and `judge --target check`, with a $0.01 estimated API budget.
It does not load Qwen, construct a vector or authorize the scoring/SFT pilots.
OpenRouter credits pay for API calls; GPU allocation still bills independently.

The route uses `openai/gpt-4.1-mini`, pins the `openai` provider, disables provider
fallbacks and automatic client retries, and caps provider token prices at
$0.40/$1.60 per million input/output tokens. The public endpoint catalog names
the 2025-04-14 snapshot, but the request uses an alias. Requested and returned
model/provider information is recorded with usage. This is a routing deviation
from the direct OpenAI configuration, not a claim of identical responses.

OpenRouter's public endpoint metadata did not advertise `logprobs` support when
inspected on 2026-10-21. The initial check sets `require_parameters: true` and
requires 20 returned top-token log probabilities plus sufficient numeric mass
for each of the two original rubrics. A successful HTTP response or a printed
numeric score alone is insufficient. Failure must not trigger a different judge
or substitute plain-text scoring. Any diagnostic routing change needs its own
config and output directory so evidence is not mixed.

The strict check returned HTTP 404 at the parameter-filtering step. A separately
recorded diagnostic with `require_parameters: false` then returned all 20 requested
log probabilities for both rubrics and passed the numeric-mass checks. The total
API charge was $0.0002544. Disabling this catalog filter did not disable response
validation or change the scoring formula. Full details and limits are in the
[compatibility note](OPENROUTER_CHECK.md).

After placing the key in the approved remote environment, run on that remote
machine only:

```bash
export DATA_DEBUGGING_REMOTE=1
source .venv-cuda/bin/activate
python remote/run.py init --config remote/openrouter-check.json --out runs/openrouter-check
python remote/run.py judge --target check --config remote/openrouter-check.json --out runs/openrouter-check --max-seconds 120
```

The budget ledger uses token counts and configured prices; it is not a
provider-enforced account spending limit. The compatibility check is at most two
one-token calls without automatic retries. Keep the original configurations and
completed text-baseline evidence unchanged. A full OpenRouter pilot would use a
new configuration and output directory. `remote/scoring-pilot-openrouter.json`
prepares that pilot, with the resolved Qwen revision pinned, a $1 estimated API
budget, catalog filtering disabled, and response validation retained. It has not
been executed and requires approval for GPU computation and spending. Use
`--config remote/scoring-pilot-openrouter.json --out runs/scoring-pilot-openrouter`
on every stage instead of the default config/output directory.

References: [OpenRouter SDK integration](https://openrouter.ai/docs/guides/community/openai-sdk),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
and [GPT-4.1 mini endpoints](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini/endpoints).

## First remote session: scoring pilot

Run from the transferred/cloned repository on the approved Linux machine:

```bash
export DATA_DEBUGGING_REMOTE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/huggingface
export MPLCONFIGDIR=/workspace/matplotlib
export TOKENIZERS_PARALLELISM=false
bash remote/setup.sh
source .venv-cuda/bin/activate
python remote/run.py init
python remote/run.py judge --target check
python remote/run.py preflight
```

`init` resolves `model_ref` to an immutable Hugging Face commit and writes it to
`runs/scoring-pilot/manifest.json` before weights are loaded. It checks the upstream
commit/archive checksum and freezes the candidate data and source hashes. Reusing
the output directory reuses that commit. To compare independent runs exactly,
copy the resolved SHA into the next config's `model_ref` before initializing it.

`preflight` loads the model, compares response-token means for two answers against
the pinned official `get_hidden_p_and_r` implementation, and records numerical
errors, environment and peak GPU memory. It does not train. Failure blocks later
GPU stages. Generation and training compatibility remain unverified until their
own first remote calls.

After reviewing the preflight, within the approved pilot budget:

```bash
python remote/run.py generate
python remote/run.py judge --target extraction
python remote/run.py vector
python remote/run.py score
python remote/run.py judge --target sample
python remote/run.py report
```

Each command has a two-hour process limit by default; `--max-seconds` changes that
per-command limit. Watch the generation batch timings and revise the remaining
estimate after the first few batches. This process limit **does not terminate or
stop a Pod and is not a cloud spending cap**. Total time across commands can exceed
two hours. Track billed time from allocation, including setup, downloads, judging
and idle periods. Stop the session before the approved spend is exhausted.

Judge and report stages do not load the target model or require CUDA. They can run
on approved remote CPU resources. Keeping a GPU
allocated during sequential API judging still bills. The judge budget in the JSON
is an estimate based on configured token prices, with usage saved per request.
Provider charges after network failures may be missing from that local ledger.

## What the default measures

- All 20 public extraction questions × five public instruction pairs × one
  stochastic answer per condition: 200 answers, 100 matched positive/negative pairs.
  This is ten times fewer answers than the full extraction procedure.
- Original system-message wording, trait and coherence rubrics, and logprob-based
  numeric score aggregation. Keep a pair only when positive trait ≥50, negative
  trait <50, and both coherence scores ≥50. Missing scores do not pass. Fewer than
  20 accepted pairs aborts extraction; increase sampling rather than weakening the
  rule after observing failure.
- Layer 16 means the output of zero-indexed block 15. Average the response-token
  residual activations, then subtract negative from positive means and normalize
  the resulting vector. BF16 averaging matches the source's order of operations.
- For each saved candidate answer, compute its projection onto that direction.
  Generate one greedy base-model answer per question and subtract its projection
  to obtain ΔP. No EOS token is appended when scoring supplied answer text, matching
  the upstream projection implementation.
- Score the existing 64 question pairs (128 answers). Compare raw projection and
  ΔP with the saved caveat/TF-IDF scores and an LLM text judgement using the same
  original trait rubric, blinded to labels and activation scores. This text judge
  sees only question and answer, not the extraction system instruction.
- The 512-token generation cap, Transformers sampling instead of vLLM, and reduced
  sampling are deviations. Cap-hit counts, missing judge scores, within-condition
  correlations and question-pair bootstrap intervals are reported. Existing text
  baseline scores are read, never refitted or overwritten.

Scoring/training examples over 2,048 tokens cause an error instead of silent
truncation or selection bias. Generation stops at 512 new tokens; cap hits are
recorded, not silently excluded. If truncation is substantial, revise the protocol
before interpreting the result. Intervals do not capture vector-extraction or
training-seed variability.

## Optional SFT pilot — a separately approved stage

Use `remote/sft-pilot.json` and a separate output directory. This is **not** the
paper's 200,000-candidate real-world subset experiment. It samples 512 paired
questions from the public synthetic data, excluding the frozen pilot, the TF-IDF
fit questions and public extraction/evaluation questions. The frozen classifier
was not serialized; this larger pool uses the LLM text baseline and answer length.

For convenience in the approved remote shell:

```bash
pv_sft() {
  python remote/run.py "$@" --config remote/sft-pilot.json --out runs/sft-pilot
}
pv_sft init
pv_sft judge --target check
pv_sft preflight
pv_sft generate
pv_sft judge --target extraction
pv_sft vector
pv_sft score
pv_sft judge --target sample
pv_sft report
pv_sft subsets
pv_sft evaluate --arm base
pv_sft judge --target evaluation --arm base
```

Review the subset manifest and the now-measured runtime before training. It selects
256 high-ΔP, 256 low-ΔP and 256 random examples, at most one answer per question per
arm. High and low have disjoint examples but may share question strings, making a
paired content contrast possible. Random can overlap either. Text scores, lengths
and source-label composition are reported; they are **not matched**. The arbitrary
256-example size is a compute pilot, not a power calculation.

Only after the optional training stage is approved:

```bash
pv_sft train --arm high
pv_sft evaluate --arm high
pv_sft judge --target evaluation --arm high
pv_sft train --arm low
pv_sft evaluate --arm low
pv_sft judge --target evaluation --arm low
pv_sft train --arm random
pv_sft evaluate --arm random
pv_sft judge --target evaluation --arm random
pv_sft report
```

Training uses BF16 rsLoRA, rank 32/alpha 64, all seven upstream projection modules,
response-only loss, AdamW 8-bit, learning rate 1e-5, five warmup steps, a linear
schedule, and effective batch size 16. Microbatch one plus gradient accumulation
16 and gradient checkpointing reduce memory. One epoch means 16 optimizer steps
per 256-example arm. Standard PEFT/Trainer replaces Unsloth/TRL; record that
implementation difference. All chosen examples train; there is no implicit 10%
split. Original public held-out behavioural prompts remain separate.

Evaluation generates three responses for each of 20 held-out questions (60 per
model), then scores hallucination and coherence. The report includes paired
question-level high-minus-low and post-minus-base differences. This is one seed;
it cannot show training-seed robustness or incremental predictive value beyond
text. A null result with this small training dose is not a failure of the paper.

## Resuming, evidence and shutdown

- Rerun the same stage/config/output directory. Completed JSONL rows are skipped.
  An interrupted generation batch is regenerated with its original batch seed.
  CUDA sampling is not guaranteed bitwise deterministic across environments.
- The config, code and input fingerprint cannot change inside a run. Use a new
  output directory for changes. Malformed JSONL fails loudly; preserve the file
  before repairing a partial final line. Do not append outputs from different runs.
- Vector extraction caches only response means. Training saves a checkpoint every
  ten optimizer steps and resumes the last checkpoint. An interruption before the
  first checkpoint repeats that arm from its fixed seed. Completed arms are skipped.
- `runs/` contains manifests, raw generations/judgements, API usage, timings, vector,
  per-example scores, `summary.json`, `comparison.png/.pdf`, and `RESEARCH_NOTE.md`.
  Optional adapters/checkpoints and subsets live there too. Outputs are ignored by
  Git until reviewed. The historical `results/` evidence is untouched.
- Back up the run directory and installed environment record to persistent storage
  and download the small evidence artifacts. Verify the transfer before releasing
  ephemeral storage. Save adapters if they will be needed again.
- Explicitly stop/terminate the GPU through the approved control-plane action and
  verify its state and remaining storage charges. A network volume can remain after
  Pod termination; a Pod volume cannot. Choose accordingly before provisioning.
  Never rely on `--stop-after` / `--terminate-after` creation flags as cost guards.

## Static review versus remote acceptance

Local checks are limited to AST parsing, JSON parsing/schema assertions, shell
syntax and diff review. No script is run as a test, and no ML package is imported.
Remote acceptance still requires: dependency installation, original judge access,
model download/load, numerical parity, viable extraction filters, generation
throughput, and (if approved) one completed training/evaluation arm. The runbook
prepares those checks; it does not claim they have passed.

See [BUDGET.md](BUDGET.md) for workload-based cost estimates.
