# Balanced SFT pilot: prepared, not executed

Status on 2026-10-28: paused before remote execution. The next session requires
an explicit instruction to resume. No scheduled restart has been configured.

## Completed preparation

- `balanced_sft.py`, `balanced-sft-2026-10-28.json` and
  `balanced-sft-2026-10-28.sh` are preserved as source and configuration.
- Python AST, JSON parsing and shell syntax checks passed. These are static
  checks; the new experimental pipeline has not been run.
- Previously completed measurements remain in their existing result directories,
  including the completed quantile audit. They must not be described as results
  of this proposed pilot.

## Proposed experiment

Use the 995 existing candidate answers with saved text-judge scores below 1.
After checking official activation parity and four cached projections, generate
natural answers with the unchanged base model and compute reference-subtracted
persona-vector projections. Preserve all generations and scores.

Select 128 examples per arm: low projection, high projection and random.
Use identical exact response-token-count and caveat-count strata across arms.
Freeze subset identities and predicted behavioural ordering before training.
Inspect the matching diagnostics and fixed textual audit samples first.

Train each arm with the same LoRA settings for eight epochs, giving 64 optimizer
steps at effective batch size 16. Reuse the saved baseline and evaluate all new
adapters on the same 20 held-out questions with five responses each. Record
hallucination and coherence judgements, answer lengths and token-cap hits.

The primary prediction is that the high arm has a higher mean hallucination
score than the random arm. Low versus random and high versus low are secondary
comparisons. Save a behavioural results table and judge-versus-projection plots.

The subset size is a resource-based pilot choice, not a power calculation.
This is one training seed. Text scores are filtered, not exactly matched;
topic, answerability and other semantic differences can remain. A result would
not establish superiority over unrestricted text-based prediction.

## Execution checkpoint

The existing GPU Pod could not restart because its host had no free GPU. Its
last confirmed state was stopped. No replacement Pod or network volume was
created, and no new reference generation, training or judging was launched.
Runpod control tools subsequently became unavailable in the chat, so there was
no fresh shutdown verification at pause time.

A subsequent local backup check verified all saved file sizes and SHA-256 hashes:
120 controlled-SFT files (1,695,006,646 bytes, including both trained adapters),
56 matching-audit files (14,981,816 bytes), and 46 quantile-audit files
(6,541,140 bytes). No mismatches were found. This verifies saved experimental
evidence, not a complete copy of the Pod's filesystem. The private judge
credential and recreatable model cache/environment were not included in these
backups. A fresh account-wide shutdown check remains blocked by unavailable
Runpod control access and an unauthenticated browser session; no new shutdown or
termination is claimed.

On resumption:

1. Restore Runpod control access and read current resource state and availability.
2. Recover the existing workspace or provision suitable remote compute. The last
   observed A40 rate was $0.59/hour; refresh this before provisioning.
3. Restore access to the private judge credential without including it in logs,
   the repository or conversation outputs.
4. Transfer the frozen source and saved inputs. Set a fixed deadline allowing
   backup within the planned two-hour compute limit; retain the $1 judge cap.
5. Run numerical checks, reference scoring and selection. Inspect results before
   continuing to SFT. Keep any failed checks and errors.
6. Back up raw outputs, adapters, configurations and checksums; verify the backup
   before stopping compute. Report completed measurements separately from work
   that did not finish.

The current launcher requires the judge credential and stops after selection
unless training is explicitly enabled. Prepare the continuation command before
execution rather than rerunning all earlier stages unnecessarily. Commit and
push only on explicit instruction.
