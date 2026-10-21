# Checkpoint — 2026-10-21

**Status:** text baselines and a two-call remote judge compatibility check are
complete. The remote environment is installed. GPU preflight, persona-vector
extraction, activation scoring, fine-tuning and downstream evaluation remain
unexecuted. No further experimental run is authorized at this checkpoint.

## Completed evidence

| Check | Result | What it establishes |
| --- | --- | --- |
| Caveat phrase baseline | AUC 0.875 on 128 answers | Ranking of the supplied dataset labels |
| TF-IDF baseline | AUC 1.000 on the same answers | Text alone separates this small sample; classifier fit used 1,024 separate question pairs |
| OpenRouter judge check | Both rubrics returned 20 top-token log probabilities; total API cost $0.0002544 | The original probability-based scoring calculation can run through this route |

The baseline metrics are preserved in `results/baseline_results.json`, with
per-answer scores in `results/sample.jsonl`. They do not measure factual accuracy
or predict post-SFT behaviour.

The judge check used one existing answer. Strict catalog parameter filtering
returned HTTP 404; a separate diagnostic with that filter disabled succeeded.
Response validation remained enabled. Details, numerical scores and limitations
are in [OPENROUTER_CHECK.md](OPENROUTER_CHECK.md). Raw evidence is preserved locally
in ignored `runs/openrouter-check/` and `runs/openrouter-diagnostic/`; the executed
runner patch and manifests record provenance. Credentials and account metadata
are excluded from public documentation.

## Prepared, not executed

The proposed configuration is
[`scoring-pilot-openrouter.json`](scoring-pilot-openrouter.json):

- Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`,
  BF16 on a remote GPU, using layer 16.
- 200 contrastively prompted answers to construct a hallucination direction,
  retaining only pairs that pass the original trait and coherence rules.
- The existing 64 question pairs / 128 candidate answers for activation scoring,
  with the model's own answer as the projection-difference reference.
- Comparison against the preserved text baselines and an LLM judge that sees
  only the question and answer.
- No SFT in this configuration. Its $1 judge ledger limit covers estimated API
  use only; it is not a combined cloud spending cap or execution authorization.

The configuration and updated runner have been copied to the remote checkout.
The OpenRouter route has been exercised; Qwen loading and CUDA execution have
not. Source changes remain uncommitted at this checkpoint, based on `89f4cc5`.

## Next experiment, after approval

1. Load Qwen on the remote GPU and check that response activations agree with the
   pinned public implementation. Stop if this numerical check fails.
2. Elicit contrasting answers, have the judge filter them, and extract the
   internal direction associated with hallucination.
3. Score the same 128 existing answers with that direction and compare the
   ranking with text-only scores. Save one figure/table and a short research note,
   including runtime, costs and deviations from the paper.

This first pilot tests whether the scoring pipeline reproduces the expected
separation and how closely its ranking follows the text. Establishing whether
activations predict later behavioural change better than text requires a later,
separately approved fine-tuning comparison.

All experimental computation remains remote-only. No compute or resource
lifecycle action was performed to create this documentation checkpoint.
