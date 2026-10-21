# OpenRouter judge compatibility — 2026-10-21

**Result:** the existing logprob-based judge can be called through OpenRouter when
catalog parameter filtering is disabled. This verifies API compatibility on one
frozen candidate; it is not an activation or fine-tuning result.

The check used the original hallucination and coherence rubrics on candidate
`1062_normal`, from the preserved 128-answer sample. The request selected
`openai/gpt-4.1-mini`, pinned the `openai` provider, disabled provider fallbacks and
automatic retries, and requested temperature 0, seed 0, one generated token and
20 top-token log probabilities. No Qwen weights were downloaded or loaded by
these stages. Initialization resolved the Qwen revision to
`a09a35458c702b33eeacc393d103063234e8bc28`.

## Routing discrepancy

1. With `require_parameters: true`, OpenRouter returned HTTP 404: no endpoint could
   handle the requested parameters. Its error identified `Filter by Parameters`
   as the failed routing step. Reported API-key usage remained zero.
2. A separate diagnostic configuration changed only catalog filtering to `false`
   and lowered the estimated API budget to $0.005. Both calls returned 20 top-token
   probabilities and sufficient numeric probability mass for the original scoring
   formula. The returned provider was `OpenAI`; the returned model was the alias
   `openai/gpt-4.1-mini`.

| Rubric | Score (0–100) | Top logprobs returned | Input tokens | Billed output tokens | API cost (USD) |
|---|---:|---:|---:|---:|---:|
| Hallucination | 0.0000000002415 | 20 | 270 | 2 | 0.0001112 |
| Coherence | 99.9838696 | 20 | 350 | 2 | 0.0001432 |
| Total | — | — | 620 | 4 | **0.0002544** |

Each response contained one generated-token logprob position, although the usage
records billed two output tokens per call. The cost uses reported usage, not the
requested generation cap. Token-rate estimates, response-reported costs and the
API-key usage delta agreed. Runpod allocation and storage are separate costs.

## Evidence and limits

Raw evidence is preserved in ignored `runs/openrouter-check/` and
`runs/openrouter-diagnostic/`, with the diagnostic configuration in
`runs/openrouter-diagnostic-config.json`. The diagnostic directory also contains
the executed runner patch against repository commit `89f4cc5`. Manifests record
source, dataset and configuration hashes. No credentials belong in these artifacts.

This check does not establish equivalence to direct OpenAI responses or verify the
exact serving snapshot: the catalog endpoint names the 2025-04-14 snapshot, while
the completion returns an alias. It also does not establish judge accuracy,
extraction-filter yield, GPU compatibility, or predictive value beyond text.

Next, once separately approved: run the GPU preflight and scoring pilot with
`remote/scoring-pilot-openrouter.json`, retaining the response checks and recording
returned provider/model metadata. No SFT is part of that configuration.

Sources: [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini/endpoints),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
and the saved remote API responses and logs.
