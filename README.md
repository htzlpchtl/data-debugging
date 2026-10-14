# Data debugging

A short BlueDot technical AI safety project asking whether we can predict the behavioural effects of supervised fine-tuning (SFT) data before training, and whether model activations add useful predictive information beyond inspecting the text.

## Current checkpoint

**Only the text baselines have completed.** No persona vector, activation ranking, fine-tuned model, or downstream behavioural result has been produced. The activation implementation has not passed its numerical check.

| Completed measurement | AUC for the supplied dataset label |
| --- | ---: |
| Fixed uncertainty/caveat phrase count | 0.875 |
| TF-IDF word and bigram classifier | 1.000 |

These results use 64 held-out question pairs (128 answers) from the public Hallucination Normal and Hallucination II datasets. The classifier was trained on 1,024 separate question pairs. Sampling used seed `20261014`, deduplicated exact question strings, and kept paired answers together. The released files contain 4,992 rows per condition and 4,990 unique questions.

AUC measures ranking of the **dataset labels**, not factual accuracy or post-training behaviour. The supervised classifier receives labelled examples; it is not a zero-shot judge. Its perfect ranking on this small sample suggests that wording alone makes this particular comparison easy. It does not establish whether activations add predictive value for less obvious training effects. No uncertainty intervals have been computed for these completed baseline results.

## Paper and reproduction target

The starting point is Chen et al., [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/html/2507.21509v1), with [official code and datasets](https://github.com/safety-research/persona_vectors).

The initial target is Figure 9's separation of normal and hallucination-inducing examples. The intended model is `Qwen/Qwen2.5-7B-Instruct`, using the output of transformer layer 16 for hallucination. Contrastive prompts elicit or suppress the trait; the difference between mean response activations gives the vector. Each answer is projected onto its unit-normalized direction. The pre-training predictor subtracts the projection of the model's own answer to the same question (Section 6.1).

The stronger validation in Section 6.3 selects high, low, and random projection-difference subsets and compares behaviour after separate fine-tunes. Appendix K also compares activation filtering with LLM text filtering; LLM filtering performs best for hallucination in that experiment. These are published findings, not results reproduced here.

## Execution constraint

**Do not run local experiments, model inference, fine-tuning, benchmarks, or model-loading checks.** The local 4-bit 7B attempt on an 8 GB Mac caused a crash, and the user explicitly prohibited further local experiments. The historical `scripts/run_local.py` entry point is disabled before MLX imports. It is retained for inspection, not execution.

Documentation, static source inspection, and Git operations are allowed. Further experimental computation requires the user's explicit approval and remote compute. No paid compute has been used.

## Files and provenance

| File | What it contains |
| --- | --- |
| `results/baseline_results.json` | Previously observed baseline metrics and their scope; recorded without rerunning the experiment. |
| `results/sample.jsonl` | The 128 public answers, labels, and saved baseline scores. |
| `results/protocol.json` | Source fingerprint, sample/train IDs, and the partly executed pilot plan. The activation settings describe intentions, not completed work. |
| `results/text_baseline_terms.json` | Largest positive and negative classifier weights. |
| `results/model_manifest.json` | Identity and revision of the downloaded quantized model; its location is project-relative. |
| `scripts/prepare.py` | Successfully executed sampling and text-baseline preparation. Do not rerun locally under the current constraint. |
| `scripts/download_model.py` | Historical model download utility, pinned to the downloaded revision. |
| `scripts/run_local.py` | Disabled, unvalidated MLX draft; not a remote GPU implementation. |
| `scripts/analyze.py` | Unexecuted analysis draft requiring activation outputs that do not exist. |
| `requirements-lock.txt` | Historical Mac environment snapshot, not a validated Linux/CUDA environment. |

Upstream commit: `b8e0f044fe2410a6fad579f38324f03f13b4e917`.

SHA-256 of upstream `dataset.zip`: `6913afdd712997599016444e789d2a4a5e383b6418e6596a4598cebdb97e943e`.

The upstream checkout, downloaded model, virtual environment, and machine-specific logs are excluded from Git. A future reproduction must restore the upstream checkout at the recorded commit and place it at `upstream/persona_vectors`; the scripts read its dataset archive and trait prompts. The fitted classifier itself was not serialized. Reproducing the fit requires the recorded data, code, and dependencies on approved compute.

See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and the upstream licence.

## Next experiment

On approved remote compute, first verify a small response-activation calculation against the official implementation, then extract the hallucination vector and score the frozen sample. Record every difference from the original method. The abandoned local plan proposed quantization, fewer extraction responses, a self-judge, and shorter generations; none of those choices is a validated remote protocol.

After that check, compare training subsets matched on text-judge score but differing in activation score, and measure behaviour after SFT on held-out questions. Include a random subset, control training amount and answer length, and repeat training seeds if feasible. The outcome of interest is incremental prediction of behavioural change, not merely separation of synthetic dataset labels.
