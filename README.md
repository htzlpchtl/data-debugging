# Data debugging

Research on predicting the behavioural effects of supervised fine-tuning (SFT) data before training, and testing whether model activations add predictive information beyond the semantic content of the text.

## Current checkpoint

**Text baselines and a two-call remote OpenRouter judge compatibility check have completed.** No persona vector, activation ranking, fine-tuned model, or downstream behavioural result has been produced. The activation implementation has not yet undergone its numerical check. See the [2026-10-21 checkpoint](remote/CHECKPOINT_2026-10-21.md) for the verified state and proposed next experiment.

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

Experimental execution is **remote-only**. Local inference, fine-tuning, activation extraction, benchmarks, and model-loading checks are prohibited. The historical `scripts/run_local.py` entry point is disabled before MLX imports and is retained as an unvalidated implementation draft.

Local work is limited to documentation, static checks, source inspection, and Git operations. Experimental runs and resource provisioning require explicit approval under the project instructions in [AGENTS.md](AGENTS.md).

## Files and provenance

| File | What it contains |
| --- | --- |
| `results/baseline_results.json` | Completed baseline metrics, evaluation scope, and limitations. |
| `results/sample.jsonl` | The 128 public answers, labels, and saved baseline scores. |
| `results/protocol.json` | Source fingerprint, sample/train IDs, and the partly executed pilot plan. The activation settings describe intentions, not completed work. |
| `results/text_baseline_terms.json` | Largest positive and negative classifier weights. |
| `results/model_manifest.json` | Identity and revision of the downloaded quantized model; its location is project-relative. |
| `scripts/prepare.py` | Sampling and text-baseline preparation used for the completed checkpoint. Execution is remote-only. |
| `scripts/download_model.py` | Historical model download utility, pinned to the downloaded revision. |
| `scripts/run_local.py` | Disabled, unvalidated MLX draft; not a remote GPU implementation. |
| `scripts/analyze.py` | Unexecuted analysis draft requiring activation outputs that do not exist. |
| `requirements-lock.txt` | Historical macOS dependency snapshot, not a validated Linux/CUDA environment. |

Upstream commit: `b8e0f044fe2410a6fad579f38324f03f13b4e917`.

SHA-256 of upstream `dataset.zip`: `6913afdd712997599016444e789d2a4a5e383b6418e6596a4598cebdb97e943e`.

The upstream checkout, downloaded model, virtual environment, and machine-specific logs are excluded from Git. A future reproduction must restore the upstream checkout at the recorded commit and place it at `upstream/persona_vectors`; the scripts read its dataset archive and trait prompts. The fitted classifier itself was not serialized. Reproducing the fit requires the recorded data, code, and dependencies on approved compute.

See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and the upstream licence.

## Next experiment

Remote execution instructions are provided in [remote/README.md](remote/README.md),
with [workload and cost estimates](remote/BUDGET.md). The remote environment is
installed and the [OpenRouter judge compatibility check](remote/OPENROUTER_CHECK.md)
passed. Qwen loading and CUDA execution remain unverified.
`remote/scoring-pilot-openrouter.json` prepares the frozen 128-answer scoring
milestone using the checked judge route; `remote/scoring-pilot.json` retains the
original direct-OpenAI configuration.
`remote/sft-pilot.json` separately proposes three 256-example, one-epoch LoRA runs.
The GPU scoring and SFT stages have not been executed. Remote outputs are written
to ignored `runs/`, separate from the completed baseline evidence. Further
experimental execution awaits explicit approval.

The first remote stage compares response-activation calculations against the official implementation, followed by vector extraction and scoring of the frozen sample. Deviations from the published method are recorded in the configuration. The historical MLX draft proposed quantization, fewer extraction responses, a self-judge, and shorter generations; those settings were not validated.

After that check, compare training subsets matched on text-judge score but differing in activation score, and measure behaviour after SFT on held-out questions. Include a random subset, control training amount and answer length, and repeat training seeds if feasible. The outcome of interest is incremental prediction of behavioural change, not merely separation of synthetic dataset labels.
