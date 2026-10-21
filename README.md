# Data debugging

Research on predicting the behavioural effects of supervised fine-tuning (SFT) data before training, and testing whether model activations add predictive information beyond the semantic content of the text.

## Current checkpoint

**The remote scoring pilot is complete.** A hallucination vector was extracted from 77 accepted contrastive pairs and used to score 128 public answers. The implementation matched the official response-activation function exactly on two checked answers. No SFT or downstream behavioural outcome has been measured. See the [research note and figure](results/remote_scoring_2026-10-21/RESEARCH_NOTE.md) for results, limitations and the next experiment. The earlier [checkpoint](remote/CHECKPOINT_2026-10-21.md) records the state before this run.

| Completed measurement | AUC for the supplied dataset label |
| --- | ---: |
| Raw activation projection | 0.9998 |
| Projection difference from the model's own answer | 0.9182 |
| GPT-4.1 mini text judgement | 1.000 |
| Fixed uncertainty/caveat phrase count | 0.875 |
| TF-IDF word and bigram classifier | 1.000 |

These results use 64 held-out question pairs (128 answers) from the public Hallucination Normal and Hallucination II datasets. The classifier was trained on 1,024 separate question pairs. Sampling used seed `20261014`, deduplicated exact question strings, and kept paired answers together. The released files contain 4,992 rows per condition and 4,990 unique questions.

AUC measures ranking of the **dataset labels**, not factual accuracy or post-training behaviour. The supervised classifier receives labelled examples; it is not a zero-shot judge. Both text methods perfectly separate this small sample. This experiment therefore provides no evidence that activations add predictive value beyond text. Question-pair bootstrap intervals and within-condition ranking correlations are in the research note; the original baseline files remain unchanged.

## Paper and reproduction target

The starting point is Chen et al., [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/html/2507.21509v1), with [official code and datasets](https://github.com/safety-research/persona_vectors).

The initial target is Figure 9's separation of normal and hallucination-inducing examples. The completed pilot used `Qwen/Qwen2.5-7B-Instruct`, BF16, and the output of transformer layer 16. Contrastive prompts elicit or suppress the trait; the difference between mean response activations gives the vector. Each supplied dataset answer is projected onto its unit-normalized direction. Projection difference subtracts the projection of the model's own answer to the same question (Section 6.1).

The stronger validation in Section 6.3 selects high, low, and random projection-difference subsets and compares behaviour after separate fine-tunes. Appendix K also compares activation filtering with LLM text filtering; LLM filtering performs best for hallucination in that experiment. These are published findings, not results reproduced here.

## Execution constraint

Experimental execution is **remote-only**. Local inference, fine-tuning, activation extraction, benchmarks, and model-loading checks are prohibited. The historical `scripts/run_local.py` entry point is disabled before MLX imports and is retained as an unvalidated implementation draft.

Local work is limited to documentation, static checks, source inspection, and Git operations. Experimental runs and resource provisioning require explicit approval under the project instructions in [AGENTS.md](AGENTS.md).

## Files and provenance

| File | What it contains |
| --- | --- |
| `results/remote_scoring_2026-10-21/` | Completed remote vector, scores, generations, judgements, figure, research note, provenance and numerical checks. |
| `results/baseline_results.json` | Completed baseline metrics, evaluation scope, and limitations. |
| `results/sample.jsonl` | The 128 public answers, labels, and saved baseline scores. |
| `results/protocol.json` | Source fingerprint, sample/train IDs, and the partly executed pilot plan. The activation settings describe intentions, not completed work. |
| `results/text_baseline_terms.json` | Largest positive and negative classifier weights. |
| `results/model_manifest.json` | Identity and revision of the downloaded quantized model; its location is project-relative. |
| `scripts/prepare.py` | Sampling and text-baseline preparation used for the completed checkpoint. Execution is remote-only. |
| `scripts/download_model.py` | Historical model download utility, pinned to the downloaded revision. |
| `scripts/run_local.py` | Disabled, unvalidated MLX draft; not a remote GPU implementation. |
| `scripts/analyze.py` | Unexecuted historical analysis draft; the completed remote report uses `remote/report.py`. |
| `requirements-lock.txt` | Historical macOS dependency snapshot, not a validated Linux/CUDA environment. |

Upstream commit: `b8e0f044fe2410a6fad579f38324f03f13b4e917`.

SHA-256 of upstream `dataset.zip`: `6913afdd712997599016444e789d2a4a5e383b6418e6596a4598cebdb97e943e`.

The upstream checkout, downloaded model, virtual environment, and machine-specific logs are excluded from Git. A future reproduction must restore the upstream checkout at the recorded commit and place it at `upstream/persona_vectors`; the scripts read its dataset archive and trait prompts. The fitted classifier itself was not serialized. Reproducing the fit requires the recorded data, code, and dependencies on approved compute.

See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and the upstream licence.

## Next experiment

Remote instructions are in [remote/README.md](remote/README.md), with workload
estimates in [remote/BUDGET.md](remote/BUDGET.md) and measured costs in the research
note. The completed run used `remote/scoring-pilot-openrouter-1000.json`. The earlier
512-token run was superseded before vector extraction because of truncation.
Full raw backups remain in ignored `runs/`; public evidence is in the new results
directory. Original baseline evidence has not been overwritten.

Next, freeze this vector and establish whether a larger public pool supports subsets
matched on text-judge score and answer length but separated in activation score.
Then compare their effects after SFT on held-out questions, with the base model
and a random-subset control. `remote/sft-pilot.json` proposes three small LoRA runs,
but currently selects **unmatched** subsets and needs revision for this comparison.
Repeat training seeds if feasible. Further experiments require explicit approval;
the outcome of interest is incremental prediction of behavioural change.
