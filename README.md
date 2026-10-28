# Data debugging

Research on predicting the behavioural effects of supervised fine-tuning (SFT) data before training, and testing whether model activations add predictive information beyond the semantic content of the text.

## Current checkpoint

**A controlled 7B SFT pilot is complete.** Two fine-tunes used Normal versus Hallucination II answers to the same 1,024 public questions. On 20 separate questions with five answers per model, mean judged hallucination was **16.56 for the base model, 1.50 after Normal SFT, and 98.73 after Hallucination II SFT**. The paired difference was +97.24 points (95% question-bootstrap interval 93.08–99.61). This is one training seed with explicitly documented reproduction deviations. See the [research note and figure](results/controlled_sft_2026-10-28/RESEARCH_NOTE.md), [raw evidence map](results/controlled_sft_2026-10-28/ARTIFACTS.md), and [remote execution record](remote/CONTROLLED_SFT.md).

Mean base-model activation projections ordered the training datasets consistently with behaviour (−6.539 versus +6.456), as did the existing simple text baseline. Numerical activation checks matched the official function exactly on two checked answers. **Incremental prediction beyond text remains untested.** All raw artifacts and both adapters are backed up locally; the Pod is stopped.

A subsequent **matching-feasibility audit** scored all 2,048 training answers with the LLM text judge. Exact-length matching found 377 pairs with similar text scores but different raw activation projections. However, caveat wording still differs between the resulting groups, and 90.6% of text scores lie near the floor or ceiling. The converse comparison yielded only 16 pairs across source labels. These are exploratory candidates, not newly trained conditions. See the [feasibility note and scatter plot](results/matching_feasibility_2026-10-28/RESEARCH_NOTE.md) and [saved evidence](results/matching_feasibility_2026-10-28/ARTIFACTS.md). All outputs are backed up; the Pod is stopped.

The subsequent [quantile sensitivity audit](results/quantile_audit_2026-10-28/RESEARCH_NOTE.md)
replaces those arbitrary absolute cutoffs with top, bottom and random subsets at
5%, 10%, 20% and 30%. Among 995 answers with text score below 1, all four sizes
retain substantial length and caveat differences. Both activation-ranked and
text-ranked selections are saved; no final training subset or new SFT result is
claimed. The temporary CPU Pod was deleted after verified backup.

## Earlier scoring pilot

On 21 October, a hallucination vector was extracted from 77 accepted contrastive pairs and used to score 128 public answers. See the unchanged [scoring research note](results/remote_scoring_2026-10-21/RESEARCH_NOTE.md). The earlier [checkpoint](remote/CHECKPOINT_2026-10-21.md) records the state before that run.

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
| `results/quantile_audit_2026-10-28/` | Completed saved-score quantile sensitivity analysis, random-subset comparisons, text-feature balance, candidate IDs, figure and provenance. |
| `results/matching_feasibility_2026-10-28/` | Text judgements for 2,048 answers, saved raw projections, candidate matches, confound audit, scatter plot and provenance. No new SFT. |
| `results/controlled_sft_2026-10-28/` | Completed baseline and matched SFT behavioural measurements, raw evaluations/judgements, training projections, numerical checks, figure and provenance. |
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

The paired Normal/Hallucination II SFT contrast, matching-feasibility audit and
quantile sensitivity analysis are complete. The [quantile protocol](remote/QUANTILE_AUDIT.md)
documents the role of random selection and the distinction from the paper's method.
Next, obtain reference-subtracted projections and compare their rankings with the
saved raw scores. Before selecting another SFT comparison, assess length/caveat
balance and question topic/answerability. If enough comparable examples remain,
freeze high, low and equally sized random subsets, then evaluate their SFT effects
on held-out questions, with coherence checks and repeated training seeds if
feasible. These are proposed steps, not completed measurements.

The [balanced SFT pilot](remote/BALANCED_SFT.md) now has prepared source and a
configuration for this comparison. It is paused before execution: no new natural
references, selected training arms or post-SFT measurements have been produced.

`remote/sft-pilot.json` proposes three small LoRA runs, but currently selects
**unmatched** subsets and needs revision for this comparison. Further experiments
require explicit approval; the outcome of interest is incremental prediction of
behavioural change beyond text.
