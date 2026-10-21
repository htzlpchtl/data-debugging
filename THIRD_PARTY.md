# Third party attribution

This project uses the public code, trait artifacts, and datasets from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors), pinned to commit `b8e0f044fe2410a6fad579f38324f03f13b4e917`.

Associated paper: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey, *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (2025), [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).

The upstream repository supplies the Apache License 2.0. An unmodified copy is included at `third_party/persona_vectors/LICENSE`. No separate NOTICE file was found in the checked-out release.

`results/sample.jsonl` contains a selected subset of the upstream `dataset/hallucination/normal.jsonl` and `misaligned_2.jsonl` files, with added IDs, labels, and text-baseline scores. Source question and answer text is preserved. The full archive is not included in this repository.

The disabled `scripts/run_local.py` is an adaptation of the response-mean activation and projection method in upstream `generate_vec.py` and `eval/cal_projection.py`. It substitutes MLX, adds chunked processing and resumable outputs, and proposes a smaller extraction procedure. It is unvalidated and does not constitute a successful reproduction.

The downloaded model is [mlx-community/Qwen2.5-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit), revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`, a quantization of Qwen2.5-7B-Instruct. Model weights are excluded from this repository.

The new, unexecuted `remote/run.py` adapts the response-mean/projection logic,
system-message construction, paired filtering and logprob-score aggregation from
upstream `generate_vec.py`, `eval/cal_projection.py`, `eval/eval_persona.py`, and
`judge.py`. It reads the original trait artifacts and coherence rubric at runtime.
Changes include a single-layer forward hook, inference-mode execution, explicit
CUDA gating, immutable run manifests, cached/resumable stages and a reduced pilot.
`remote/train.py` uses the upstream LoRA hyperparameters with standard PEFT/Trainer
instead of Unsloth/TRL and an explicitly smaller dataset. The upstream Apache 2.0
licence applies to adapted material; the copy above is retained. The intended
remote model is the original BF16 `Qwen/Qwen2.5-7B-Instruct`, with its Hub commit
resolved and frozen before loading. No remote model has been downloaded or run.
