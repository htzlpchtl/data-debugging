# Project instructions

- Experimental execution is remote-only. Do not run local inference, training, activation extraction, model-loading checks, benchmarks, or baseline experiments. Do not import MLX to check availability.
- Static file inspection, documentation edits, small syntax/JSON checks, and Git operations are permitted. Do not run experiment scripts as tests.
- Experimental computation, resource provisioning, and spending require explicit approval before execution.
- Preserve completed evidence and keep proposed protocols separate from executed results. The remote scoring pilot is in `results/remote_scoring_2026-10-21/`. The completed 7B baseline, paired Normal/Hallucination II SFT runs and held-out behavioural measurements are in `results/controlled_sft_2026-10-28/`. The subsequent text-scoring and matching-feasibility audit is in `results/matching_feasibility_2026-10-28/`; its candidates retain textual confounds and have not been used in a new SFT run. None of these experiments establishes incremental predictive value beyond text.
- Keep the work focused on one reproducible experiment and the comparison against text-only prediction. Avoid building a general framework.
- Commits and pushes require explicit instruction.
