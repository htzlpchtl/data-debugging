# Project instructions

- Experimental execution is remote-only. Do not run local inference, training, activation extraction, model-loading checks, benchmarks, or baseline experiments. Do not import MLX to check availability.
- Static file inspection, documentation edits, small syntax/JSON checks, and Git operations are permitted. Do not run experiment scripts as tests.
- Experimental computation, resource provisioning, and spending require explicit approval before execution.
- Preserve completed evidence and keep proposed protocols separate from executed results. The remote scoring pilot is in `results/remote_scoring_2026-10-21/`. The completed 7B baseline, paired Normal/Hallucination II SFT runs and held-out behavioural measurements are in `results/controlled_sft_2026-10-28/`. The subsequent text-scoring and matching-feasibility audit is in `results/matching_feasibility_2026-10-28/`; its candidates retain textual confounds and have not been used in a new SFT run. None of these experiments establishes incremental predictive value beyond text.
- Keep the work focused on one reproducible experiment and the comparison against text-only prediction. Avoid building a general framework.
- The completed remote CPU quantile sensitivity audit is in `results/quantile_audit_2026-10-28/`. It reuses saved raw scores and adds random-subset diagnostics; its candidates remain text-confounded. No natural-answer references or new SFT were run for it. Its temporary CPU Pod was deleted after checksum-verified backup.
- Commits and pushes require explicit instruction.
- The balanced SFT pilot is prepared but unexecuted and paused at the user's request. Its handover is `remote/BALANCED_SFT.md`. Resume experimental work only after a new instruction to resume; no automatic restart is scheduled.
