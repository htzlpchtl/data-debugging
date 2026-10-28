# Executed controlled SFT pilot

The 28 October 2026 run completed a held-out baseline, two SFT arms, identical held-out evaluations, training-data activation projections and the existing lexical baseline. Results and limitations are in the [research note](../results/controlled_sft_2026-10-28/RESEARCH_NOTE.md). This is a reduced reproduction using the official data, prompts, rubrics, activation reference and training hyperparameters, with a standard Transformers/PEFT trainer.

`controlled_sft.py` reuses `run.py` and `train.py`. `controlled-sft-2026-10-28.json` is the frozen configuration. The launcher `controlled-sft-2026-10-28.sh` records the exact historical stage sequence and deadline; **its dated cutoff must not be reused for a future run**. It does not stop cloud billing. New runs require approval, a new output directory and a reviewed configuration/cutoff.

The executed stages were:

1. Freeze 1,024 disjoint paired training questions from the pinned public archive, excluding earlier pilot/classifier-fit questions and all extraction/evaluation questions.
2. Check the OpenRouter judge and compare layer-16 response means against the official function; both checked answers had maximum absolute error 0.
3. Generate and hash-lock 100 base-model answers; score the original hallucination and coherence rubrics.
4. Train Normal and Hallucination II adapters independently from the same pinned base model, with identical training seed, question order, example count and hyperparameters. Generate and score the same 100 held-out jobs for each.
5. Produce the behavioural report, then score all 2,048 training answers with the unchanged base model and the saved 21 October vector. Generate no additional natural-reference answers: their contribution cancels in the paired between-arm mean contrast.
6. Reuse the pre-existing caveat-count rule for a descriptive text comparison, save all evidence, verify the 1.70 GB local backup against every remote checksum, then stop the Pod.

Execution used one A40, BF16 Qwen2.5-7B-Instruct, and the pinned CUDA environment from `setup.sh`. Both training runs completed 64 steps, one epoch, with peak PyTorch allocated memory at most 16.10 GiB. The code retains a conservative 40 GiB GPU guard; these measurements do not validate a smaller GPU configuration. Each full training stage took about five minutes, while the sequential pipeline including evaluation, API judging and projection took 36 minutes. Setup/downloads, connection handling and backup extended total allocation to about 74 minutes. Estimated compute plus reported API usage was $0.83, excluding disk costs and the previous scoring pilot.

The baseline and both adapters inherited Qwen's top-k 20 and repetition penalty 1.05 during generation, alongside explicit temperature 1 and top-p 1. The official unsteered evaluation uses different sampling defaults. This discrepancy was documented after a static audit; settings were not changed between conditions. Training token counts also differ because answers differ. A strong text judge on the new training pool, text-matched subset experiments, additional seeds and independent factual/capability evaluations remain unexecuted.

Full raw evidence and optimizer checkpoints are retained under ignored `runs/controlled-sft-2026-10-28/`; the [artifact map](../results/controlled_sft_2026-10-28/ARTIFACTS.md) lists the public subset. Executed source snapshots and checksums preserve the pre-run code despite its uncommitted working-tree state. No credentials are included.
