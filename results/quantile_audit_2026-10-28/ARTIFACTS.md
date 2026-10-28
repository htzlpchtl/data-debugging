# Quantile audit artifacts

- `manifest.json` records the script/config/input hashes, environment versions,
  remote execution timestamp and scope. `executed_source/` contains the exact
  executed Python script and configuration. `completion.json` records completion
  and elapsed analysis time.
- `scored_pool.jsonl` preserves all original QA pairs and saved projections/text
  scores, augmented with the existing fixed caveat counts. The original source
  inputs remain in the preceding result directories and are hash-identified.
- `pool_summaries.json` and `quantile_results.json` contain every measured pool,
  arm distribution, comparison and boundary-tie count. `quantile_table.md` is a
  compact view of the filtered activation-ranking analysis.
- `selections/` retains candidate IDs for all 32 cases. They are exploratory,
  not approved or finalized training subsets. Random controls may overlap the
  selected tails and are drawn from the entire pool.
- `random_draw_summaries.jsonl` retains all 16,000 random-draw mean summaries.
  These are sampling diagnostics, not SFT runs. Exact draws can be reconstructed
  from the frozen inputs, code, seeds and NumPy version.
- `quantile_balance.png` and `.pdf` show the four activation-ranking fractions in
  the text-score-below-1 pool, including the random-subset range and text-feature
  imbalances. `verification.json` records implementation checks.
- `backup_manifest.json` covers 46 raw files (6,541,140 bytes) in the Git-ignored
  `runs/quantile-audit-2026-10-28/`. All sizes and SHA-256 hashes matched after
  transfer. Two installation/execution logs are retained in that raw backup but
  omitted from this public directory. The original input and local source hashes
  also matched the remote manifest.
- `shutdown_and_backup.json` is a separate local record written after backup and
  verified deletion. It is outside the earlier backup manifest, as are these
  explanatory Markdown notes.

All analysis and plotting ran on remote CPU compute. Local work consisted of
source authoring, syntax checks, file inspection, transfer, checksum verification
and image viewing. No model weights or credentials were transferred to this Pod.
