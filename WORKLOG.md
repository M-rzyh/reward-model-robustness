# WORKLOG (newest first)

## 2026-10-27
- Trained seeds 3 and 4 for both sources (jobs <IDs>), evaluated them, re-ran the pool scoring and the best-of-8 analysis → 5 seeds per configuration.
- Found exact score ties under bf16 scoring (up to 82 uf_test pairs and 58 pool prompts for one model). Traced to bf16 rounding of large rewards. Re-scored every model in float32 (`scripts/precision_check.sh`, job <ID>); conclusions unchanged, float32 numbers are the ones reported, bf16 kept as reference.
- Added `src/eval/runs_table.py` and `src/eval/readme_tables.py` so every README table is generated from `results/`.
- README rewritten as the results README; pushed to github.com/M-rzyh/reward-model-robustness; tag v0.1-2026-10-27.

## 2026-10-23
- Evaluated seeds 1–2 → first 3-seed 2×2 (`results/summary_2x2.md`).
- Generated the frozen best-of-8 pool (job 1909608): 500 HH-test prompts × 8 responses from Qwen2.5-1.5B-Instruct, T=1.0, top-p 1.0, top-k off, repetition penalty off, max 512 new tokens, seed 1234. At 256 new tokens 59% of candidates were truncated (smoke test), so the cap was raised to 512 → 3.5% truncated, mean 158 tokens. pool_sha1 8afc0922.
- Scored the pool with all six reward models (`src/eval/score_pool.py`). Picks-the-longest 0.28–0.29 (HH) and 0.23–0.27 (UF) vs chance 0.125.
- Wrote `src/eval/best_of_n.py` (flip rate vs seed floor, Kendall τ, selected length) with `tests/test_flip_rate.py`.

## 2026-10-22
- `requirements.txt` regenerated with the env active (the 15 Oct freeze had captured the cluster's module Python).
- Loaders: HH splitter parses 4,686 of 4,708 test transcripts cleanly; the 22 anomalies are dataset quirks ("Humans:" typos, doubled assistant turns) and are dropped as `bad_turns`.
  - HH test 2,354 → 2,333 kept (empty 4, prefix_mismatch 9, bad_turns 6, identical 2); HH train 43,835 → 43,537 (bad_turns 59, empty 50, prefix_mismatch 128, identical 61).
  - UF test 2,000 → 1,985 (identical 12, empty 3); UF train 61,135 → 60,630 (identical 424, empty 81). 5 unit tests pass.
- Length-only baseline (`results/length_baseline.json`, Qwen tokenizer, ties = 0.5 credit): hh_test 0.601 [0.583, 0.622], uf_test 0.565 [0.545, 0.586]; mean chosen/rejected length 56/43 (HH) and 282/242 (UF) tokens. Human labels track length slightly more than GPT-4 labels here; caveats: UF responses ~5× longer, UF "rejected" is a random lower-scored completion, different prompts.
- Training script (`src/train/train_rm.py`, TRL 1.15.0 RewardTrainer, Bradley-Terry, 20k seeded subsample, run_meta.json per run). transformers 5 removed `warmup_ratio` → `warmup_steps: 0.03`. Smoke test job 1895033 (64 pairs, 2 steps) passed; ~1.6 s/step on HH.
- CORE seed 0: HH job 1895062 (10.0 min, 5 pairs filtered), UF job 1895063 (15.8 min, 1,246 pairs = 6.2% filtered by the 1,024-token limit).
- Evaluation (`src/eval/pairwise.py`): HH s0 → hh_test 0.706 [0.688, 0.725], uf_test 0.565 [0.543, 0.585]; UF s0 → hh_test 0.616 [0.598, 0.635], uf_test 0.683 [0.663, 0.703]. Seeds 1–2 launched.

## 2026-10-17
- Smoke job 1821752 (rack02-11): L40S 46 GB, driver 595.91 (CUDA 13.2), offline data + model load OK, logits shape (2, 1); queue wait ~1 s.
- Datasets saved with `save_to_disk` to /scratch/marzii/rmr/data (HH helpful-base 43,835 / 2,354; UF train_prefs 61,135 / test_prefs 2,000) because offline `load_dataset(data_dir=...)` fails in datasets 5.1.0.
- Qwen2.5-0.5B-Instruct tokenizer: pad `<|endoftext|>`, eos `<|im_end|>`; `model.config.pad_token_id` is None on load and must be set before batched scoring.

## 2026-10-15
- Env /scratch/marzii/envs/rmr (miniforge3, Python 3.11); `PIP_CONFIG_FILE=/dev/null` on the env so pip ignores the Alliance wheelhouse; `HF_HOME=/scratch/marzii/hf_cache` set explicitly in every command and sbatch script (a shell hook overrides it otherwise).
- Pinned torch 2.14.1+cu130, transformers 5.19.0, trl 1.15.0, datasets 5.1.0 (one release behind the same-day 5.20.0 / 1.16.0 on purpose).
- Models cached: Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775; Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.