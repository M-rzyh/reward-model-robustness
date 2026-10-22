## 2026-10-22

Regenerated requirements.txt from pip freeze with the env active; the 15 Oct freeze had been taken from the cluster’s module Python by mistake.

The two multi-turn examples parse exactly as they should, and 4,686 of 4,708 transcripts (99.5%) parse cleanly. Write the two counts in WORKLOG.

The loader works. The numbers add up exactly:

test: 2,333 kept + 4 + 9 + 6 + 2 = 2,354 ✓
train: 43,537 kept + 59 + 50 + 128 + 61 = 43,835 ✓
unique pair_ids = kept, so no duplicates, and the record has the exact shape from spec §3.2.

UF has 424 training rows where chosen and rejected are the same text (0.7%).


**Training script (Step 3)**
- `src/train/train_rm.py` + `configs/rm_base.yaml` + `scripts/train_rm.sh`. TRL 1.15.0 `RewardTrainer`,
  Bradley-Terry loss, centring term off, 20k seeded subsample, run_meta.json per run.
- transformers 5 removed `warmup_ratio`; using `warmup_steps: 0.03` (float = fraction of total steps).
- Smoke test, job 1895033 (64 pairs, 2 optimizer steps): path works end to end, model + run_meta saved,
  0 pairs filtered by max_length. ~1.6 s/step on HH → full run ≈ 17 min est.
- Launched CORE seed 0: HH job 1895062, UF job 1895063. Outputs in /scratch/marzii/rmr/runs/core_{hh,uf}_q0_s0/.

**Length-only baseline (Step 1c)** — `results/length_baseline.json`, Qwen tokenizer, ties = 0.5 credit
- hh_test  n=2333  len chosen/rejected 56/43   chosen longer 0.594  ties 0.015  acc 0.601 [0.583, 0.622]
- uf_test  n=1985  len chosen/rejected 282/242 chosen longer 0.561  ties 0.008  acc 0.565 [0.545, 0.586]
- hh_train n=43537 … acc 0.588 [0.583, 0.593] · uf_train n=60630 … acc 0.557 [0.553, 0.561]
- Note: human (HH) labels follow length *more* than GPT-4 (UF) labels here. Caveat: UF responses ~5× longer,
  UF "rejected" is a random lower-scored completion, different prompts.

**Loaders (Step 1a–b)** — common schema `{pair_id, source, prompt, chosen, rejected}`
- HH test: kept 2333, dropped {empty_response 4, prefix_mismatch 9, bad_turns 6, identical 2}
- HH train: kept 43537, dropped {bad_turns 59, empty_response 50, prefix_mismatch 128, identical 61}
- UF test_prefs: kept 1985, dropped {identical 12, empty_response 3}
- UF train_prefs: kept 60630, dropped {identical 424, empty_response 81}
- bad_turns = roles not alternating user/assistant (HH has "Humans:" typos and doubled assistant turns).
- 5 unit tests for the HH splitter pass (`pytest`).

2026-10-17

Smoke job 1821752 on node rack02-11: L40S 46 GB, driver 595.91 (CUDA 13.2), cuda available: True, logits shape (2, 1). Queue wait about 1 s, runtime 25 s.
Data saved with save_to_disk to /scratch/marzii/rmr/data/: HH helpful-base train 43,835 / test 2,354; UF train_prefs 61,135 / test_prefs 2,000. Reason: offline load_dataset(data_dir=...) fails in datasets 5.1.0.
Qwen2.5-0.5B-Instruct tokenizer: pad <|endoftext|>, eos <|im_end|>; model.config.pad_token_id is None, so it must be set from the tokenizer before batched scoring.

2026-10-15

Env /scratch/marzii/envs/rmr (miniforge3, Python 3.11). PIP_CONFIG_FILE=/dev/null on the env so pip ignores the Alliance wheelhouse. HF_HOME=/scratch/marzii/hf_cache, but a shell hook overrides it, so it’s set explicitly in every command and sbatch script.
Pinned: torch 2.14.1+cu130, transformers 5.19.0, trl 1.15.0, datasets 5.1.0 (one release behind the same-day 5.20.0 / 1.16.0 on purpose).
Models cached: Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775; Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.