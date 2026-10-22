2026-10-22

Regenerated requirements.txt from pip freeze with the env active; the 15 Oct freeze had been taken from the cluster’s module Python by mistake.

The two multi-turn examples parse exactly as they should, and 4,686 of 4,708 transcripts (99.5%) parse cleanly. Write the two counts in WORKLOG.

The loader works. The numbers add up exactly:

test: 2,333 kept + 4 + 9 + 6 + 2 = 2,354 ✓
train: 43,537 kept + 59 + 50 + 128 + 61 = 43,835 ✓
unique pair_ids = kept, so no duplicates, and the record has the exact shape from spec §3.2.

UF has 424 training rows where chosen and rejected are the same text (0.7%).

2026-10-17

Smoke job 1821752 on node rack02-11: L40S 46 GB, driver 595.91 (CUDA 13.2), cuda available: True, logits shape (2, 1). Queue wait about 1 s, runtime 25 s.
Data saved with save_to_disk to /scratch/marzii/rmr/data/: HH helpful-base train 43,835 / test 2,354; UF train_prefs 61,135 / test_prefs 2,000. Reason: offline load_dataset(data_dir=...) fails in datasets 5.1.0.
Qwen2.5-0.5B-Instruct tokenizer: pad <|endoftext|>, eos <|im_end|>; model.config.pad_token_id is None, so it must be set from the tokenizer before batched scoring.

2026-10-15

Env /scratch/marzii/envs/rmr (miniforge3, Python 3.11). PIP_CONFIG_FILE=/dev/null on the env so pip ignores the Alliance wheelhouse. HF_HOME=/scratch/marzii/hf_cache, but a shell hook overrides it, so it’s set explicitly in every command and sbatch script.
Pinned: torch 2.14.1+cu130, transformers 5.19.0, trl 1.15.0, datasets 5.1.0 (one release behind the same-day 5.20.0 / 1.16.0 on purpose).
Models cached: Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775; Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.