# Do human- and GPT-4-trained reward models pick the same response?

**Status:** in progress, started 2026-10-14; last updated 2026-10-27. 5 seeds per configuration.

Reward models turn preference labels into the training signal for language models. This project trains the same
reward model on **human** preference labels (Anthropic HH-RLHF, helpful-base) and on **GPT-4** preference labels
(UltraFeedback-binarized), and asks two questions: do they agree on held-out pairs, and, more importantly, **do they
select the same response** when used for best-of-n?

Every number below comes from the files in `results/` and `experiments/index.csv`. The tables are inserted by
`python -m src.eval.readme_tables` and are not edited by hand.

## Why

This continues the AAAI-27 student abstract *"Mind the Gap: Human and Synthetic Feedback Performance Diverge in
Learning from Preferences"* (under review), which found that human and synthetic feedback lead to different results
in preference learning for control tasks. Here the same question is asked for language-model reward models. Public
datasets already carry the human-vs-AI label axis: HH-RLHF was labelled by crowdworkers (Bai et al., 2022), while
UltraFeedback was rated by GPT-4 (Cui et al., 2023), the kind of AI feedback used in RLAIF (Lee et al., 2023).
Reward models are known to pick up surface features such as length (Singhal et al., 2023) and agreement with the
user (Sharma et al., 2023), so a reward model can hold its accuracy and still change which response wins. That
change is what reaches a policy through best-of-n or RL.

## Setup

| | |
|---|---|
| Data | HH-RLHF `helpful-base` (train / test) and `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` / `test_prefs`), mapped to one schema `{pair_id, source, prompt, chosen, rejected}` |
| Training pairs | 20,000 per run, seeded subsample, then pairs longer than 1,024 tokens are dropped (table below) |
| Backbone | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` with a scalar head |
| Loss / trainer | Bradley-Terry, TRL 1.15.0 `RewardTrainer` |
| Hyper-parameters | lr 1e-5, cosine, 3% warmup, effective batch 32 (4 × 8 accumulation), 1 epoch, bf16 training, max length 1,024 |
| Seeds | 5 per training source (0–4); the seed sets both the subsample and the training order |
| Evaluation | full HH helpful-base test (2,333 pairs kept) and UF `test_prefs` (1,985 kept); a pair counts as correct only if r(chosen) > r(rejected), so ties count as wrong |
| Best-of-8 pool | 500 HH-test prompts × 8 responses from `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` (T = 1.0, top-p 1.0, top-k off, no repetition penalty, ≤512 new tokens, seed 1234), frozen, sha1 `8afc0922`; every model scores the same 4,000 candidates |

**Loader drop counts.** HH test 2,354 → 2,333 (empty response 4, prompt-prefix mismatch 9, roles not alternating 6,
chosen = rejected 2). HH train 43,835 → 43,537 (59 / 50 / 128 / 61 in the order bad turns, empty, prefix mismatch,
identical). UF test 2,000 → 1,985 (identical 12, empty 3). UF train 61,135 → 60,630 (identical 424, empty 81).

**Training runs** (from `experiments/index.csv`):

<!-- BEGIN results/runs.md -->
| run | train pairs after 1,024-token filter | filtered (of 20,000) | optimizer steps | train time (min) |
|---|---|---|---|---|
| core_hh_q0_s0 | 19,995 | 5 (0.0%) | 625 | 10.0 |
| core_hh_q0_s1 | 19,996 | 4 (0.0%) | 625 | 9.9 |
| core_hh_q0_s2 | 19,995 | 5 (0.0%) | 625 | 10.1 |
| core_hh_q0_s3 | 19,995 | 5 (0.0%) | 625 | 10.1 |
| core_hh_q0_s4 | 19,995 | 5 (0.0%) | 625 | 9.9 |
| core_uf_q0_s0 | 18,754 | 1,246 (6.2%) | 587 | 15.8 |
| core_uf_q0_s1 | 18,723 | 1,277 (6.4%) | 586 | 15.7 |
| core_uf_q0_s2 | 18,709 | 1,291 (6.5%) | 585 | 16.0 |
| core_uf_q0_s3 | 18,743 | 1,257 (6.3%) | 586 | 15.7 |
| core_uf_q0_s4 | 18,752 | 1,248 (6.2%) | 586 | 15.9 |
<!-- END results/runs.md -->

UltraFeedback responses are about five times longer than HH responses (mean chosen length 282 vs 56 tokens), so
the 1,024-token filter removes about 6% of UF training pairs and almost no HH pairs. The two sources therefore train
on slightly different numbers of pairs. At evaluation no pair is dropped; long pairs are truncated to 1,024 tokens.

## Results

### 1. Held-out pairwise accuracy (2×2)

Mean ± sample std over 5 seeds. The length-only baseline picks the longer response (ties get half credit).

<!-- BEGIN results/summary_2x2_fp32.md -->
| trained on | seeds | hh_test (human labels) | uf_test (GPT-4 labels) |
|---|---|---|---|
| length-only baseline | – | 0.601 | 0.565 |
| RM trained on hh (q=0) | 5 | 0.709 ± 0.004 | 0.551 ± 0.016 |
| RM trained on uf (q=0) | 5 | 0.631 ± 0.008 | 0.689 ± 0.004 |
<!-- END results/summary_2x2_fp32.md -->

In-source accuracy is 11–12 points above the length baseline. Cross-source accuracy falls to about the length
baseline in both directions: the human-trained model scores below it on GPT-4 labels, and the GPT-4-trained model
scores 3 points above it on human labels. The datasets differ in prompts and responses as well as in who labelled
them, so this table alone cannot separate the label source from the data; the best-of-8 pool below addresses part of
that.

### 2. Best-of-8 selection on a shared pool

For each prompt, a model's pick is the highest-scoring of the 8 candidates. The **flip rate** between two models is
the fraction of the 500 prompts where they pick different candidates. Within-config pairs (two seeds of the same
training source) give the noise floor; cross-config pairs are all 25 HH-seed × UF-seed combinations. Kendall τ-b
compares the full ranking of the 8 candidates and is averaged over prompts. The gap CI resamples prompts, because
the pairings share models and their spread is not an independent uncertainty.

<!-- BEGIN results/best_of_n/fp32/summary.md -->
Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 10 | 40.0 ± 3.1 | 0.715 ± 0.019 |
| within_uf | 10 | 37.6 ± 2.8 | 0.659 ± 0.030 |
| cross_hh_uf | 25 | 64.2 ± 3.5 | 0.397 ± 0.032 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 24.1 | [21.2, 27.3] |
| cross minus within uf | 26.6 | [23.1, 29.6] |
| cross minus within both | 25.4 | [22.8, 27.9] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 5 | 209.5 ± 5.8 | 30.8 ± 2.7 |
| trained on uf | 5 | 205.7 ± 4.5 | 26.4 ± 1.3 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
<!-- END results/best_of_n/fp32/summary.md -->

On the same best-of-8 pool, HH- and UF-trained reward models pick a different winner on 64.2 ± 3.5% of prompts,
against 40.0 ± 3.1% (HH seeds) and 37.6 ± 2.8% (UF seeds) between seeds of the same configuration. The cross-config
flip rate is 25.4 points above the seed floor (95% CI 22.8–27.9), and rankings agree much less across sources
(τ ≈ 0.40) than within a source (τ ≈ 0.66–0.72). The divergence is not a length effect: both configurations choose
responses of about the same length (206–210 tokens, against a pool mean of 158), and both pick the longest
candidate 2–2.5× more often than chance (the HH model slightly more often: 30.8% vs 26.4%). The seed floor is itself high: two seeds of the same model disagree on about 40%
of prompts, so best-of-8 selection with a 0.5B reward model is noisy before the label source is changed.

`results/best_of_n/fp32/disagreements_hh_s0_vs_uf_s0.md` lists the first five prompts where HH seed 0 and UF seed 0
disagree, with both chosen responses.

### Scoring precision

Training uses bf16. Scoring in bf16 rounds large rewards enough to create exact ties: up to 82 of 1,985 `uf_test`
pairs for one model, and up to 58 of 500 pool prompts with a tied top score. All results above are therefore scored
in float32 (`scripts/precision_check.sh`); the bf16 results are kept for reference and lead to the same
conclusions. In float32 the only remaining ties are real: 28 `uf_test` pairs whose prompt alone is longer than 1,024
tokens, so both responses are truncated away (identical inputs; counted as wrong for every model, which caps
`uf_test` accuracy at 0.986), and 5–7 pool prompts per model whose top candidates are duplicate texts (28 prompts
in the pool contain duplicates; one has 8 identical candidates, so τ is undefined there and it is skipped).

<details><summary>bf16 scoring (reference)</summary>

<!-- BEGIN results/summary_2x2.md -->
| trained on | seeds | hh_test (human labels) | uf_test (GPT-4 labels) |
|---|---|---|---|
| length-only baseline | – | 0.601 | 0.565 |
| RM trained on hh (q=0) | 5 | 0.705 ± 0.004 | 0.547 ± 0.017 |
| RM trained on uf (q=0) | 5 | 0.623 ± 0.008 | 0.686 ± 0.004 |
<!-- END results/summary_2x2.md -->

<!-- BEGIN results/best_of_n/summary.md -->
Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 10 | 41.1 ± 3.6 | 0.702 ± 0.020 |
| within_uf | 10 | 38.3 ± 2.5 | 0.649 ± 0.027 |
| cross_hh_uf | 25 | 64.9 ± 3.5 | 0.392 ± 0.032 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 23.7 | [20.6, 27.1] |
| cross minus within uf | 26.5 | [23.2, 29.5] |
| cross minus within both | 25.1 | [22.6, 27.7] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 5 | 208.0 ± 6.0 | 30.6 ± 2.4 |
| trained on uf | 5 | 205.5 ± 5.3 | 25.9 ± 1.7 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
<!-- END results/best_of_n/summary.md -->

</details>

## Limitations

- **Dataset confound.** HH and UF differ in prompts, response style and length as well as in labeller. The shared
  pool removes the confound on the *evaluation* side (every model ranks the same candidates), not on the training
  side. The pool prompts come from HH test, so they are in-domain for the HH model and out-of-domain for the UF model.
- **5 seeds** per configuration: enough for the headline gap, but small for per-seed spread.
- **One small backbone** (0.5B). Larger reward models may agree more with each other, or less.
- **Unequal training sizes** after the 1,024-token filter (about 18.7k UF vs 20.0k HH pairs).
- **One generator** for the pool (Qwen2.5-1.5B-Instruct), 512-token cap with 3.5% of candidates truncated, 28 of 500
  prompts with duplicate candidates.
- **No held-out judge**, so the results say the selections differ, not which selection is better.
- The AAAI-27 abstract this builds on is under review.

## Reproduce

Paths assume the Vulcan layout (`/scratch/marzii/...`); change them in `src/data/schema.py` and the `scripts/`.

```bash
conda activate /scratch/marzii/envs/rmr                         # versions pinned in requirements.txt
python scripts/prefetch_data.py                                 # needs internet; saves both datasets to disk
python -m pytest -q tests

for src in hh uf; do for s in 0 1 2 3 4; do SOURCE=$src SEED=$s sbatch scripts/train_rm.sh; done; done
for r in core_{hh,uf}_q0_s{0..4}; do RUN_ID=$r sbatch scripts/eval_rm.sh; done        # bf16 eval + experiments/index.csv
sbatch scripts/make_pool.sh                                     # frozen best-of-8 pool
sbatch scripts/score_pool.sh                                    # bf16 pool scores
sbatch scripts/precision_check.sh                               # float32 eval + pool scores

sbatch scripts/cpu_job.sh bash -c "python -m src.eval.length_stats && python -m src.eval.summarize \
  && python -m src.eval.summarize --eval_dir results/eval_fp32 --out results/summary_2x2_fp32 \
  && python -m src.eval.best_of_n && python -m src.eval.best_of_n --score_dir results/best_of_n/scores_fp32 --out_dir results/best_of_n/fp32 \
  && python -m src.eval.runs_table && python -m src.eval.readme_tables"
```

## Next steps

- Relabel the *same* pairs with both sources, to remove the dataset confound.
- More seeds and a second backbone size, to see whether the seed floor shrinks.
- A held-out judge to score which selection is better (win rate), not just whether it differs.
- A length-bias noise operator on the human labels, q ∈ {0.1, 0.2, 0.3}, with the same metrics.
- DPO downstream: does the label source change what a trained policy writes?

## References

- Bai et al. 2022, *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*.
- Cui et al. 2023, *UltraFeedback: Boosting Language Models with High-quality Feedback*.
- Lee et al. 2023, *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*.
- Singhal et al. 2023, *A Long Way to Go: Investigating Length Correlations in RLHF*.
- Sharma et al. 2023, *Towards Understanding Sycophancy in Language Models*.
