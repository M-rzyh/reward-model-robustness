# Sprint spec: do human- and GPT-4-trained reward models pick the same response?

**Status:** in progress, started 2026-10-14. This file is the technical spec for the first slice of the project.
The long-term plan is in `README.md` / `PLAN.md`. Anything not listed here is out of scope for this slice.
Every number in the results comes from `results/*.json` and `experiments/index.csv`. No number is written by hand.

---

## 1. The question

> **Do reward models trained on human vs. GPT-4 preference labels (and on length-biased human labels) pick the same
> best-of-8 response?**

**Headline metric:** the **rank-flip rate** between configurations on a frozen best-of-8 pool, compared with the
**seed-to-seed flip rate** (the noise floor) of the same configuration.

**Why.** This continues the AAAI-27 student abstract *"Mind the Gap: Human and Synthetic Feedback Performance
Diverge in Learning from Preferences"* (under review). That work found human and synthetic feedback diverge in
control tasks. This slice asks whether a similar divergence shows up in reward models for language models.
Public preference datasets already contain the human-vs-AI label axis, so the first experiment needs no artificial
corruption. Accuracy alone is not the interesting number. A reward model can keep its accuracy and still change
which response wins. That change is what reaches a policy through best-of-n or RL.

---

## 2. Design at a glance

| Block | What | Runs | Status gate |
|---|---|---|---|
| **CORE** | Bradley-Terry RMs on HH-RLHF (human labels) and UltraFeedback (GPT-4 labels); **2×2 cross-evaluation** (train on each, evaluate on both test sets); a **length-only baseline** ("pick the longer response") on both test sets | 2 datasets × 3 seeds = **6** | must finish |
| **CORE+** | Frozen best-of-8 candidate pool (500 prompts × 8). Score with every CORE RM. **Rank-flip rate** cross-config vs. the **seed floor**; mean **Kendall τ**; **selected-response length** | 1 generation job + scoring | after CORE |
| **STRETCH A** | **Length-bias operator** on HH training labels, q ∈ {0.1, 0.2, 0.3} (q = 0 reuses CORE). Same metrics as CORE + CORE+ | 3 q × 3 seeds = **9** | after CORE+ code works |
| **STRETCH B** | **DPO** on a 0.5B instruct policy, clean vs. q = 0.3 HH labels. Measure output length and clean-RM score | 2 × 3 seeds (or 2 × 2, labelled *pilot*) | **gated** (§12) |

Out of scope for this slice (kept in the long-term plan): position bias, annotator heterogeneity, difficulty-
correlated noise, OASST, ensembling as a mitigation, noise-aware losses, PPO/GRPO, a served inference API.

---

## 3. Data

### 3.1 Sources
| Role | HF id | Subset / split | Train size | Eval |
|---|---|---|---|---|
| Human labels | `Anthropic/hh-rlhf` | `data_dir="helpful-base"`; `train` / `test` | **20,000** pairs, seeded subsample | full helpful-base `test` (≈2.3k; record exact `len()`) |
| GPT-4 labels | `HuggingFaceH4/ultrafeedback_binarized` | `train_prefs` / `test_prefs` | **20,000** pairs, seeded subsample | full `test_prefs` (≈2,000; record exact `len()`) |

- **helpful-base only.** Harmless-base is mostly about refusals and is not comparable with UltraFeedback.
- **Equal train size (20k each)** so dataset size is not a hidden variable.
- Record the dataset **revision (commit SHA)** of each dataset in `run_meta.json`.
- UltraFeedback-binarized: chosen = highest-rated completion, rejected = a random lower-rated one (per the dataset
  card; verify on 10 rows and note it in WORKLOG).
- `trl-lib/hh-rlhf-helpful-base` exists as a preprocessed variant. Do **not** depend on it unless it is checked
  against the raw data on 10 examples. The raw dataset plus our own splitter is the reference.

### 3.2 Common schema
Both loaders return records with exactly these fields:

```
{pair_id: str, source: "hh" | "uf", prompt: list[message], chosen: list[message], rejected: list[message]}
message = {"role": "user" | "assistant", "content": str}
```

- `prompt` is the conversation up to (not including) the final assistant turn. `chosen` / `rejected` each hold
  exactly one assistant message.
- **Chat format for both datasets**, rendered with the backbone tokenizer's chat template at train, eval and pool-
  scoring time. One input format for every RM keeps the 2×2 and the pool scoring comparable.
- `pair_id = sha1(source + canonical_json(prompt) + chosen_text + rejected_text)[:16]`.

### 3.3 HH-RLHF split rule
Raw HH rows are two full transcripts (`chosen`, `rejected`) that share a prefix and differ in the final assistant
turn.
1. Split each transcript at the **last** occurrence of `"\n\nAssistant:"`. Left part = prompt text; right part =
   the response (strip leading whitespace).
2. The prompt text of `chosen` and `rejected` must be **identical**. If not, drop the row and count it
   (`n_dropped_prefix_mismatch`).
3. Parse the prompt text into turns on `"\n\nHuman:"` / `"\n\nAssistant:"` → `user` / `assistant` messages.
4. Drop rows with an empty response or `chosen == rejected` (count each reason).
5. Unit test on hand-written transcripts, including multi-turn ones and one with "Assistant:" inside the text.

### 3.4 UltraFeedback rule
`chosen` / `rejected` are already message lists that include the user prompt. Take everything except the final
assistant message as `prompt` (check both share it), the final assistant message as the response. Same drop rules.

### 3.5 Subsampling and splits
- Subsample 20k from each train split with `numpy.random.default_rng(seed)`. Write the sorted list of `pair_id`s
  and its hash (`subsample_hash`) to `run_meta.json`.
- Test sets are never subsampled, never relabelled, never shuffled in a way that changes content.
- **No train/test overlap by `pair_id`** (unit test). Also report (not fail on) exact-prompt overlap counts.

### 3.6 Length statistics (first thing computed, before any training)
Response length = number of tokens of the response text under the backbone tokenizer (no chat template).
Per dataset and split: mean/median length of chosen and rejected; fraction of pairs where chosen is longer;
fraction of ties.

---

## 4. Models

### 4.1 Reward-model backbone
- `Qwen/Qwen2.5-0.5B-Instruct` with a scalar head (`AutoModelForSequenceClassification`, `num_labels=1`).
  Pin the exact model revision in `run_meta.json`. Pad token set explicitly if missing.
- **Why a decoder and not DeBERTa-v3-base:** DeBERTa's practical context is 512 tokens. Multi-turn HH transcripts
  would be truncated, and **truncation hides response length**, which would quietly break the length-bias study.
  A decoder with `max_length = 1024` avoids most of this.
- **Truncation is measured, not assumed.** Log the fraction of pairs whose rendered `prompt + response` exceeds
  `max_length` (per dataset, per split) and how the installed TRL version handles them (truncate vs. drop). Report
  it in the README.
- **Why small:** the question is about the labels, not model capacity.

### 4.2 Generator (candidate pool)
- `Qwen/Qwen2.5-1.5B-Instruct`, temperature 1.0, top_p 1.0, `max_new_tokens 256`, n = 8 samples per prompt, fixed
  seed. vLLM if it installs cleanly on the cluster, otherwise batched HF `generate`. Record which, and the revision.
- Different model from the RM backbone, so no RM is scoring its own outputs.

---

## 5. Training

- **Loss:** Bradley-Terry pairwise, `L = −log σ(r(x, y_chosen) − r(x, y_rejected))`. TRL `RewardTrainer` default.
  The optional **margin** term and **reward centering** (`center_rewards_coefficient`) are **off**. Written in config.
- **Library:** `torch`, `transformers`, `trl` (`RewardTrainer`, `RewardConfig`), `datasets`, `accelerate`. Versions
  pinned in `requirements.txt`. The TRL reward API has changed between versions: follow the docs of the pinned
  version only.
- **Frozen hyper-parameters (identical for every run in every cell):**

| Param | Value |
|---|---|
| epochs | 1 |
| optimizer / lr | AdamW, **1e-5** |
| schedule | cosine, 3% warmup |
| effective batch | 32 (e.g. per-device 8 × grad-accum 4; or 4 × 8 if memory needs it) |
| precision | bf16 |
| gradient checkpointing | on |
| `max_length` | 1024 |
| weight decay | 0.0 |
| eval during training | off (evaluation is a separate step, §7) |

- **No hyper-parameter search.** If the HH baseline is unstable (loss diverges or NaN), lr 5e-6 may be tried **once**,
  then it becomes the frozen value for every cell and the change is recorded in WORKLOG.
- Checkpoints: keep the final model only, outside git.

---

## 6. Grid and seeds

| Block | Train set | Label condition | Seeds | Runs |
|---|---|---|---|---|
| CORE | HH | clean (human) | 0, 1, 2 | 3 |
| CORE | UF | clean (GPT-4) | 0, 1, 2 | 3 |
| STRETCH A | HH | length-bias q = 0.1, 0.2, 0.3 | 0, 1, 2 | 9 |
| STRETCH B | HH (DPO) | clean, q = 0.3 | 0, 1, 2 (or 0, 1 = pilot) | 4–6 |

The seed controls the subsample, the scalar-head init, the data order, and the noise operator (one seed, logged).
With 3 seeds, results are labelled **preliminary** (the long-term plan asks for ≥5 seeds for headline numbers).

---

## 7. Metrics and evaluation protocol

**Rule: evaluation labels are always the original, clean labels.** Noise touches training labels only.

1. **Pairwise accuracy** = fraction of test pairs with `r(chosen) > r(rejected)`, for every RM on **both** test sets
   (gives the 2×2 for CORE). Report per seed, mean ± std over seeds, and a **95% bootstrap CI** over pairs
   (1,000 resamples, seeded). Ties (`r_c == r_r`) count as wrong and are counted.
2. **Length-only baseline** = accuracy of "pick the longer response" on each test set. Ties = 0.5 credit, reported.
   Every RM number is shown next to it.
3. **Rank-flip rate (headline).** On the frozen pool, each RM picks `argmax` over the 8 candidates per prompt.
   Flip rate(A, B) = fraction of prompts where the picks differ. Report:
   - **within-config (seed floor):** all seed pairs of the same config (3 pairs for 3 seeds), mean and range;
   - **cross-config:** HH-RM vs. UF-RM (all 3 × 3 = 9 seed pairs), and clean-HH vs. each q (9 pairs each);
   - a bootstrap CI over prompts for each.
   A cross-config rate is only meaningful **relative to the seed floor**.
4. **Kendall τ** between the two RMs' full rankings of the 8 candidates, averaged over prompts (same pairings as 3).
5. **Selected-response length:** mean token length of each RM's best-of-8 pick, and of the pool as a whole.
6. **(Stretch B)** On 200 held-out HH test prompts **not in the pool**: mean response length and mean clean-HH-RM
   score (average of the 3 CORE HH RMs) of each DPO policy and of the base policy. Limitation: the judge is one of our
   own RMs. A held-out judge is a next step.
7. *(Optional)* ECE of `σ(r_c − r_r)` on the clean test sets, 10 equal-width bins.

Sanity rules: accuracy **below 50%** on the RM's own test set means a bug (orientation, split), not a finding. Every
trained RM must beat 50% and should be compared with the length baseline before anything else is run on it.

---

## 8. Noise operator: length bias (`src/noise/length_bias.py`)

**Definition.** For each training pair, let `L(y)` be the token length of response `y`. If `L(rejected) > L(chosen)`
(the shorter response was chosen), then with probability `q` swap `chosen` and `rejected`. Otherwise leave the pair
unchanged. So with probability `q` the annotator "prefers the longer response".

**Rules.**
- Seeded: one `numpy.random.default_rng(seed)` draw per pair, in a fixed pair order (sorted by `pair_id`), so the
  output is **deterministic given (seed, q)** and independent of input order.
- **Tie rule:** if `L(chosen) == L(rejected)`, the pair is never flipped. Documented and tested.
- Applied **only** to the training subsample. Eval splits never pass through it.
- Returns the new records plus a stats dict (`n_eligible`, `n_flipped`, `realised_flip_rate`), saved in `run_meta.json`.

**The 4 unit tests (`tests/test_length_bias.py`).**
1. **q = 0 → identity:** output equals input record-for-record.
2. **q = 1 → degenerate:** in every non-tied pair, `chosen` is the longer response; tied pairs unchanged.
3. **Rate tolerance:** on 10k synthetic pairs at q = 0.2, `n_flipped / n_eligible` is within ±0.02 of 0.2, and pairs
   where the longer response was already chosen are never touched. Plus: same `(seed, q)` → identical output;
   different seed → different output.
4. **Eval untouched:** the serialized eval splits are byte-identical before and after building any noisy training set.

This follows the same discipline as earlier corruption operators in this line of work (seeded, train-only, tested
at both extremes).

---

## 9. Provenance

### 9.1 `run_meta.json` (one per run, written at start and updated at end)
```
run_id, block (core|stretch_a|stretch_b|pool|eval), train_source (hh|uf), label_condition (clean|length_bias),
q, seed,
dataset_ids + revisions, subsample_size, subsample_hash, n_dropped_{prefix_mismatch,empty,identical},
noise_stats {n_eligible, n_flipped, realised_flip_rate},
model_id + revision, max_length, truncation_rate_train,
hyperparams {lr, epochs, eff_batch, per_device_batch, grad_accum, warmup, schedule, bf16, grad_ckpt, margin, centering},
lib_versions {python, torch, transformers, trl, datasets, accelerate, vllm?},
git_sha, git_dirty (bool), config_path,
cluster, slurm_job_id, slurm_array_task_id, gpu_type, n_gpus,
start_time, end_time, wall_time_s, status (running|done|failed),
final_train_loss, metrics {acc_hh, acc_uf, ci_hh, ci_uf, ...}
```

### 9.2 `experiments/index.csv` (append-only, header written only if the file is new)
```
run_id,block,train_source,label_condition,q,seed,config,slurm_job_id,cluster,status,out_dir,acc_hh,acc_uf,date
```
Submitters append a `submitted` row; the eval step appends a `done` row with metrics. Never truncate the file.

### 9.3 Results files
- `results/eval/<run_id>.json` — accuracies, CIs, ties, per test set.
- `results/baselines/length_only.json` — length baseline + length stats.
- `results/pool/pool_meta.json` + `pool.jsonl` (outside git if large; hash recorded).
- `results/best_of_n/<pairing>.json` — flip rates, Kendall τ, selected lengths.
- `results/tables/*.md` and `figures/*.png` are **generated** from the JSON by a script, never edited by hand.

---

## 10. Repo layout

```
reward-model-robustness/
├── README.md               # question, setup, results table, limitations, repro commands
├── PLAN.md                 # long-term spec (status line updated)
├── SPRINT_SPEC.md          # this file
├── WORKLOG.md              # dated entries, newest at top
├── WRITEUP.md              # 1–2 pages, at the end
├── LICENSE
├── requirements.txt        # pinned
├── .gitignore              # checkpoints/, outputs/, wandb/, *.safetensors, *.bin, *.pt, .env, hf_cache/
├── configs/
│   ├── rm_base.yaml        # frozen hyper-parameters
│   ├── data.yaml           # dataset ids, subsets, sizes
│   ├── pool.yaml           # generator, n, temperature, prompts
│   └── dpo.yaml            # Stretch B
├── src/
│   ├── data/               # hh.py, uf.py, schema.py, subsample.py, length_stats.py
│   ├── noise/              # length_bias.py
│   ├── train/              # train_rm.py, train_dpo.py, run_meta.py
│   ├── eval/               # pairwise.py, length_baseline.py, best_of_n.py, bootstrap.py
│   └── gen/                # make_pool.py
├── scripts/                # SLURM: smoke, CORE array, Stretch A array, pool, eval, DPO; prefetch on login node
├── experiments/index.csv
├── results/                # small JSON only
├── figures/                # ≤ 3 plots, generated
└── tests/                  # test_schema.py, test_hh_split.py, test_length_bias.py, test_flip_rate.py, test_no_overlap.py
```

---

## 11. README requirements (end of the slice)

- One-line question and status ("in progress", date started).
- Short "why" linking to the AAAI-27 student abstract (**under review**).
- Setup: datasets + revisions, backbone + revision, loss, frozen hyper-parameters, seeds.
- **Results table:** length-only baseline; the 2×2 (mean ± std over seeds, 95% CI); within- vs. cross-config flip
  rates with Kendall τ; selected-response length; Stretch A/B rows only if they ran. Null results stated as null.
- **Limitations:** the dataset confound (§13); 3 seeds (preliminary); one small backbone; one generator; no external
  judge; truncation rate; English-only, helpful-base only.
- **Reproduce:** copy-paste commands (env setup, prefetch, tests, each sbatch, table/figure regeneration).
- Next steps (§13).

---

## 12. Gates and stop rules

- **After the first CORE runs:** at least one HH and one UF RM finished; accuracy on its own test set clearly above
  50% **and** above the length baseline; no NaNs; chosen/rejected orientation hand-checked on 10 pairs. If not:
  debug (bounded time). If the baseline still fails, stop the slice and report it as not reached.
- **Stretch B starts only if all hold:** 2×2 and CORE best-of-n numbers exist; ≥ 6 of 9 Stretch A runs finished
  cleanly; the DPO CPU smoke test passed.
- **Cut order if behind:** Stretch B → ECE → Stretch A reduced to q ∈ {0, 0.3} → pool reduced to 200 × 4 → UF arm to
  2 seeds (labelled). **Never cut:** the trained baseline, held-out accuracy, the length-only control, the README,
  the tests, `run_meta.json`.
- **Experiment freeze** before write-up: after the freeze no new runs, only analysis of finished runs.

---

## 13. Known confound and next step

HH-RLHF and UltraFeedback differ in **prompts and responses**, not only in who labelled them. The 2×2 therefore
measures **label source and distribution shift together**. Two things in this slice partly help: the length-only
baseline (shows how much each label source tracks length) and the best-of-8 pool (both RMs score the *same*
responses). The clean fix is a **same-pairs relabel**: label ~5k HH-RLHF pairs with an LLM judge and train on human
vs. judge labels for identical pairs. That is the first next step and is **not** in this slice.

Other next steps: ≥ 5 seeds; a held-out judge for best-of-n win rate; larger backbones; position-bias and annotator-
heterogeneity operators; downstream DPO/RL to test whether selection differences change policy behaviour.

---

## 14. Runtime estimates (EST — not measured; replace with timing-job numbers)

| Job | 1× L40S 48 GB | 1× H100 80 GB |
|---|---|---|
| One RM run, 0.5B, 20k pairs, 1 epoch, len 1024 | ~30–90 min (EST) | ~20–45 min (EST) |
| Pool generation, 500 × 8, 1.5B, 256 tokens | ~10–30 min vLLM; ~1–2 h HF generate (EST) | less |
| Scoring the pool with ~15 RMs + all eval sets | ~30–60 min total (EST) | less |
| One DPO run, 0.5B, 10k pairs + eval generation | ~1–2 h (EST) | ~40–80 min (EST) |
| CORE + STRETCH A total | ~15–25 GPU-h (EST) | |

---

## 15. Milestones

| # | Milestone | Done when | Target |
|---|---|---|---|
| M1 | Data + baseline | Both loaders → common schema; tests green; length stats + length-only baseline in `results/`; smoke + timing job ran | Oct 14, 2026 |
| M2 | CORE trained | 6 RMs finished, `run_meta.json` each, indexed | Oct 14–15, 2026 |
| M3 | 2×2 table | All 6 RMs evaluated on both test sets, CIs, table generated | Oct 15, 2026 |
| M4 | Best-of-8 headline | Pool frozen; within- vs. cross-config flip rates, Kendall τ, selected length | Oct 15–16, 2026 |
| M5 | Stretch A | 9 length-bias runs evaluated; accuracy / flip / length vs. q | Oct 16, 2026 |
| M6 | Stretch B | DPO clean vs. q = 0.3, gated (§12) | Oct 16–17, 2026 |
| M7 | README / WRITEUP / tag | Results table, limitations, repro commands; `pytest` green; tag `v0.1-<freeze date>` | Oct 17, 2026 |
