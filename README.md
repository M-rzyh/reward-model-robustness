# Reward-Model Robustness Under Noisy Human Feedback

**Status:** not started. This README is the full plan — it is the spec to build against.
**Owner:** Marzieh Ghayour · **Written:** 2026-09-28

---

## 1. The question

Reward models are the component that turns human preferences into a training signal for
language models. They are trained on preference labels collected from people, and people are
noisy: they skim, they prefer the longer answer, they prefer whichever option was shown first,
they disagree with each other, and they make more mistakes on genuinely close calls.

**Central question:** how does realistic annotator noise degrade reward models, and — the part
that actually matters — **does that degradation change which responses get selected?**

A reward model losing three points of pairwise accuracy is a boring result. A reward model that
still scores 90% but *flips which answer wins* is the finding.

## 2. Why this project, specifically

This is a direct continuation of the AAAI-27 student abstract,
**"Mind the Gap: Human and Synthetic Feedback Performance Diverge in Learning from Preferences."**

That work showed human and synthetic feedback diverge in control tasks. This project asks
whether the same divergence breaks reward models for language models. Framing matters: this is
a **line of work**, not a portfolio exercise. Say it that way in cover letters and interviews.

### Free experiment, before any noise injection
The public preference datasets already contain the human-vs-synthetic axis:

| Dataset | Label source |
|---|---|
| **HH-RLHF** (Anthropic) | Human crowdworkers |
| **UltraFeedback** | GPT-4 — **synthetic** |
| **OpenAssistant (OASST)** | Human, with per-annotator metadata |

Train on synthetic labels, evaluate on human ones, and vice versa. That is the abstract's
experiment at LLM scale, with zero artificial corruption. **Run this first** — it is the result
that ties the project to existing work, and it costs one training run per direction.

---

## 3. Scope

### Minimum publishable core — build this first, ~3 weeks
Everything below is a complete story on its own. If time runs out, cut extensions, not this.

- One dataset pair: **HH-RLHF ↔ UltraFeedback**
- Two noise models: **position bias**, **length/verbosity bias**
- Two metrics: **pairwise accuracy**, **rank-flip rate**
- One mitigation: **ensembling**
- **Dockerized batched-inference API**

### Extensions, in priority order
1. OASST + annotator-heterogeneity noise
2. Calibration (ECE) and ensemble-disagreement uncertainty
3. Cross-dataset generalization sweep
4. Uncertainty-aware weighting
5. Noise-aware losses (symmetric CE, co-teaching) — **only if time permits**, this is a time sink

---

## 4. Noise models

Do **not** ship "random label flips" as the headline. Random flips are the weakest and least
realistic model and everyone tries them first. The contribution lives here, so these must be
concrete and defensible.

| Noise model | Definition | Parameter |
|---|---|---|
| **Position bias** | With prob. *q*, prefer whichever response was displayed first, regardless of quality | `q ∈ {0, .1, .2, .3}` |
| **Length / verbosity bias** | With prob. *q*, prefer the longer response | `q ∈ {0, .1, .2, .3}` |
| **Annotator skill heterogeneity** | Simulate *N* annotators with per-annotator accuracy drawn from a distribution; aggregate by majority vote | `N ∈ {1,3,5}`, skill ~ Beta |
| **Difficulty-correlated noise** | Error probability rises as the true reward margin between the two responses shrinks | slope *α* |
| *(baseline)* Uniform label flip | Include **only** as the trivial control to compare against | `p ∈ {0, .1, .2, .3}` |

**Implementation rule:** every operator is seeded, deterministic given `(seed, params)`, and
applied *only to the training labels* — evaluation labels stay clean. Unit-test each one:
at `q=1` the operator must collapse to its degenerate behaviour, at `q=0` it must be the
identity. (Same discipline as the IRL3 corruption operators.)

---

## 5. Metrics

| Metric | What it measures | Notes |
|---|---|---|
| **Pairwise accuracy** | Fraction of held-out preference pairs ranked correctly | The standard number |
| **Rank-flip rate** | % of prompts where the noisy model's top-ranked candidate differs from the clean model's | **The headline metric** |
| **Best-of-n win rate** | Judge-scored win rate of best-of-n selections, clean vs noisy reward model | Use a held-out judge |
| **Calibration (ECE)** | Whether predicted preference probabilities are honest | Noise may preserve accuracy but destroy calibration |
| **Uncertainty** | **Ensemble disagreement across seeds** — variance of scores | Same runs feed the ensembling mitigation. One artifact, two uses. |
| **Cross-dataset generalization** | Accuracy when trained on A, evaluated on B | Noise may barely dent in-distribution accuracy while wrecking transfer — that asymmetry is the paper |

---

## 6. Downstream protocol — best-of-n, NOT fine-tuning

The expensive version of "does this change which responses get selected" is to PPO-finetune a
policy under each reward model. **Do not do this.** It multiplies cost by an order of magnitude
and will not change the conclusion.

**The cheap version:**
1. Fix a candidate pool of *n* responses per prompt (n = 8 or 16), generated once, frozen.
2. Rank the pool with the clean reward model → record the argmax.
3. Rank the same pool with each noisy reward model → record the argmax.
4. Report **rank-flip rate** (how often the selection changes) and **win rate** of each
   selection against a held-out judge.

This isolates the reward model as the only moving part.

---

## 7. Models and compute

- **Backbone:** DeBERTa-v3-base, or a ≤1B decoder with a scalar head. The question is about the
  *labels*, not model capacity — a bigger backbone only makes every grid cell slower.
- **Loss:** Bradley-Terry pairwise preference loss.
- **Library:** HuggingFace `transformers` + `trl` (`RewardTrainer`) + PyTorch.
- **Compute:** one GPU per run, minutes-to-hours at this scale. SLURM for the sweeps.
- **Seeds:** ≥3 per cell for the core, ≥5 for anything reported as a headline number.

---

## 8. Engineering requirements

These exist because the target job description asks for them explicitly
(*"modular, robust, scalable software in Python"*, *"building and deploying machine learning
algorithms"*, *"Unix command line and bash scripting"*, *"transfer research work into
production"*).

- **Modular pipeline** — dataset loaders, noise operators, training, evaluation as separate,
  independently testable modules. No monolithic script.
- **Config-driven** — Hydra or plain YAML. No hard-coded paths, no magic numbers in code.
- **Seeded and reproducible** — every run writes a `run_meta.json` recording seed, noise params,
  dataset, model, git SHA, and achieved metrics.
- **SLURM submitters** — one `sbatch` script per sweep, appending to an experiment index CSV.
- **Experiment tracking** — Weights & Biases.
- **Dockerized batched-inference API** — FastAPI endpoint accepting a batch of
  (prompt, response) pairs, returning scores plus ensemble uncertainty. `Dockerfile` +
  `docker-compose.yml`. **This step is not optional** — it is the part that answers the
  deployment requirement.
- **README with a results table and repro commands** — a stranger should be able to reproduce
  a figure from the repo alone.

### Suggested layout
```
reward-model-robustness/
├── README.md
├── configs/           # YAML: dataset, noise, model, sweep
├── src/
│   ├── data/          # loaders for HH-RLHF, UltraFeedback, OASST
│   ├── noise/         # one module per noise operator + tests
│   ├── train/         # reward-model training (Bradley-Terry)
│   ├── eval/          # accuracy, ECE, rank-flip, best-of-n
│   └── serve/         # FastAPI app
├── scripts/           # SLURM submitters
├── experiments/       # run index CSVs
├── figures/           # all plots write here
├── tests/             # unit tests for noise operators
├── Dockerfile
└── WORKLOG.md         # running log, newest at top (same as IRL3)
```

---

## 9. Plan

| Week | Deliverable |
|---|---|
| **1** | Datasets loading; baseline reward model trained; pairwise accuracy on held-out. **Stop here and the resume entry is already true.** |
| **1–2** | Human ↔ synthetic cross-training (HH-RLHF ↔ UltraFeedback). First real result. |
| **2–3** | Position bias + length bias operators, unit-tested; accuracy-vs-noise curves |
| **3–4** | Best-of-n harness; rank-flip rate; **the headline result** |
| **4–5** | Ensembling; calibration; ensemble-disagreement uncertainty |
| **5–6** | Cross-dataset generalization sweep; annotator heterogeneity |
| **6** | FastAPI + Docker + W&B dashboard + README results table |
| **7–8** | Write-up, 4–8 pages |

**Venues:** NeurIPS / ICML workshop, or RLC. Non-archival workshop first is fine.

---

## 10. Honesty rules

- The resume entry reads **"in progress."** That becomes true the day the first baseline trains,
  and **not before**. Do not send a resume carrying this entry until then.
- An interviewer will ask what you found. Be able to talk for five minutes about it — which
  datasets, which noise model, what the curve looked like. A bullet you can defend in
  conversation is an honest bullet; one you cannot is a liability worse than an empty section.
- Do not claim "deployed to production." The API is a research artifact. "Packaged for batched
  inference behind a containerized API" is accurate and sufficient.

---

## 11. Why this project, for this application

Maps onto the RBC Borealis ML Researcher internship requirements:

| Requirement | Answered by |
|---|---|
| *"building and deploying machine learning algorithms"* (required) | The Dockerized inference API |
| *"implement state-of-the-art ML techniques"* | Reward modelling is current RLHF practice |
| *"modular, robust, scalable software in Python"* | Section 8 |
| *"Unix command line and bash scripting"* | SLURM sweeps |
| *"data processing frameworks… PyTorch"* | Whole pipeline |
| *preferred:* publications at a top-tier AI conference | Workshop-paper sized |
| *preferred:* transfer research work into production | The API |

Research areas hit: **Reinforcement Learning**, **LLMs**, **Responsible AI**,
**Explainable AI & Model Interpretability** — four of their thirteen.
