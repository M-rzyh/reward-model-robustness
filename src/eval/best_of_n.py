"""Best-of-8 analysis on the frozen pool: flip rate, Kendall tau and selected length, within vs across configs.

Inputs: results/best_of_n/scores/*.json (one 500 x 8 matrix per model) and results/pool/pool.jsonl (lengths).
Outputs: results/best_of_n/summary.{json,md}.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from src.eval.length_stats import bootstrap_ci



def winners(scores):
    """Argmax per prompt; np.argmax already breaks ties toward the lowest index."""
    return np.asarray(scores).argmax(axis=1)


def n_ties(scores):
    """Number of prompts where the top score is shared by more than one candidate."""
    s = np.asarray(scores)
    return int(((s == s.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())


def flip_rate(a, b):
    """Fraction of prompts where the two models pick a different winner."""
    return float((winners(a) != winners(b)).mean())


def mean_kendall_tau(a, b):
    """Kendall tau-b between the two score vectors of each prompt, averaged over prompts.

    tau is undefined (nan) when one model gives all candidates the same score; those prompts are
    skipped and counted so the caller can see it."""
    taus = np.array([kendalltau(x, y).statistic for x, y in zip(np.asarray(a), np.asarray(b))])
    return float(np.nanmean(taus)), int(np.isnan(taus).sum())


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if len(xs) > 1 else None}   # sample std


def load_models(score_dir, pool_sha1):
    models = {}
    for f in sorted(score_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d["pool_sha1"] != pool_sha1:
            raise ValueError(f"{f} was scored on a different pool ({d['pool_sha1'][:8]} != {pool_sha1[:8]})")
        models[d["run_id"]] = d
    return models


def pairings(models):
    """within = seed pairs of the same train_source; cross = every hh seed x every uf seed."""
    by_src = {}
    for rid, d in models.items():
        by_src.setdefault(d["train_source"], []).append(rid)
    groups = {f"within_{src}": list(itertools.combinations(sorted(rids), 2)) for src, rids in sorted(by_src.items())}
    groups["cross_hh_uf"] = list(itertools.product(sorted(by_src["hh"]), sorted(by_src["uf"])))
    return groups


def flip_gap(models, groups, cross, within):
    """Per-prompt flip indicator averaged over the pairings of each group; returns cross minus within per prompt.

    The mean of the returned vector is exactly mean(cross flip) - mean(within flip). Bootstrapping over prompts
    gives an honest CI for that gap; the std over pairings does not, because pairings share models."""
    def per_prompt(names):
        pairs = [p for g in names for p in groups[g]]
        return np.mean([winners(models[x]["scores"]) != winners(models[y]["scores"]) for x, y in pairs], axis=0)
    return per_prompt(cross) - per_prompt(within)


def disagreements(pool, models, a, b, n):
    """Markdown for the first n prompts where models a and b pick different winners, for a by-eye read."""
    wa, wb = winners(models[a]["scores"]), winners(models[b]["scores"])
    out = [f"# First {n} prompts where {a} and {b} pick different winners\n"]
    for i in np.flatnonzero(wa != wb)[:n]:
        p = pool[i]
        out.append(f"## prompt {p['prompt_id']}\n\n**user (last turn):** {p['prompt'][-1]['content']}\n")
        for rid, w in ((a, wa[i]), (b, wb[i])):
            c = p["candidates"][w]
            out.append(f"**{rid} picks #{w} ({c['n_tokens']} tokens):**\n\n{c['text']}\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/pool/pool.jsonl")
    ap.add_argument("--score_dir", default="results/best_of_n/scores")
    ap.add_argument("--out_dir", default="results/best_of_n")
    ap.add_argument("--show", type=int, default=5, help="disagreements to print between HH seed 0 and UF seed 0")
    a = ap.parse_args()

    pool_sha1 = hashlib.sha1(Path(a.pool).read_bytes()).hexdigest()
    pool = [json.loads(line) for line in open(a.pool)]
    lengths = np.array([[c["n_tokens"] for c in p["candidates"]] for p in pool])
    models = load_models(Path(a.score_dir), pool_sha1)
    prompt_ids = [p["prompt_id"] for p in pool]
    for rid, d in models.items():
        assert d["prompt_ids"] == prompt_ids, f"{rid}: prompt order differs from the pool"
    rows = np.arange(len(pool))

    per_model = {}
    for rid, d in sorted(models.items()):
        w = winners(d["scores"])
        per_model[rid] = {"train_source": d["train_source"], "seed": d["seed"],
                          "selected_len": float(lengths[rows, w].mean()),
                          "picks_longest": float((w == lengths.argmax(1)).mean()),
                          "n_ties": n_ties(d["scores"])}

    per_config = {}
    for src in sorted({m["train_source"] for m in per_model.values()}):
        ms = [m for m in per_model.values() if m["train_source"] == src]
        per_config[src] = {"n_seeds": len(ms),
                           "selected_len": mean_std([m["selected_len"] for m in ms]),
                           "picks_longest": mean_std([m["picks_longest"] for m in ms])}

    groups = {}
    for g, pairs in pairings(models).items():
        each = []
        for x, y in pairs:
            tau, n_nan = mean_kendall_tau(models[x]["scores"], models[y]["scores"])
            each.append({"a": x, "b": y, "flip_rate": flip_rate(models[x]["scores"], models[y]["scores"]),
                         "kendall_tau": tau, "tau_nan_prompts": n_nan})
        groups[g] = {"n_pairs": len(each), "flip_rate": mean_std([e["flip_rate"] for e in each]),
                     "kendall_tau": mean_std([e["kendall_tau"] for e in each]), "pairs": each}

    pairs_by_group = pairings(models)
    gaps = {}
    for name, within in (("cross_minus_within_hh", ["within_hh"]), ("cross_minus_within_uf", ["within_uf"]),
                         ("cross_minus_within_both", ["within_hh", "within_uf"])):
        d = flip_gap(models, pairs_by_group, ["cross_hh_uf"], within)
        gaps[name] = {"mean": float(d.mean()), "ci95_bootstrap_prompts": bootstrap_ci(d)}

    K = lengths.shape[1]
    summary = {"score_dir": a.score_dir, "pool_sha1": pool_sha1, "n_prompts": len(pool), "n_candidates": K,
               "reference": {"pool_mean_len": float(lengths.mean()), "longest_mean_len": float(lengths.max(1).mean()),
                             "picks_longest_chance": 1 / K, "flip_rate_two_random_pickers": 1 - 1 / K},
               "per_model": per_model, "per_config": per_config, "pairings": groups, "flip_gap": gaps}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def fmt(c, pct=False):
        k = 100 if pct else 1
        return f"{k * c['mean']:.1f} ± {k * c['std']:.1f}" if pct else f"{c['mean']:.3f} ± {c['std']:.3f}"

    ref = summary["reference"]
    lines = [f"Best-of-{lengths.shape[1]} on {len(pool)} HH-test prompts (pool sha1 {pool_sha1[:8]}); mean ± sample std over pairings / seeds.",
             "", "| pairing | pairs | flip rate (%) | Kendall τ |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['n_pairs']} | {fmt(v['flip_rate'], pct=True)} | {fmt(v['kendall_tau'])} |" for g, v in groups.items()]
    lines += [f"| two independent random pickers | – | {100 * (1 - 1 / K):.1f} | 0 |", "",
              "| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |", "|---|---|---|"]
    lines += [f"| {g.replace('_', ' ')} | {100 * v['mean']:.1f} | [{100 * v['ci95_bootstrap_prompts'][0]:.1f}, "
              f"{100 * v['ci95_bootstrap_prompts'][1]:.1f}] |" for g, v in gaps.items()]
    lines += ["", "| reward model | seeds | selected length (tokens) | picks the longest (%) |", "|---|---|---|---|"]
    lines += [f"| trained on {src} | {c['n_seeds']} | {c['selected_len']['mean']:.1f} ± {c['selected_len']['std']:.1f} | "
              f"{fmt(c['picks_longest'], pct=True)} |" for src, c in per_config.items()]
    lines += [f"| pool mean (all {lengths.size}) | – | {ref['pool_mean_len']:.1f} | – |",
              f"| always pick the longest | – | {ref['longest_mean_len']:.1f} | 100 |",
              f"| random pick (chance) | – | {ref['pool_mean_len']:.1f} | {100 * ref['picks_longest_chance']:.1f} |"]
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    print(table)
    print("\nties per model:", {r: m["n_ties"] for r, m in per_model.items()})
    print("tau nan prompts:", {f"{e['a']}|{e['b']}": e["tau_nan_prompts"] for v in groups.values() for e in v["pairs"] if e["tau_nan_prompts"]} or "none")

    if a.show and "core_hh_q0_s0" in models and "core_uf_q0_s0" in models:
        (out_dir / "disagreements_hh_s0_vs_uf_s0.md").write_text(
            disagreements(pool, models, "core_hh_q0_s0", "core_uf_q0_s0", a.show))
        print("wrote", out_dir / "disagreements_hh_s0_vs_uf_s0.md")


if __name__ == "__main__":
    main()
