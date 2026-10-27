"""Score held-out pairs with a trained reward model: pairwise accuracy + bootstrap CI on both test sets."""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.hh import load_hh
from src.data.uf import load_uf
from src.eval.length_stats import bootstrap_ci

TEST_SETS = {"hh_test": (load_hh, "test"), "uf_test": (load_uf, "test_prefs")}
INDEX = Path("experiments/index.csv")


@torch.no_grad()
def score(model, tok, conversations: list[list[dict]], batch_size: int, max_length: int) -> np.ndarray:
    """Reward for each full conversation (prompt + response), rendered with the chat template like TRL does."""
    out = []
    for i in range(0, len(conversations), batch_size):
        batch = tok.apply_chat_template(conversations[i:i + batch_size], tokenize=True, return_dict=True,
                                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out.append(model(**batch).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="folder with model/ and run_meta.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs per test set")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32 = precision check; bf16 rounds large rewards enough to create exact ties")
    p.add_argument("--out_dir", default="results/eval")
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    run_id = run_dir.name
    max_length = meta["config"]["max_length"]

    tok = AutoTokenizer.from_pretrained(run_dir / "model")
    model = AutoModelForSequenceClassification.from_pretrained(run_dir / "model", dtype=getattr(torch, a.dtype)).to("cuda").eval()
    model.config.pad_token_id = tok.pad_token_id

    results, scores = {}, {}
    for name, (loader, split) in TEST_SETS.items():
        records, _ = loader(split)
        records = records[: a.limit] if a.limit else records
        r_c = score(model, tok, [r["prompt"] + r["chosen"] for r in records], a.batch_size, max_length)
        r_r = score(model, tok, [r["prompt"] + r["rejected"] for r in records], a.batch_size, max_length)
        correct = (r_c > r_r).astype(float)                       # ties count as wrong
        results[name] = {
            "n_pairs": int(len(records)),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": bootstrap_ci(correct),
            "n_ties": int((r_c == r_r).sum()),
            "mean_margin": float((r_c - r_r).mean()),
        }
        scores[name] = {"pair_id": [r["pair_id"] for r in records], "r_chosen": r_c.tolist(), "r_rejected": r_r.tolist()}
        print(f"{run_id}  {name}: acc={results[name]['accuracy']:.4f} CI={results[name]['accuracy_ci95']}  n={len(records)}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "run_meta": meta, "dtype": a.dtype, "eval": results}, indent=2))
    (out_dir / f"{run_id}_scores.json").write_text(json.dumps(scores))

    if not a.limit and a.dtype == "bfloat16":                     # smoke and precision-check runs stay out of the index
        row = {"run_id": run_id, "train_source": meta["train_source"], "seed": meta["seed"], "q": meta["noise"]["q"],
               "acc_hh_test": results["hh_test"]["accuracy"], "acc_uf_test": results["uf_test"]["accuracy"],
               "n_train": meta["n_train_after_length_filter"], "steps": meta["n_optimizer_steps"],
               "train_wall_s": meta["wall_time_s"], "git_sha": meta["git_sha"], "date": date.today().isoformat()}
        new = not INDEX.exists()
        INDEX.parent.mkdir(exist_ok=True)
        with INDEX.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row)
            if new:
                w.writeheader()
            w.writerow(row)
    print("wrote", out_dir / f"{run_id}.json")


if __name__ == "__main__":
    main()