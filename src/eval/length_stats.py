"""Length statistics and the length-only baseline ("pick the longer response"). Spec section 3.6."""
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.data.hh import load_hh
from src.data.uf import load_uf

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = Path("results/length_baseline.json")
SPLITS = [("hh_test", load_hh, "test"), ("uf_test", load_uf, "test_prefs"),
          ("hh_train", load_hh, "train"), ("uf_train", load_uf, "train_prefs")]


def token_lengths(tok, texts: list[str], batch: int = 1000) -> np.ndarray:
    """Response length in tokens, no chat template, no special tokens."""
    out = []
    for i in range(0, len(texts), batch):
        out.extend(len(ids) for ids in tok(texts[i:i + batch], add_special_tokens=False)["input_ids"])
    return np.array(out)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% CI of the mean by resampling pairs with replacement."""
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def length_stats(tok, records: list[dict]) -> dict:
    lc = token_lengths(tok, [r["chosen"][0]["content"] for r in records])
    lr = token_lengths(tok, [r["rejected"][0]["content"] for r in records])
    credit = np.where(lc > lr, 1.0, np.where(lc == lr, 0.5, 0.0))   # ties get half credit
    return {
        "n_pairs": int(len(records)),
        "mean_len_chosen": float(lc.mean()), "median_len_chosen": float(np.median(lc)),
        "mean_len_rejected": float(lr.mean()), "median_len_rejected": float(np.median(lr)),
        "frac_chosen_longer": float((lc > lr).mean()),
        "frac_ties": float((lc == lr).mean()),
        "length_only_accuracy": float(credit.mean()),
        "accuracy_ci95": bootstrap_ci(credit),
    }


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    results = {"tokenizer": TOKENIZER, "splits": {}}
    for name, loader, split in SPLITS:
        records, _ = loader(split)
        s = results["splits"][name] = length_stats(tok, records)
        print(f"{name:8s} n={s['n_pairs']:6d}  mean len chosen/rejected={s['mean_len_chosen']:.0f}/{s['mean_len_rejected']:.0f}"
              f"  chosen longer={s['frac_chosen_longer']:.3f}  ties={s['frac_ties']:.3f}"
              f"  length-only acc={s['length_only_accuracy']:.3f} CI=[{s['accuracy_ci95'][0]:.3f}, {s['accuracy_ci95'][1]:.3f}]")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()