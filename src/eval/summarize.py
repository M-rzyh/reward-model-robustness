"""Aggregate per-run eval JSONs into the 2x2 table: mean ± std over seeds, with the length-only baseline row."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_DIR = Path("results/eval")
TEST_SETS = ["hh_test", "uf_test"]


def main():
    runs = defaultdict(list)                      # (train_source, q) -> one entry per seed
    for f in sorted(EVAL_DIR.glob("*.json")):
        if f.name.endswith("_scores.json") or f.name.startswith("smoke"):
            continue
        d = json.loads(f.read_text())
        key = (d["run_meta"]["train_source"], d["run_meta"]["noise"]["q"])
        runs[key].append({"seed": d["run_meta"]["seed"], **{t: d["eval"][t]["accuracy"] for t in TEST_SETS}})

    base = json.loads(Path("results/length_baseline.json").read_text())["splits"]
    summary = {"length_only": {t: base[t]["length_only_accuracy"] for t in TEST_SETS}, "configs": {}}
    lines = ["| trained on | seeds | hh_test (human labels) | uf_test (GPT-4 labels) |", "|---|---|---|---|",
             f"| length-only baseline | – | {base['hh_test']['length_only_accuracy']:.3f} | {base['uf_test']['length_only_accuracy']:.3f} |"]

    def fmt(c):
        return f"{c['mean']:.3f} ± {c['std']:.3f}" if c["std"] is not None else f"{c['mean']:.3f} (1 seed)"

    for (src, q), rs in sorted(runs.items()):
        name = f"{src} (q={q:g})"
        cells = {}
        for t in TEST_SETS:
            accs = np.array([r[t] for r in rs])
            cells[t] = {"mean": float(accs.mean()),
                        "std": float(accs.std(ddof=1)) if len(accs) > 1 else None,   # sample std over seeds
                        "per_seed": {str(r["seed"]): r[t] for r in sorted(rs, key=lambda r: r["seed"])}}
        summary["configs"][name] = {"n_seeds": len(rs), **cells}
        lines.append(f"| RM trained on {name} | {len(rs)} | {fmt(cells['hh_test'])} | {fmt(cells['uf_test'])} |")

    table = "\n".join(lines)
    print(table)
    Path("results/summary_2x2.json").write_text(json.dumps(summary, indent=2))
    Path("results/summary_2x2.md").write_text(table + "\n")
    print("wrote results/summary_2x2.{json,md}")


if __name__ == "__main__":
    main()