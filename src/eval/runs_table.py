"""Training runs table for the README, from experiments/index.csv -> results/runs.md."""
import csv
from pathlib import Path

import yaml


def main():
    n_req = yaml.safe_load(open("configs/rm_base.yaml"))["n_train"]
    rows = sorted(csv.DictReader(open("experiments/index.csv")), key=lambda r: (r["train_source"], int(r["seed"])))
    lines = [f"| run | train pairs after 1,024-token filter | filtered (of {n_req:,}) | optimizer steps | train time (min) |",
             "|---|---|---|---|---|"]
    for r in rows:
        n = int(r["n_train"])
        lines.append(f"| {r['run_id']} | {n:,} | {n_req - n:,} ({100 * (n_req - n) / n_req:.1f}%) | {r['steps']} | "
                     f"{float(r['train_wall_s']) / 60:.1f} |")
    Path("results/runs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
