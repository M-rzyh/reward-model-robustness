"""Train one Bradley-Terry reward model (TRL RewardTrainer) on HH or UF and write run_meta.json."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import trl
import yaml
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.data.hh import load_hh
from src.data.uf import load_uf

LOADERS = {"hh": (load_hh, "train"), "uf": (load_uf, "train_prefs")}


def subsample(records: list[dict], n: int, seed: int) -> tuple[list[dict], str]:
    """Seeded, order-independent subsample. Returns (records, sha1 of the sorted pair_ids)."""
    records = sorted(records, key=lambda r: r["pair_id"])          # fixed order before drawing
    idx = np.random.default_rng(seed).permutation(len(records))[:n]
    chosen = [records[i] for i in sorted(idx)]
    digest = hashlib.sha1(",".join(r["pair_id"] for r in chosen).encode()).hexdigest()[:16]
    return chosen, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rm_base.yaml")
    p.add_argument("--train_source", choices=LOADERS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="smoke test: use only this many pairs")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    loader, split = LOADERS[a.train_source]
    records, dropped = loader(split)
    n_train = a.limit or cfg["n_train"]
    records, subsample_hash = subsample(records, n_train, a.seed)
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in records])

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], num_labels=1)

    args = RewardConfig(
        output_dir=str(out / "trainer"),
        seed=a.seed,
        max_length=cfg["max_length"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg["warmup_steps"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_strategy="no",          # we save the final model ourselves below
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = RewardTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    n_after_filter = len(trainer.train_dataset)       # pairs kept after the max_length filter
    trainer.train()

    trainer.save_model(str(out / "model"))
    tok.save_pretrained(str(out / "model"))

    step_logs = [l for l in trainer.state.log_history if "loss" in l]          # every logging_steps
    summary = next((l for l in trainer.state.log_history if "train_loss" in l), {})   # end-of-training entry
    meta = {
        "train_source": a.train_source, "seed": a.seed, "noise": {"kind": "none", "q": 0.0},
        "n_train_requested": n_train, "n_train_after_length_filter": n_after_filter,
        "n_filtered_by_max_length": n_train - n_after_filter, "subsample_hash": subsample_hash,
        "loader_dropped": dropped, "config": cfg,
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__},
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "wall_time_s": round(time.time() - t0, 1),
        "final_train_loss": step_logs[-1]["loss"] if step_logs else summary.get("train_loss"),
        "final_train_accuracy": step_logs[-1].get("accuracy") if step_logs else summary.get("accuracy"),
        "n_optimizer_steps": trainer.state.global_step,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()