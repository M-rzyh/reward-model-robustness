"""Score every pool candidate with trained reward models -> results/best_of_n/scores/<run_id>.json (500 x 8 each)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.eval.pairwise import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/pool/pool.jsonl")
    ap.add_argument("--run_dirs", nargs="+", required=True, help="run folders, each with model/ and run_meta.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--out_dir", default="results/best_of_n/scores")
    a = ap.parse_args()

    pool = [json.loads(line) for line in open(a.pool)]
    pool_sha1 = hashlib.sha1(Path(a.pool).read_bytes()).hexdigest()
    K = len(pool[0]["candidates"])
    convs = [p["prompt"] + [{"role": "assistant", "content": c["text"]}] for p in pool for c in p["candidates"]]
    longest = np.array([[c["n_tokens"] for c in p["candidates"]] for p in pool]).argmax(1)   # index of longest candidate
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rd in map(Path, a.run_dirs):
        meta = json.loads((rd / "run_meta.json").read_text())
        tok = AutoTokenizer.from_pretrained(rd / "model")
        model = AutoModelForSequenceClassification.from_pretrained(rd / "model", dtype=getattr(torch, a.dtype)).to("cuda").eval()
        model.config.pad_token_id = tok.pad_token_id
        s = score(model, tok, convs, a.batch_size, meta["config"]["max_length"]).reshape(len(pool), K)
        (out_dir / f"{rd.name}.json").write_text(json.dumps({
            "run_id": rd.name, "train_source": meta["train_source"], "seed": meta["seed"], "q": meta["noise"]["q"],
            "dtype": a.dtype, "pool_sha1": pool_sha1, "prompt_ids": [p["prompt_id"] for p in pool], "scores": s.tolist()}))
        print(f"{rd.name}: scores {s.shape}  mean={s.mean():.3f}  "
              f"picks-the-longest={(s.argmax(1) == longest).mean():.3f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()