"""Generate the frozen best-of-n pool: N deduplicated HH test prompts x K sampled responses from an instruct model."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.hh import load_hh


def pick_prompts(records: list[dict], tok, n: int, max_prompt_tokens: int, seed: int) -> tuple[list[dict], dict]:
    """Deduplicate prompts, drop very long ones, then draw n with a seeded permutation (order-independent)."""
    seen, uniq = set(), []
    for r in sorted(records, key=lambda r: r["pair_id"]):
        key = json.dumps(r["prompt"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    lengths = [len(tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True)) for r in uniq]
    short = [r for r, L in zip(uniq, lengths) if L <= max_prompt_tokens]
    idx = np.random.default_rng(seed).permutation(len(short))[:n]
    stats = {"n_test_records": len(records), "n_unique_prompts": len(uniq), "n_within_max_prompt_tokens": len(short)}
    return [short[i] for i in sorted(idx)], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pool.yaml")
    ap.add_argument("--out", default="results/pool/pool.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only this many prompts")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["generator_id"], revision=cfg["generator_revision"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["generator_id"], revision=cfg["generator_revision"], dtype=torch.bfloat16).to("cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    records, _ = load_hh("test")
    prompts, stats = pick_prompts(records, tok, a.limit or cfg["n_prompts"], cfg["max_prompt_tokens"], cfg["seed"])
    torch.manual_seed(cfg["seed"])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    K, n_tokens, n_trunc = cfg["n_per_prompt"], [], 0
    with out.open("w") as f, torch.no_grad():
        for i in range(0, len(prompts), cfg["batch_prompts"]):
            batch = prompts[i:i + cfg["batch_prompts"]]
            enc = tok.apply_chat_template([p["prompt"] for p in batch], add_generation_prompt=True, tokenize=True,
                                          return_dict=True, padding=True, return_tensors="pt").to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"],
                                 top_k=cfg["top_k"], repetition_penalty=cfg["repetition_penalty"],
                                 max_new_tokens=cfg["max_new_tokens"], num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]                      # only the generated part
            for j, p in enumerate(batch):
                cands = []
                for k in range(K):
                    ids = new[j * K + k]
                    finished = bool(((ids == im_end) | (ids == tok.pad_token_id)).any())   # hit an end token?
                    text = tok.decode(ids, skip_special_tokens=True).strip()
                    n = len(tok(text, add_special_tokens=False)["input_ids"])
                    cands.append({"text": text, "n_tokens": n, "truncated": not finished})
                    n_tokens.append(n)
                    n_trunc += int(not finished)
                f.write(json.dumps({"prompt_id": p["pair_id"], "prompt": p["prompt"], "candidates": cands}) + "\n")
            print(f"{min(i + len(batch), len(prompts))}/{len(prompts)} prompts  {time.time() - t0:.0f}s", flush=True)

    meta = {
        "config": cfg, "n_prompts": len(prompts), "n_per_prompt": K, **stats,
        "pool_sha1": hashlib.sha1(out.read_bytes()).hexdigest(),
        "mean_n_tokens": float(np.mean(n_tokens)), "frac_truncated": n_trunc / len(n_tokens),
        "wall_time_s": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "backend": "transformers.generate",
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()