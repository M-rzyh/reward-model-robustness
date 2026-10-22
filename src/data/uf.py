"""UltraFeedback-binarized loader: message lists -> common preference records."""
from collections import Counter

from datasets import load_from_disk

from src.data.schema import DATA_DIR, make_pair_id, valid_turns


def _normalise(msgs: list[dict]) -> list[dict]:
    """Keep only role/content, in that order, with stripped content (same shape as the HH loader)."""
    return [{"role": m["role"], "content": m["content"].strip()} for m in msgs]


def load_uf(split: str) -> tuple[list[dict], dict]:
    """Return (records, drop_counts) for 'train_prefs' or 'test_prefs'.

    Drop reasons: bad_format (a side doesn't end with an assistant message), then the same
    four as HH: prefix_mismatch, empty_response, identical, bad_turns.
    """
    ds = load_from_disk(str(DATA_DIR / "uf_prefs"))[split]
    records, dropped = [], Counter()
    for row in ds:
        c, r = _normalise(row["chosen"]), _normalise(row["rejected"])
        if not c or not r or c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant":
            dropped["bad_format"] += 1
            continue
        prompt_c, chosen = c[:-1], c[-1]["content"]
        prompt_r, rejected = r[:-1], r[-1]["content"]
        if prompt_c != prompt_r:
            dropped["prefix_mismatch"] += 1
            continue
        if not chosen or not rejected:
            dropped["empty_response"] += 1
            continue
        if chosen == rejected:
            dropped["identical"] += 1
            continue
        if not valid_turns(prompt_c):
            dropped["bad_turns"] += 1
            continue
        records.append({
            "pair_id": make_pair_id("uf", prompt_c, chosen, rejected),
            "source": "uf",
            "prompt": prompt_c,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return records, dict(dropped)