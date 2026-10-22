"""HH-RLHF (helpful-base) loader: raw transcripts -> common preference records."""
import re
from collections import Counter

from datasets import load_from_disk

from src.data.schema import DATA_DIR, make_pair_id, valid_turns

ASSISTANT = "\n\nAssistant:"
TURN_RE = re.compile(r"\n\n(Human|Assistant):")
ROLE = {"Human": "user", "Assistant": "assistant"}


def split_transcript(text: str) -> tuple[str, str]:
    """Split a raw HH transcript at its LAST assistant turn -> (prompt_text, response)."""
    idx = text.rfind(ASSISTANT)
    if idx == -1:
        raise ValueError("transcript has no assistant turn")
    prompt_text = text[:idx]
    response = text[idx + len(ASSISTANT):].lstrip()
    return prompt_text, response


def parse_turns(prompt_text: str) -> list[dict]:
    """Turn prompt text into chat messages: [{"role": "user"|"assistant", "content": str}, ...]."""
    parts = TURN_RE.split(prompt_text)          # ['', 'Human', ' hi', 'Assistant', ' hello', 'Human', ' bye']
    if parts[0].strip():
        raise ValueError(f"text before the first turn marker: {parts[0][:40]!r}")
    messages = []
    for role, content in zip(parts[1::2], parts[2::2]):
        messages.append({"role": ROLE[role], "content": content.strip()})
    return messages

def load_hh(split: str) -> tuple[list[dict], dict]:
    """Return (records, drop_counts) for one split of HH helpful-base.

    Drop reasons: prefix_mismatch (chosen/rejected share no identical prompt),
    empty_response, identical (chosen == rejected), bad_turns (roles don't alternate).
    """
    ds = load_from_disk(str(DATA_DIR / "hh_helpful_base"))[split]
    records, dropped = [], Counter()
    for row in ds:
        prompt_c, chosen = split_transcript(row["chosen"])
        prompt_r, rejected = split_transcript(row["rejected"])
        if prompt_c != prompt_r:
            dropped["prefix_mismatch"] += 1
            continue
        if not chosen or not rejected:
            dropped["empty_response"] += 1
            continue
        if chosen == rejected:
            dropped["identical"] += 1
            continue
        try:
            prompt = parse_turns(prompt_c)
        except ValueError:
            prompt = []
        if not valid_turns(prompt):
            dropped["bad_turns"] += 1
            continue
        records.append({
            "pair_id": make_pair_id("hh", prompt, chosen, rejected),
            "source": "hh",
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return records, dict(dropped)