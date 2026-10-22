"""Common preference-record schema shared by the HH and UF loaders."""
import hashlib
import json
from pathlib import Path

DATA_DIR = Path("/scratch/marzii/rmr/data")


def make_pair_id(source: str, prompt: list[dict], chosen_text: str, rejected_text: str) -> str:
    """Deterministic 16-hex id: same pair -> same id, on any machine, in any run."""
    key = source + json.dumps(prompt, sort_keys=True) + chosen_text + rejected_text
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def valid_turns(msgs: list[dict]) -> bool:
    """Roles must alternate user/assistant, starting and ending with user."""
    roles = [m["role"] for m in msgs]
    expected = (["user", "assistant"] * len(roles))[: len(roles)]
    return bool(roles) and roles == expected and roles[-1] == "user"