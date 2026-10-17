"""Download HH-RLHF (helpful-base) and UltraFeedback-binarized once; save them for offline use."""
from pathlib import Path
from datasets import DatasetDict, load_dataset

OUT = Path("/scratch/marzii/rmr/data")

hh = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")      # splits: train, test
hh.save_to_disk(str(OUT / "hh_helpful_base"))

uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized")           # 6 splits; we keep 2
uf = DatasetDict({s: uf[s] for s in ["train_prefs", "test_prefs"]})
uf.save_to_disk(str(OUT / "uf_prefs"))

for name, d in [("hh_helpful_base", hh), ("uf_prefs", uf)]:
    print(name, {k: len(v) for k, v in d.items()})
