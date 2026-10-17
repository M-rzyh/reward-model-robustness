#!/bin/bash
#SBATCH --job-name=rmr-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
hostname
nvidia-smi

#!/bin/bash
#SBATCH --job-name=rmr-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
hostname
nvidia-smi

python - <<'EOF'
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. GPU
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

# 2. Data (offline, from the folders saved by prefetch_data.py)
for name in ["hh_helpful_base", "uf_prefs"]:
    d = load_from_disk(f"/scratch/marzii/rmr/data/{name}")
    print(name, {k: len(v) for k, v in d.items()})

# 3. Model with a scalar head; tell the model which token is padding
M = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M, num_labels=1, dtype=torch.bfloat16).to("cuda")
model.config.pad_token_id = tok.pad_token_id

# 4. Two texts of DIFFERENT length, so padding is actually exercised
batch = tok(["Hi.", "This second response is clearly much longer than the first one."],
            padding=True, return_tensors="pt").to("cuda")

# 5. Forward pass: one scalar score per text
with torch.no_grad():
    logits = model(**batch).logits
print("logits:", logits.float().cpu().tolist(), "| shape:", tuple(logits.shape))
EOF
