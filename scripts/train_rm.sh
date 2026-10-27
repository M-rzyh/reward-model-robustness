#!/bin/bash
#SBATCH --job-name=rmr-train
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

cd ~/reward-model-robustness
RUN_ID="${RUN_ID:-core_${SOURCE}_q0_s${SEED}}"
OUT=/scratch/marzii/rmr/runs/$RUN_ID
echo "run_id=$RUN_ID  source=$SOURCE  seed=$SEED  extra=$EXTRA  job=$SLURM_JOB_ID  host=$(hostname)"
python -m src.train.train_rm --train_source "$SOURCE" --seed "$SEED" --out_dir "$OUT" $EXTRA