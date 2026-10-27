#!/bin/bash
#SBATCH --job-name=rmr-pool
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd ~/reward-model-robustness
echo "pool job=$SLURM_JOB_ID host=$(hostname) extra=$EXTRA"
python -m src.gen.make_pool $EXTRA