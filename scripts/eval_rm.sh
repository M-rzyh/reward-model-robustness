#!/bin/bash
#SBATCH --job-name=rmr-eval
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd ~/reward-model-robustness
echo "eval run_id=$RUN_ID job=$SLURM_JOB_ID host=$(hostname)"
python -m src.eval.pairwise --run_dir /scratch/marzii/rmr/runs/$RUN_ID $EXTRA