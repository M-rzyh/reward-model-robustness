#!/bin/bash
#SBATCH --job-name=rmr-scorepool
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd ~/reward-model-robustness
RUN_DIRS="${RUN_DIRS:-$(ls -d /scratch/marzii/rmr/runs/core_*_q0_s*)}"
echo "score-pool job=$SLURM_JOB_ID host=$(hostname)"; echo "$RUN_DIRS"
python -m src.eval.score_pool --run_dirs $RUN_DIRS