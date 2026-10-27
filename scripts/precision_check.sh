#!/bin/bash
#SBATCH --job-name=rmr-fp32
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
# Re-score the held-out pairs and the best-of-8 pool in float32 (bf16 rounds large rewards into exact ties).
# Writes results/eval_fp32/ and results/best_of_n/scores_fp32/; the bf16 results are left untouched.

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd ~/reward-model-robustness
RUN_DIRS="${RUN_DIRS:-$(ls -d /scratch/marzii/rmr/runs/core_*_q0_s*)}"
echo "fp32 precision check job=$SLURM_JOB_ID host=$(hostname)"; echo "$RUN_DIRS"
set -e
python -m src.eval.score_pool --dtype float32 --out_dir results/best_of_n/scores_fp32 --run_dirs $RUN_DIRS
for rd in $RUN_DIRS; do
  python -m src.eval.pairwise --dtype float32 --out_dir results/eval_fp32 --run_dir "$rd"
done
