#!/bin/bash
#SBATCH --job-name=rmr-cpu
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

module --force purge
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/rmr
unset PYTHONPATH
export HF_HOME=/scratch/marzii/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/reward-model-robustness
echo "running: $*"
"$@"