#!/bin/bash
#SBATCH --job-name=baseline-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python -u train.py \
    --model baseline \
    --device cuda \
    --epochs 30 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --workers 2 \
    --patience 8 \
    --seed 3710 \
    --image-size 224 \
    --profile-batches 20 \
    --skip-test \
    --output-dir outputs/baseline_seed3710
