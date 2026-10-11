#!/bin/bash
#SBATCH --job-name=convnext-smoke
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python train.py \
    --model convnext \
    --epochs 1 \
    --batch-size 8 \
    --workers 2 \
    --max-train-batches 5 \
    --max-eval-batches 3 \
    --profile-batches 3 \
    --output-dir outputs/smoke_convnext
