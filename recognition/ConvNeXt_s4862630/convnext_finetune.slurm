#!/bin/bash
#SBATCH --job-name=convnext-finetune
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"

mode="${1:-smoke}"
case "$mode" in
  smoke)
    python -u train_convnext_finetune.py \
      --epochs 1 --head-epochs 0 --batch-size 8 --workers 2 \
      --max-train-batches 5 --max-eval-batches 3 --profile-batches 3 \
      --output-dir "outputs/smoke_convnext_pretrained_${SLURM_JOB_ID}"
    ;;
  train)
    python -u train_convnext_finetune.py \
      --epochs 30 --head-epochs 2 --batch-size 32 --workers 2 \
      --learning-rate 0.00005 --head-learning-rate 0.00025 \
      --weight-decay 0.01 --label-smoothing 0.05 --patience 8 \
      --seed 3710 --device cuda \
      --output-dir "outputs/convnext_pretrained_seed3710_${SLURM_JOB_ID}"
    ;;
  *) echo 'Usage: sbatch convnext_finetune.slurm [smoke|train]' >&2; exit 2 ;;
esac
