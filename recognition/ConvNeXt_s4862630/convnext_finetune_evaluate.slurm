#!/bin/bash
#SBATCH --job-name=convnext-ft-evaluate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"
training_job="${1:?Usage: sbatch convnext_finetune_evaluate.slurm TRAIN_JOB_ID [calibrate|test]}"
mode="${2:-calibrate}"
[[ "$training_job" =~ ^[0-9]+$ ]] || { echo 'Invalid training job ID' >&2; exit 2; }
[[ "$mode" == calibrate || "$mode" == test ]] || { echo 'Mode must be calibrate or test' >&2; exit 2; }
python -u evaluate_models.py \
  --mode "$mode" \
  --checkpoint "outputs/convnext_pretrained_seed3710_${training_job}/best_model.pth" \
  --output-dir "outputs/evaluation_convnext_pretrained_seed3710_${training_job}" \
  --batch-size 32 --workers 2 --device cuda
