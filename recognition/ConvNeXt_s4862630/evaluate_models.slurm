#!/bin/bash
#SBATCH --job-name=adni-evaluate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# Default: calibration/validation only. Final testing: sbatch this_file test.
mode="${1:-calibrate}"
if [[ "$mode" != calibrate && "$mode" != test ]]; then
    echo "Usage: sbatch evaluate_models.slurm [calibrate|test]" >&2
    exit 2
fi

for model in baseline convnext; do
    python -u evaluate_models.py \
        --mode "$mode" \
        --checkpoint "outputs/${model}_seed3710/best_model.pth" \
        --output-dir "outputs/evaluation_${model}_seed3710" \
        --batch-size 32 \
        --workers 2 \
        --device cuda
done
