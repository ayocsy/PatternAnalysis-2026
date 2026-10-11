#!/bin/bash
#SBATCH --job-name=adni-predict
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# This preselected case is incorrectly classified by baseline and correctly by
# ConvNeXt. It demonstrates model differences, not overall superiority.
patient="${1:-168_S_6086}"
for model in baseline convnext; do
    python -u predict.py \
        --checkpoint "outputs/${model}_seed3710/best_model.pth" \
        --calibration "outputs/evaluation_${model}_seed3710/calibration.json" \
        --patient-id "$patient" \
        --split test \
        --batch-size 32 \
        --workers 2 \
        --device cuda \
        --output-dir "outputs/demo_${model}_${patient}_${SLURM_JOB_ID}"
done
