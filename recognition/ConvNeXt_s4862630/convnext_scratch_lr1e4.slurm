#!/bin/bash
#SBATCH --job-name=convnext-scratch-lr1e4
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Controlled comparison with convnext_train.slurm: change only the initial LR.
# train.py constructs random weights; no checkpoint or pretrained weights load.
# Training uses validation only. Do not run evaluate_models.slurm automatically:
# its fixed paths refer to the original experiments.
set -euo pipefail

source "$HOME/comp3710/venvs/convnext-report/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit this job from the ConvNeXt project directory}"

output_dir="outputs/convnext_scratch_lr1e4_seed3710_${SLURM_JOB_ID:?Use sbatch}"
mkdir -p outputs
# Atomic reservation prevents accidental reuse of an existing run directory.
mkdir "$output_dir"

python -u train.py \
    --model convnext \
    --device cuda \
    --epochs 30 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --weight-decay 0.05 \
    --workers 2 \
    --patience 8 \
    --seed 3710 \
    --image-size 224 \
    --profile-batches 20 \
    --skip-test \
    --output-dir "$output_dir"

echo "Completed validation-only scratch experiment: $output_dir"
echo "Compare history.json and training_curves.png with outputs/convnext_seed3710."
