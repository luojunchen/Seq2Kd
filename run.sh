#!/bin/bash
# Seq2Kd Inference Script
# Run binding affinity prediction

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o logs/infer-%j.out
#SBATCH -J seq2kd_infer

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
# Activate conda environment (update this path for your environment)
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate seq2kd

# Timer start
echo "[TIMER] Job ${SLURM_JOB_ID:-N/A} starting at $(date -Is)"
_t_start=$(date +%s)

# Run inference
python src/inference.py \
  --model-path ./checkpoints/seq2kd_basic.pt \
  --config-path ./cfg/cfg01.yaml \
  --input-path ./_inputs/example_input.txt \
  --output-dir ./_outputs

# Timer end
_t_end=$(date +%s)
_t_elapsed=$((_t_end - _t_start))
printf "[TIMER] Job ${SLURM_JOB_ID:-N/A} finished at %s\n" "$(date -Is)"
printf "[TIMER] Elapsed: %ds (%02d:%02d:%02d)\n" \
  "$_t_elapsed" "$((_t_elapsed/3600))" "$((_t_elapsed%3600/60))" "$((_t_elapsed%60))"
