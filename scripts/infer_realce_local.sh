#!/usr/bin/env bash
set -euo pipefail

# Real-CE val inference for a local checkpoint.
#
# The 13mm LQ is already registered to the 52mm GT size, so upscale=1 and the
# large images go through tiled latent inference (tile 512 = training res).
#
# Usage:
#   EXP=<exp_dir> STEP=00020000 GPU=0 TAG=2a_local bash scripts/infer_realce_local.sh

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${EXP:?set EXP=<exp dir under exp_vosr_*>}"
STEP="${STEP:-00020000}"
GPU="${GPU:-0}"
TAG="${TAG:?set TAG=<short name for output dir>}"
INPUT="${INPUT:-/data/ywk/datasets/Real-CE/val/13mm}"
OUT_ROOT="${OUT_ROOT:-preset/results/realce}"

mkdir -p "${OUT_ROOT}/_logs"
CUDA_VISIBLE_DEVICES="${GPU}" python inference_vosr_onestep.py \
  -c "${EXP}/checkpoints/checkpoint-${STEP}" \
  -i "${INPUT}" \
  -o "${OUT_ROOT}/${TAG}_step${STEP}" \
  -u 1 --tile_size 512 --tile_overlap 64 \
  --infer_steps 1 --align_method nofix --force_rerun \
  2>&1 | tee "${OUT_ROOT}/_logs/${TAG}_step${STEP}.log"
