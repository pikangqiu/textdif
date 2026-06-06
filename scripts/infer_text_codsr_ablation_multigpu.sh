#!/usr/bin/env bash
set -euo pipefail

# Run inference for CODSR-inspired text-SR main-training ablations.
#
# Examples:
#   GPUS=0,1,2,3 STEP=00020000 bash scripts/infer_text_codsr_ablation_multigpu.sh
#   GPUS=0,1,2,3 STEP=latest bash scripts/infer_text_codsr_ablation_multigpu.sh
#   INCLUDE='*ragp_only*' GPUS=0 bash scripts/infer_text_codsr_ablation_multigpu.sh
#   INCLUDE='*vae_lora*' GPUS=0,1 bash scripts/infer_text_codsr_ablation_multigpu.sh

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

GPUS="${GPUS:-0,1,2,3}"
STEP="${STEP:-latest}"
INCLUDE="${INCLUDE:-*text_codsr*}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_codsr_ablation}"
EXP_ROOTS="${EXP_ROOTS:-exp_vosr_text_codsr exp_vosr_text_codsr_distill}"
read -r -a EXP_ROOT_ARRAY <<< "${EXP_ROOTS}"

python scripts/infer_experiments_multigpu.py \
  --exp-roots "${EXP_ROOT_ARRAY[@]}" \
  --include "${INCLUDE}" \
  --checkpoint "${STEP}" \
  --gpus "${GPUS}" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --upscale 4 \
  --align-method nofix \
  --force-rerun
