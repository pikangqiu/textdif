#!/usr/bin/env bash
set -euo pipefail

# Run guided-target text distillation with REPA-style representation alignment.
#
# Usage:
#   bash scripts/train_text_repa_ablation.sh dino
#   bash scripts/train_text_repa_ablation.sh ocr
#   NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino

export PS1="${PS1:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${1:-dino}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CONFIG_ROOT="configs/train_yml/one_step/text_distill_ablation"

case "${EXP}" in
  dino)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml"
    ;;
  ocr)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml"
    ;;
  *)
    echo "Unknown REPA experiment: ${EXP}. Expected: dino or ocr." >&2
    exit 2
    ;;
esac

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_vosr_distill.py --config "${CONFIG}"
