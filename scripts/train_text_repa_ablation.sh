#!/usr/bin/env bash
set -euo pipefail

# Run guided-target text distillation with REPA-style representation alignment.
#
# Usage:
#   bash scripts/train_text_repa_ablation.sh dino
#   bash scripts/train_text_repa_ablation.sh dino_token
#   bash scripts/train_text_repa_ablation.sh ocr
#   bash scripts/train_text_repa_ablation.sh ocr_local
#   bash scripts/train_text_repa_ablation.sh seg
#   bash scripts/train_text_repa_ablation.sh seg_token
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
  dino_token)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_dino_token_repa.yml"
    ;;
  ocr)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml"
    ;;
  ocr_local)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_ocr_local_repa.yml"
    ;;
  seg)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_seg_repa.yml"
    ;;
  seg_token)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc_seg_token_repa.yml"
    ;;
  *)
    echo "Unknown REPA experiment: ${EXP}. Expected: dino, dino_token, ocr, ocr_local, seg, or seg_token." >&2
    exit 2
    ;;
esac

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_vosr_distill.py --config "${CONFIG}"
