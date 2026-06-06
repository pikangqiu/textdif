#!/usr/bin/env bash
set -euo pipefail

# Distill the CODSR-inspired RAGP + LQ-mod text-SR teacher.
#
# Prerequisite:
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh ragp_lq_mod
#
# Usage:
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_distill.sh no_rc
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_distill.sh shortcut
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_distill.sh vae_lora_ocr_repa

export PS1="${PS1:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${1:-shortcut}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CONFIG_ROOT="configs/train_yml/one_step/text_codsr_distill"

case "${EXP}" in
  no_rc)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_lq_mod_guided_target_no_rc.yml"
    TEACHER_NAME="text_codsr_ragp_lq_mod"
    ;;
  shortcut)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_lq_mod_guided_target_shortcut.yml"
    TEACHER_NAME="text_codsr_ragp_lq_mod"
    ;;
  vae_lora_ocr_repa)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_lq_mod_vae_lora_guided_target_ocr_repa.yml"
    TEACHER_NAME="text_codsr_ragp_lq_mod_vae_lora"
    ;;
  *)
    echo "Unknown CODSR distill experiment: ${EXP}. Expected: no_rc, shortcut, or vae_lora_ocr_repa." >&2
    exit 2
    ;;
esac

TEACHER_CKPT="exp_vosr_text_codsr/ldit_fm_bs016_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_${TEACHER_NAME}/checkpoints/checkpoint-00020000/clean_weights/ema_model.safetensors"
if [ ! -f "${TEACHER_CKPT}" ]; then
  echo "Missing teacher checkpoint: ${TEACHER_CKPT}" >&2
  echo "Run: NPROC_PER_NODE=${NPROC_PER_NODE} bash scripts/train_text_codsr_ablation.sh ${TEACHER_NAME#text_codsr_}" >&2
  exit 1
fi

if [ "${EXP}" = "vae_lora_ocr_repa" ]; then
  VAE_LORA_CKPT="$(dirname "${TEACHER_CKPT}")/vae_encoder_lora.safetensors"
  if [ ! -f "${VAE_LORA_CKPT}" ]; then
    echo "Missing VAE encoder LoRA checkpoint: ${VAE_LORA_CKPT}" >&2
    exit 1
  fi
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_vosr_distill.py --config "${CONFIG}"
