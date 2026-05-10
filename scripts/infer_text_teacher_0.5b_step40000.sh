#!/usr/bin/env bash
set -euo pipefail

# Inference for the pre-distillation 0.5B text fine-tuned multi-step teacher.
# This is the teacher used by doc/text_distill_ablation_commands.md.
#
# Example:
#   bash scripts/infer_text_teacher_0.5b_step40000.sh
#   GPU=0 INPUT_DIR=/path/to/LQ bash scripts/infer_text_teacher_0.5b_step40000.sh

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

GPU="${GPU:-0}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_distill_ablation}"
UPSCALE="${UPSCALE:-4}"
INFER_STEPS="${INFER_STEPS:-25}"
CFG_SCALE="${CFG_SCALE:-0.5}"
WEAK_COND="${WEAK_COND:-0.1}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
FORCE_RERUN="${FORCE_RERUN:-1}"

CKPT="${CKPT:-exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000}"
OUT_DIR="${OUTPUT_ROOT}/teacher_ms_step40000"
LOG_DIR="${OUTPUT_ROOT}/_logs"
LOG_FILE="${LOG_DIR}/teacher_ms_step40000.log"

if [ ! -d "${CKPT}" ]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

cmd=(
  python inference_vosr.py
  -c "${CKPT}"
  -i "${INPUT_DIR}"
  -o "${OUT_DIR}"
  -u "${UPSCALE}"
  --infer_steps "${INFER_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --weak_cond_strength_aelq "${WEAK_COND}"
  --align_method "${ALIGN_METHOD}"
)

if [ "${FORCE_RERUN}" = "1" ] || [ "${FORCE_RERUN}" = "true" ]; then
  cmd+=(--force_rerun)
fi

{
  echo "GPU: ${GPU}"
  echo "Checkpoint: ${CKPT}"
  echo "Output: ${OUT_DIR}"
  echo "Command: CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}"
  echo
  CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
} > "${LOG_FILE}" 2>&1

echo "Done. Output: ${OUT_DIR}/sd2_steps${INFER_STEPS}_cfg${CFG_SCALE}_wc${WEAK_COND}"
echo "Log: ${LOG_FILE}"
