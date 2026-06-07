#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU inference for B + REPA experiments.
#
# Defaults evaluate DINO-REPA at checkpoint-00020000.
#
# Examples:
#   bash scripts/infer_text_repa_ablation_multigpu.sh
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh dino
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh dino_token
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh ocr_local
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh ocr_ctc
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh seg
#   GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh seg_token
#   GPUS=0,1 bash scripts/infer_text_repa_ablation_multigpu.sh all

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${1:-dino}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_repa_ablation}"
STEP="${STEP:-00020000}"
GPUS="${GPUS:-0}"
UPSCALE="${UPSCALE:-4}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
INFER_STEPS="${INFER_STEPS:-1}"
FORCE_RERUN="${FORCE_RERUN:-1}"

EXP_DINO="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_dino_repa"
EXP_DINO_TOKEN="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_dino_token_repa"
EXP_OCR="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_repa"
EXP_OCR_LOCAL="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_local_repa"
EXP_OCR_CTC="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_ctc"
EXP_SEG="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_seg_repa"
EXP_SEG_TOKEN="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_seg_token_repa"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "GPUS must contain at least one GPU id, e.g. GPUS=0,1" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/_logs"

run_one() {
  local gpu="$1"
  local tag="$2"
  local exp_dir="$3"
  local ckpt="${exp_dir}/checkpoints/checkpoint-${STEP}"
  local out_dir="${OUTPUT_ROOT}/${tag}_step${STEP}"
  local log_file="${OUTPUT_ROOT}/_logs/${tag}_step${STEP}.log"

  if [ ! -d "${ckpt}" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 2
  fi

  local -a cmd=(
    python inference_vosr_onestep.py
    -c "${ckpt}"
    -i "${INPUT_DIR}"
    -o "${out_dir}"
    -u "${UPSCALE}"
    --infer_steps "${INFER_STEPS}"
    --align_method "${ALIGN_METHOD}"
  )

  if [ "${FORCE_RERUN}" = "1" ] || [ "${FORCE_RERUN}" = "true" ]; then
    cmd+=(--force_rerun)
  fi

  {
    echo "GPU: ${gpu}"
    echo "Checkpoint: ${ckpt}"
    echo "Output: ${out_dir}"
    echo "Command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    echo
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } > "${log_file}" 2>&1
}

case "${EXP}" in
  dino)
    JOBS=("D_dino_repa ${EXP_DINO}")
    ;;
  dino_token)
    JOBS=("D2_dino_token_repa ${EXP_DINO_TOKEN}")
    ;;
  ocr)
    JOBS=("E_ocr_repa ${EXP_OCR}")
    ;;
  ocr_local)
    JOBS=("E2_ocr_local_repa ${EXP_OCR_LOCAL}")
    ;;
  ocr_ctc)
    JOBS=("E3_ocr_ctc ${EXP_OCR_CTC}")
    ;;
  seg)
    JOBS=("F_seg_repa ${EXP_SEG}")
    ;;
  seg_token)
    JOBS=("F2_seg_token_repa ${EXP_SEG_TOKEN}")
    ;;
  all)
    JOBS=("D_dino_repa ${EXP_DINO}" "D2_dino_token_repa ${EXP_DINO_TOKEN}" "E_ocr_repa ${EXP_OCR}" "E2_ocr_local_repa ${EXP_OCR_LOCAL}" "E3_ocr_ctc ${EXP_OCR_CTC}" "F_seg_repa ${EXP_SEG}" "F2_seg_token_repa ${EXP_SEG_TOKEN}")
    ;;
  *)
    echo "Unknown REPA inference experiment: ${EXP}. Expected: dino, dino_token, ocr, ocr_local, ocr_ctc, seg, seg_token, or all." >&2
    exit 2
    ;;
esac

declare -a PIDS=()
declare -a TAGS=()

for i in "${!JOBS[@]}"; do
  tag="${JOBS[$i]%% *}"
  exp_dir="${JOBS[$i]#* }"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"

  echo "[GPU ${gpu}] ${tag} checkpoint-${STEP}"
  run_one "${gpu}" "${tag}" "${exp_dir}" &
  PIDS+=("$!")
  TAGS+=("${tag}")
done

failures=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[OK] ${TAGS[$i]}"
  else
    echo "[FAIL] ${TAGS[$i]} log=${OUTPUT_ROOT}/_logs/${TAGS[$i]}_step${STEP}.log" >&2
    failures=$((failures + 1))
  fi
done

exit "${failures}"
