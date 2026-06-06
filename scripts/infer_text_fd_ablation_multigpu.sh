#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU inference for FD-Loss text distillation experiments.
#
# Examples:
#   bash scripts/infer_text_fd_ablation_multigpu.sh shortcut_fd
#   GPUS=0 STEP=00020000 bash scripts/infer_text_fd_ablation_multigpu.sh shortcut_fd
#   GPUS=0,1 STEP=00020000 bash scripts/infer_text_fd_ablation_multigpu.sh all

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${1:-shortcut_fd}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_fd_ablation}"
STEP="${STEP:-00020000}"
GPUS="${GPUS:-0}"
UPSCALE="${UPSCALE:-4}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
INFER_STEPS="${INFER_STEPS:-1}"
FORCE_RERUN="${FORCE_RERUN:-1}"

EXP_NO_RC_FD="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_fd"
EXP_SHORTCUT_FD="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw1.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_shortcut_fd"

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
  no_rc_fd)
    JOBS=("FD_guided_target_no_rc ${EXP_NO_RC_FD}")
    ;;
  shortcut_fd)
    JOBS=("FD_guided_target_shortcut ${EXP_SHORTCUT_FD}")
    ;;
  all)
    JOBS=(
      "FD_guided_target_no_rc ${EXP_NO_RC_FD}"
      "FD_guided_target_shortcut ${EXP_SHORTCUT_FD}"
    )
    ;;
  *)
    echo "Unknown FD inference experiment: ${EXP}. Expected: no_rc_fd, shortcut_fd, or all." >&2
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
