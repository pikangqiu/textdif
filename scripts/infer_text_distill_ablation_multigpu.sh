#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU inference for doc/text_distill_ablation_commands.md.
#
# Defaults evaluate the three ablation experiments at checkpoint-00020000:
#   A_full_target_no_rc
#   B_guided_target_no_rc
#   C_guided_target_rc
#
# Example:
#   bash scripts/infer_text_distill_ablation_multigpu.sh
#   GPUS=0,1,2 STEP=00020000 bash scripts/infer_text_distill_ablation_multigpu.sh

export PS1="${PS1:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_distill_ablation}"
STEP="${STEP:-00020000}"
GPUS="${GPUS:-0,1,2}"
UPSCALE="${UPSCALE:-4}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
INFER_STEPS="${INFER_STEPS:-1}"
FORCE_RERUN="${FORCE_RERUN:-1}"

EXP_A="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs1.0-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_full_target_no_rc"
EXP_B="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc"
EXP_C="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw1.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distrcgm_text_ablation_guided_target_rc"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "GPUS must contain at least one GPU id, e.g. GPUS=0,1,2" >&2
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
    echo "Command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    echo
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } > "${log_file}" 2>&1
}

declare -a JOBS=(
  "A_full_target_no_rc ${EXP_A}"
  "B_guided_target_no_rc ${EXP_B}"
  "C_guided_target_rc ${EXP_C}"
)

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
