#!/usr/bin/env bash
set -euo pipefail

# CFG sweep for the fine-tuned VOSR-0.5B 40000-step checkpoint.
# By default this creates/uses a 100-image subset to keep evaluation quick.
# Usage:
#   bash scripts/test_text_0.5b_step40000_cfg_sweep.sh
#   INPUT_DIR=/path/to/LQ_100 bash scripts/test_text_0.5b_step40000_cfg_sweep.sh

CKPT="${CKPT:-exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000}"
SOURCE_DIR="${SOURCE_DIR:-/data/ywk/datasets/real_test/LQ}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ_100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-preset/results/text_0.5b_step40000_cfg_sweep}"

UPSCALE="${UPSCALE:-4}"
WEAK_COND="${WEAK_COND:-0.1}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
INFER_STEPS="${INFER_STEPS:-25}"

if [ ! -d "${INPUT_DIR}" ] || [ -z "$(find "${INPUT_DIR}" -maxdepth 1 -type f | head -1)" ]; then
  mkdir -p "${INPUT_DIR}"
  find "${SOURCE_DIR}" -maxdepth 1 -type f | sort | head -100 | xargs -I{} cp {} "${INPUT_DIR}/"
fi

for CFG_SCALE in -0.5 0 0.5 1.0 1.5 2.0; do
  SAFE_CFG="${CFG_SCALE/-/m}"
  SAFE_CFG="${SAFE_CFG/./p}"
  python inference_vosr.py \
    -c "${CKPT}" \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_ROOT}/cfg${SAFE_CFG}_wc${WEAK_COND}_${ALIGN_METHOD}" \
    -u "${UPSCALE}" \
    --infer_steps "${INFER_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --weak_cond_strength_aelq "${WEAK_COND}" \
    --align_method "${ALIGN_METHOD}" \
    --force_rerun
done
