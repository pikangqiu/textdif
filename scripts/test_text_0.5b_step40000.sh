#!/usr/bin/env bash
set -euo pipefail

# Fine-tuned VOSR-0.5B multi-step checkpoint at 40000 steps.
# Usage:
#   bash scripts/test_text_0.5b_step40000.sh
#   INPUT_DIR=/path/to/LQ OUTPUT_DIR=preset/results/my_eval bash scripts/test_text_0.5b_step40000.sh

CKPT="${CKPT:-exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000}"
INPUT_DIR="${INPUT_DIR:-/data/ywk/datasets/real_test/LQ}"
OUTPUT_DIR="${OUTPUT_DIR:-preset/results/text_0.5b_step40000}"

UPSCALE="${UPSCALE:-4}"
CFG_SCALE="${CFG_SCALE:-0.5}"
WEAK_COND="${WEAK_COND:-0.1}"
ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
INFER_STEPS="${INFER_STEPS:-25}"

python inference_vosr.py \
  -c "${CKPT}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -u "${UPSCALE}" \
  --infer_steps "${INFER_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --weak_cond_strength_aelq "${WEAK_COND}" \
  --align_method "${ALIGN_METHOD}" \
  --force_rerun
