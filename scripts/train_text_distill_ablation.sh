#!/usr/bin/env bash
set -euo pipefail

# Run the text distillation ablation jobs.
#
# Usage:
#   bash scripts/train_text_distill_ablation.sh full_target_no_rc
#   bash scripts/train_text_distill_ablation.sh guided_target_no_rc
#   bash scripts/train_text_distill_ablation.sh guided_target_rc
#   bash scripts/train_text_distill_ablation.sh all
#
# Optional:
#   NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh all

EXP="${1:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

CONFIG_ROOT="configs/train_yml/one_step/text_distill_ablation"

run_one() {
  local config="$1"
  echo "==> Running ${config}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" train_vosr_distill.py --config "${config}"
}

case "${EXP}" in
  full_target_no_rc)
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_full_target_no_rc.yml"
    ;;
  guided_target_no_rc)
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc.yml"
    ;;
  guided_target_rc)
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_rc.yml"
    ;;
  all)
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_full_target_no_rc.yml"
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_no_rc.yml"
    run_one "${CONFIG_ROOT}/VOSR_0.5B_text_guided_target_rc.yml"
    ;;
  *)
    echo "Unknown experiment: ${EXP}" >&2
    exit 2
    ;;
esac
