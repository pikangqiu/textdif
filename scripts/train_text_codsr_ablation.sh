#!/usr/bin/env bash
set -euo pipefail

# Run text-SR main-framework ablations inspired by CODSR.
#
# Usage:
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh ragp_only
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh lq_mod_only
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh ragp_lq_mod
#   NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh ragp_lq_mod_vae_lora

export PS1="${PS1:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

EXP="${1:-ragp_lq_mod}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CONFIG_ROOT="configs/train_yml/one_step/text_codsr"

case "${EXP}" in
  ragp_only)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_only.yml"
    ;;
  lq_mod_only)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_lq_mod_only.yml"
    ;;
  ragp_lq_mod)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_lq_mod.yml"
    ;;
  ragp_lq_mod_vae_lora)
    CONFIG="${CONFIG_ROOT}/VOSR_0.5B_text_ragp_lq_mod_vae_lora.yml"
    ;;
  *)
    echo "Unknown CODSR-inspired text experiment: ${EXP}. Expected: ragp_only, lq_mod_only, ragp_lq_mod, or ragp_lq_mod_vae_lora." >&2
    exit 2
    ;;
esac

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_vosr.py --config "${CONFIG}"
