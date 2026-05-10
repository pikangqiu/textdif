#!/usr/bin/env bash
set -euo pipefail

export PS1="${PS1:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

if [ "${CONDA_DEFAULT_ENV:-}" != "vosr" ]; then
  source /home/ywk/anaconda3/etc/profile.d/conda.sh
  conda activate vosr
fi

torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" train_vosr.py \
  --config configs/train_yml/multi_step/VOSR_1.4B_official_mydata_smoke.yml
