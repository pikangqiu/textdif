#!/usr/bin/env bash
# Auto-chain: wait for the 2b-GT run (ocr_ctc_gt) to finish, GPU-smoke the new
# OCR-cond code path on one card, then launch the real 4-card E1 training.
# Runs detached in its own screen so it survives terminal/agent sessions.
set -uo pipefail
cd /data/ywk/VOSR

CHAIN_LOG=logs/auto_chain_ocr_cond.log
SMOKE_LOG=logs/smoke_ocr_cond.log
CFG=configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_ocr_cond.yml

log() { echo "[$(date '+%F %T')] $*" >> "${CHAIN_LOG}"; }

export PS1="${PS1:-}"
source /home/ywk/anaconda3/etc/profile.d/conda.sh
conda activate vosr
export TORCH_COMPILE_DISABLE=1

log "chain started; waiting for 2b-GT (ocr_ctc_gt) to finish"

# --- 1. Wait for the 2b torchrun to exit -----------------------------------
while pgrep -f "text_guided_target_no_rc_ocr_ctc_gt.yml" > /dev/null; do
  sleep 120
done

if grep -q "20000/20000" logs/train_ocr_ctc_gt.log; then
  log "2b-GT reached 20000/20000 (clean finish)"
else
  log "WARNING: 2b-GT process gone WITHOUT reaching 20000 - check logs/train_ocr_ctc_gt.log. Proceeding with E1 anyway."
fi

sleep 60  # let GPU memory release

# --- 2. Single-GPU batch-1 smoke of the OCR-cond path ----------------------
sed -e "s/^train_batch_size: 4/train_batch_size: 1/" \
    -e "s/^suffix: .*/suffix: '_smoke_ocr_cond'/" \
    -e "s/^tracker_project_name: .*/tracker_project_name: smoke_ocr_cond/" \
    "${CFG}" > /tmp/ocr_cond_smoke.yml

log "starting smoke on GPU3 (batch=1)"
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29589 \
  train_vosr_distill.py --config /tmp/ocr_cond_smoke.yml > "${SMOKE_LOG}" 2>&1 &
SMOKE_PID=$!

# Wait up to 40 min for the step-50 postfix (ocr_cond_gate=) or a crash.
SMOKE_OK=0
for i in $(seq 1 240); do
  if grep -q "Traceback" "${SMOKE_LOG}" 2>/dev/null; then break; fi
  if grep -q "ocr_cond_gate=" "${SMOKE_LOG}" 2>/dev/null; then SMOKE_OK=1; break; fi
  if ! kill -0 "${SMOKE_PID}" 2>/dev/null; then break; fi
  sleep 10
done

kill "${SMOKE_PID}" 2>/dev/null
pkill -f "ocr_cond_smoke.yml" 2>/dev/null
sleep 20
rm -rf exp_vosr_text_distill_ablation/*smoke_ocr_cond* 2>/dev/null

if [ "${SMOKE_OK}" != "1" ]; then
  log "SMOKE FAILED - E1 NOT launched. See ${SMOKE_LOG} (kept). Last lines:"
  tail -c 1500 "${SMOKE_LOG}" >> "${CHAIN_LOG}" 2>/dev/null
  exit 1
fi
log "smoke passed (ocr_cond_gate visible at step 50); launching real 4-card E1"

# --- 3. Launch the real E1 run ----------------------------------------------
screen -dmS train_ocr_cond bash -c 'NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_cond > logs/train_ocr_cond.log 2>&1'
sleep 10
if pgrep -f "text_guided_target_no_rc_ocr_cond.yml" > /dev/null; then
  log "E1 (OCR-cond v1) training launched in screen train_ocr_cond, log logs/train_ocr_cond.log"
else
  log "ERROR: E1 launch did not start a torchrun - check logs/train_ocr_cond.log"
  exit 1
fi
