# Text Distillation Ablation Commands

This document records the runnable training and inference commands for the
current one-step text distillation ablation.

## Shared Paths

Teacher and student initialization checkpoint:

```text
exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000/clean_weights/ema_model.safetensors
```

Training output root:

```text
exp_vosr_text_distill_ablation/
```

Default evaluation input:

```text
/data/ywk/datasets/real_test/LQ
```

Recommended shell variables:

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_distill_ablation
export STEP=00020000
export NPROC_PER_NODE=4
```

`STEP` should match the checkpoint you want to evaluate. The configs save every
1000 steps and train to 20000 steps by default.

## Experiment A: Full Target, No RC

Purpose:

```text
Baseline one-step distillation. The student learns the full teacher prediction
v_T^f without guided-target mixing or RC.
```

Config:

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_full_target_no_rc.yml
```

Key settings:

```yaml
cfg_scale: 1.0
distill_type: shortcut
u_weight: 0.0
```

Training:

```bash
bash scripts/train_text_distill_ablation.sh full_target_no_rc
```

Expected experiment directory:

```text
exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs1.0-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_full_target_no_rc
```

Inference:

```bash
CKPT="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs1.0-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_full_target_no_rc/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py \
  -c "${CKPT}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/A_full_target_no_rc_step${STEP}" \
  -u 4 \
  --infer_steps 1 \
  --align_method nofix \
  --force_rerun
```

## Experiment B: Guided Target, No RC

Purpose:

```text
Tests whether the guided teacher target v_T^g is better than directly learning
the full teacher target v_T^f.
```

Config:

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc.yml
```

Key settings:

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
```

Training:

```bash
bash scripts/train_text_distill_ablation.sh guided_target_no_rc
```

Expected experiment directory:

```text
exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc
```

Inference:

```bash
CKPT="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py \
  -c "${CKPT}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/B_guided_target_no_rc_step${STEP}" \
  -u 4 \
  --infer_steps 1 \
  --align_method nofix \
  --force_rerun
```

## Experiment C: Guided Target + RC

Purpose:

```text
Tests whether RCGM / recursive consistency improves one-step OCR stability on
top of the guided teacher target.
```

Config:

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_rc.yml
```

Key settings:

```yaml
cfg_scale: 0.5
distill_type: rcgm
u_weight: 1.0
rcgm_delta_t: 0.01
rcgm_n_steps: 2
```

Training:

```bash
bash scripts/train_text_distill_ablation.sh guided_target_rc
```

Expected experiment directory:

```text
exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw1.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distrcgm_text_ablation_guided_target_rc
```

Inference:

```bash
CKPT="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw1.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distrcgm_text_ablation_guided_target_rc/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py \
  -c "${CKPT}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/C_guided_target_rc_step${STEP}" \
  -u 4 \
  --infer_steps 1 \
  --align_method nofix \
  --force_rerun
```

## Run All Training Jobs

Run the three current ablations sequentially:

```bash
bash scripts/train_text_distill_ablation.sh all
```

Override GPU count:

```bash
NPROC_PER_NODE=1 bash scripts/train_text_distill_ablation.sh guided_target_rc
```

## Run All Inference Jobs

After training has produced `checkpoint-${STEP}` for all three experiments:

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_distill_ablation
export STEP=00020000

CKPT_A="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs1.0-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_full_target_no_rc/checkpoints/checkpoint-${STEP}"
CKPT_B="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc/checkpoints/checkpoint-${STEP}"
CKPT_C="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw1.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distrcgm_text_ablation_guided_target_rc/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py -c "${CKPT_A}" -i "${INPUT_DIR}" -o "${OUTPUT_ROOT}/A_full_target_no_rc_step${STEP}" -u 4 --infer_steps 1 --align_method nofix --force_rerun
python inference_vosr_onestep.py -c "${CKPT_B}" -i "${INPUT_DIR}" -o "${OUTPUT_ROOT}/B_guided_target_no_rc_step${STEP}" -u 4 --infer_steps 1 --align_method nofix --force_rerun
python inference_vosr_onestep.py -c "${CKPT_C}" -i "${INPUT_DIR}" -o "${OUTPUT_ROOT}/C_guided_target_rc_step${STEP}" -u 4 --infer_steps 1 --align_method nofix --force_rerun
```

## Optional Teacher Baseline Inference

Run the text fine-tuned multi-step teacher for comparison:

```bash
python inference_vosr.py \
  -c exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000 \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/teacher_ms_step40000" \
  -u 4 \
  --infer_steps 25 \
  --cfg_scale 0.5 \
  --weak_cond_strength_aelq 0.1 \
  --align_method nofix \
  --force_rerun
```

## Evaluation Targets

Use the same OCR and image-quality evaluation scripts for every output folder.
The result table should include:

```text
Word Acc
Char Acc
Normalized Edit Distance
PSNR
SSIM
LPIPS
Runtime
```

Primary comparisons:

```text
B vs A: guided target effect
C vs B: RC / trajectory-compression effect
C vs teacher_ms: whether one-step distillation beats the multi-step teacher on OCR
```

The OCR-loss row is intentionally not listed as a runnable command yet. The
current codebase still needs label-aware data loading and a frozen recognizer
loss before that ablation is real.
