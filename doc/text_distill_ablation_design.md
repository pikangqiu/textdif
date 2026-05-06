# Text Distillation Ablation Design

## Goal

Test whether one-step text recognizability comes from the distillation target
and trajectory compression, rather than from OCR supervision alone.

The immediate comparison is:

| ID | Method | Target | RC | OCR loss | Runnable now |
| --- | --- | --- | --- | --- | --- |
| A | one-step full target | `v_T^f` | no | no | yes |
| B | one-step guided target | `v_T^g` | no | no | yes |
| C | one-step guided target + RC | `v_T^g` | yes | no | yes |
| D | one-step guided target + RC + OCR loss | `v_T^g` | yes | yes | blocked |

## Current teacher / initialization

Use the existing text fine-tuned multi-step checkpoint:

```text
exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000/clean_weights/ema_model.safetensors
```

In these configs the same file is used in two roles:

- `teacher_ckpt`: frozen multi-step teacher that produces the distillation target.
- `pretrained_ckpt`: student initialization before one-step distillation.

This matches the existing text distillation setup and isolates the ablation to
target construction and RC.

## Runnable configs

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_full_target_no_rc.yml
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc.yml
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_rc.yml
```

Run one experiment:

```bash
bash scripts/train_text_distill_ablation.sh full_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_rc
```

Run all three sequentially:

```bash
bash scripts/train_text_distill_ablation.sh all
```

Use fewer or more GPUs by overriding:

```bash
NPROC_PER_NODE=1 bash scripts/train_text_distill_ablation.sh guided_target_rc
```

## How the configs encode the ablation

### A. Full target, no RC

Config:

```text
VOSR_0.5B_text_full_target_no_rc.yml
```

Settings:

```yaml
cfg_scale: 1.0
distill_type: shortcut
u_weight: 0.0
```

Because VOSR computes the teacher target as:

```text
v_T^g = v_T^p + omega * (v_T^f - v_T^p)
```

setting `cfg_scale: 1.0` makes the target exactly `v_T^f`.

### B. Guided target, no RC

Config:

```text
VOSR_0.5B_text_guided_target_no_rc.yml
```

Settings:

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
```

This keeps only direct velocity distillation to the guided teacher target.

### C. Guided target + RC

Config:

```text
VOSR_0.5B_text_guided_target_rc.yml
```

Settings:

```yaml
cfg_scale: 0.5
distill_type: rcgm
u_weight: 1.0
```

This tests whether RCGM trajectory compression explains the one-step OCR gain.

## Why `cfg_ratio: 0.0`

The one-step inference path uses the full condition only. For this first
distillation-target ablation, `cfg_ratio: 0.0` avoids mixing weak-condition
student inputs into the training batches and keeps the target comparison clean.
The teacher still computes both full and weak predictions internally to form the
guided target when `cfg_scale != 1.0`.

## OCR-loss row is intentionally blocked

The current repository does not yet expose:

- text labels from the dataset loader,
- a frozen OCR recognizer inside `train_vosr_distill.py`,
- CTC / CE OCR loss wiring,
- a recognizer preprocessing contract.

Therefore the fourth row should not be represented by a YAML-only switch. The
correct next implementation is to add a label-aware dataset path and an OCR loss
module, then copy `VOSR_0.5B_text_guided_target_rc.yml` and enable that loss.

## Evaluation table

After training, evaluate checkpoints with the same OCR recognizers and image
metrics:

| ID | Target | RC | OCR loss | Word Acc | Char Acc | NED | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `v_T^f` | no | no | | | | | | |
| B | `v_T^g` | no | no | | | | | | |
| C | `v_T^g` | yes | no | | | | | | |
| D | `v_T^g` | yes | yes | | | | | | |

The important claim is supported only if B improves over A, and C improves over
B on OCR metrics without requiring OCR loss.
