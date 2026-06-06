# VAE Encoder LoRA + OCR-REPA 两阶段实验

## 实验目的

该实验不是直接把 OCR loss 塞进现有蒸馏，而是先验证 VAE 的 LQ 编码是否限制了文本结构表达，再验证 OCR-REPA 能否在一步蒸馏时保留这种结构收益。

两阶段设置：

1. **Teacher**：RAGP + LQ token modulation + VAE encoder LoRA，多步 flow-matching 训练。
2. **Student**：加载上述 teacher 和同一份 VAE LoRA，冻结 VAE LoRA，执行 guided-target + OCR-REPA 一步蒸馏。

## VAE LoRA 实现

- 目标模块选择忠实沿用 CODSR：VAE encoder 中的卷积、注意力投影，以及 `quant_conv`。
- LoRA rank：`4`。
- 初始化：PEFT `init_lora_weights="gaussian"`。
- DiT 学习率：`5e-6`。
- VAE LoRA 学习率：`5e-5`，与 CODSR Stage 1 默认学习率一致。
- 只有 LQ 编码分支经过可训练 LoRA。
- HQ target 使用禁用 adapter 的基础 VAE 编码，避免训练目标 latent 随 LoRA 一起漂移。
- 蒸馏和推理阶段加载并冻结 VAE LoRA。

CODSR 原仓库使用 `peft==0.9.0`，但本项目的 `diffusers==0.35.0` 要求 `peft>=0.17.0`，因此这里使用兼容版本 `peft==0.17.0`。LoRA 的目标层、rank 和初始化方式不变。

## 依赖

```bash
cd /data/ywk/VOSR
conda activate vosr
pip install peft==0.17.0
```

## Stage 1：训练多步 Teacher

```bash
cd /data/ywk/VOSR
NPROC_PER_NODE=4 bash scripts/train_text_codsr_ablation.sh ragp_lq_mod_vae_lora
```

配置：

```text
configs/train_yml/one_step/text_codsr/VOSR_0.5B_text_ragp_lq_mod_vae_lora.yml
```

20k checkpoint：

```text
exp_vosr_text_codsr/ldit_fm_bs016_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_codsr_ragp_lq_mod_vae_lora/checkpoints/checkpoint-00020000/
```

关键权重：

```text
clean_weights/ema_model.safetensors
clean_weights/vae_encoder_lora.safetensors
```

### Teacher 推理

```bash
GPUS=0,1,2,3 \
STEP=00020000 \
EXP_ROOTS=exp_vosr_text_codsr \
INCLUDE='*ragp_lq_mod_vae_lora' \
OUTPUT_ROOT=preset/results/text_codsr_vae_lora_teacher \
bash scripts/infer_text_codsr_ablation_multigpu.sh
```

输出位于：

```text
preset/results/text_codsr_vae_lora_teacher/
```

## Stage 2：OCR-REPA 一步蒸馏

Stage 1 的 20k DiT 和 VAE LoRA 权重必须同时存在。

```bash
cd /data/ywk/VOSR
NPROC_PER_NODE=4 bash scripts/train_text_codsr_distill.sh vae_lora_ocr_repa
```

配置：

```text
configs/train_yml/one_step/text_codsr_distill/VOSR_0.5B_text_ragp_lq_mod_vae_lora_guided_target_ocr_repa.yml
```

该配置保持现有 OCR-REPA 基线：

- `distill_type: shortcut`
- `u_weight: 0.0`
- `cfg_scale: 0.5`
- `repa_type: ocr`
- `repa_weight: 0.5`
- `repa_layer: 13`

这里的 `shortcut` 表示进入 guided-target shortcut 实现，但 `u_weight=0`，因此没有额外 RC/shortcut consistency loss。这样可以把变化集中在结构增强 teacher、VAE LoRA 和 OCR-REPA 上。

### Student 推理

```bash
GPUS=0,1,2,3 \
STEP=00020000 \
EXP_ROOTS=exp_vosr_text_codsr_distill \
INCLUDE='*vae_lora*ocr_repa*' \
OUTPUT_ROOT=preset/results/text_codsr_vae_lora_ocr_repa_distill \
bash scripts/infer_text_codsr_ablation_multigpu.sh
```

输出位于：

```text
preset/results/text_codsr_vae_lora_ocr_repa_distill/
```

推理脚本会自动从当前 checkpoint 的 `clean_weights/vae_encoder_lora.safetensors` 加载 LoRA。迁移服务器时，应同时复制整个 checkpoint 目录，而不是只复制 `ema_model.safetensors`。

## 公平对比

建议至少比较以下四组，并统一 20k 新训练步数、数据、CFG、推理步数和评测集：

| 组别 | RAGP + LQ-mod | VAE LoRA | OCR-REPA | 作用 |
|---|---:|---:|---:|---|
| 原结构增强 teacher | 是 | 否 | 否 | 当前结构增强基线 |
| 新 teacher | 是 | 是 | 否 | 判断 VAE LQ 编码适配是否有效 |
| 原结构增强 distill | 是 | 否 | 否 | 当前一步蒸馏基线 |
| 新 distill | 是 | 冻结 | 是 | 判断 VAE 适配与 OCR 表征蒸馏是否互补 |

优先观察：

- OCR-CharAcc / OCR-CER：文字可读性。
- PSNR / SSIM：结构和像素保真。
- LPIPS / DISTS：感知结构。
- NIQE / MUSIQ：是否因文本约束牺牲自然度。

如果新 teacher 本身没有提升，不应只根据新 student 的结果判断 OCR-REPA；这意味着 VAE LoRA 阶段需要先单独调学习率或训练步数。
