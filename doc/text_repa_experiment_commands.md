# Text REPA Experiment Commands

本文档记录当前在 B baseline 基础上加入 REPA-style representation alignment
的三组实验设置和运行命令：

```text
B + DINO-REPA
B + DINO-Token-REPA
B + OCR-REPA
B + Segmentation-REPA
```

其中 B baseline 指：

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
```

也就是当前 OCR 表现最好的 guided-target direct distillation 设置。

---

## 1. 共同设置

Teacher 和 student 初始化 checkpoint：

```text
exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000/clean_weights/ema_model.safetensors
```

训练输出根目录：

```text
exp_vosr_text_distill_ablation/
```

默认评测输入：

```text
/data/ywk/datasets/real_test/LQ
```

推荐 shell 变量：

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_repa_ablation
export STEP=00020000
export NPROC_PER_NODE=4
```

REPA 训练的共同主损失：

```text
L = L_guided_distill + lambda_repa * L_repa
```

其中：

```text
L_guided_distill = || v_S - v_T^g ||^2
v_T^g = v_T^p + 0.5(v_T^f - v_T^p)
```

第一版 REPA 使用 global pooled token cosine alignment：

```text
student DiT hidden tokens -> projector -> pooled student representation
frozen encoder tokens -> pooled target representation
L_repa = 1 - cosine(student, target)
```

这样做是为了避免 DINO/OCR token 数量和 DiT latent patch token 数量不一致造成不稳定。

注意：原 REPA 仓库默认的 DINOv2 target 不是 DINO 第 8 层，而是 final normalized
patch tokens；原命令中的 `--encoder-depth=8` 指 student SiT 第 8 个 block 接
projector。为了不覆盖已经完成的旧 DINO 实验，本文保留实验 D，并新增 D2 作为更接近
原 REPA 的实现。

---

## 2. 实验 D：B + DINO-REPA

### 2.1 实验目的

验证在当前最优 B baseline 上，加入通用视觉表征对齐是否能够进一步改善
one-step text SR。

该实验使用 frozen DINOv2 从 HR 图像中提取 visual representation，作为 REPA target。

### 2.2 配置文件

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml
```

### 2.3 关键设置

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0

repa_type: dino
repa_weight: 0.5
repa_layer: 13
repa_dino_layer: 8
repa_target_dim: 768
repa_projector_hidden_dim: 1024
```

含义：

- `repa_type: dino`：使用 DINOv2 feature 作为 REPA target；
- `repa_layer: 13`：取 student DiT 第 13 个 block 后的 hidden tokens；
- `repa_dino_layer: 8`：取 DINOv2 第 8 层 feature；
- `repa_weight: 0.5`：REPA loss 权重；
- `u_weight: 0.0`：不启用 shortcut/RC consistency，只在 B 的 direct target matching 上加 REPA。

### 2.4 训练命令

推荐使用脚本：

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino
```

等价完整命令：

```bash
torchrun --nproc_per_node=4 train_vosr_distill.py \
  --config configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml
```

### 2.5 预期实验目录

```text
exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_dino_repa
```

### 2.6 推理命令

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_repa_ablation
export STEP=00020000

CKPT_DINO="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_dino_repa/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py \
  -c "${CKPT_DINO}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/D_dino_repa_step${STEP}" \
  -u 4 \
  --infer_steps 1 \
  --align_method nofix \
  --force_rerun
```

实际输出图像目录通常为：

```text
preset/results/text_repa_ablation/D_dino_repa_step00020000/sd2_steps1_seed42_shortcut/
```

---

## 3. 实验 D2：B + DINO-Token-REPA

### 3.1 实验目的

复现更接近原 REPA 的 DINO 表征对齐方式，用于判断之前 DINO-REPA 效果是否受
global pooling、DINO layer 选择、student 对齐层选择影响。

### 3.2 配置文件

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_dino_token_repa.yml
```

### 3.3 关键设置

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0

repa_type: dino
repa_weight: 0.5
repa_layer: 7
repa_align_mode: token
repa_dino_layer: norm
repa_target_dim: 768
repa_projector_hidden_dim: 1024
```

含义：

- `repa_align_mode: token`：使用 token-wise cosine alignment，更接近原 REPA；
- `repa_dino_layer: norm`：使用 DINOv2 final normalized patch tokens；
- `repa_layer: 7`：0-index block 7，对应第 8 个 student block，接近原 REPA `encoder_depth=8`；
- `repa_weight: 0.5`：保持与旧 DINO/OCR/Seg REPA 一致。

### 3.4 训练命令

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino_token
```

### 3.5 推理命令

```bash
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh dino_token
```

实际输出图像目录通常为：

```text
preset/results/text_repa_ablation/D2_dino_token_repa_step00020000/sd2_steps1_seed42_shortcut/
```

---

## 4. 实验 E：B + OCR-REPA

### 4.1 实验目的

验证将 REPA target 从通用 DINO visual representation 替换为 OCR recognizer
的 text-aware representation 后，是否更有利于 OCR fidelity。

该实验仍然不直接使用 OCR recognition loss，而是使用 frozen OCR encoder 的中间表征
作为 representation alignment target。

### 4.2 配置文件

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml
```

### 4.3 关键设置

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0

repa_type: ocr
repa_ocr_model: microsoft/trocr-base-printed
repa_weight: 0.5
repa_layer: 13
repa_projector_hidden_dim: 1024
```

含义：

- `repa_type: ocr`：使用 OCR/recognizer encoder feature 作为 REPA target；
- `repa_ocr_model`：默认使用 HuggingFace `microsoft/trocr-base-printed`；
- `repa_layer: 13`：取 student DiT 第 13 个 block 后的 hidden tokens；
- `repa_weight: 0.5`：REPA loss 权重；
- `u_weight: 0.0`：不启用 shortcut/RC consistency。

如果服务器无法联网下载 HuggingFace 模型，或者你希望使用本地 OCR 模型，可将：

```yaml
repa_ocr_model: microsoft/trocr-base-printed
```

改成本地模型目录，例如：

```yaml
repa_ocr_model: /path/to/local/trocr-or-ocr-model
```

### 4.4 训练命令

推荐使用脚本：

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr
```

等价完整命令：

```bash
torchrun --nproc_per_node=4 train_vosr_distill.py \
  --config configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml
```

### 4.5 预期实验目录

```text
exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_repa
```

### 4.6 推理命令

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_repa_ablation
export STEP=00020000

CKPT_OCR="exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_repa/checkpoints/checkpoint-${STEP}"

python inference_vosr_onestep.py \
  -c "${CKPT_OCR}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_ROOT}/E_ocr_repa_step${STEP}" \
  -u 4 \
  --infer_steps 1 \
  --align_method nofix \
  --force_rerun
```

实际输出图像目录通常为：

```text
preset/results/text_repa_ablation/E_ocr_repa_step00020000/sd2_steps1_seed42_shortcut/
```

---

## 5. 实验 F：B + Segmentation-REPA

### 5.1 实验目的

验证将 REPA target 替换为语义分割模型的结构/语义表征后，是否能提升文字区域附近的结构稳定性。

当前保留两个版本：

```text
F  = Seg-Global-REPA：最后一层 16x16 feature + global pooled cosine
F2 = Seg-Token-REPA：倒数第二层 32x32 feature + token-wise cosine
```

从机制上，F2 更适合文字 SR：VOSR student latent token 网格为 32x32，SegFormer-B2
倒数第二层也是 32x32，可以保留空间结构进行逐 token 对齐；F 更偏整图语义均值。

这个实验仍然沿用 B baseline：

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
```

也就是说，它只改变 REPA target，不改变 guided teacher target 蒸馏本身。

### 5.2 配置文件

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_seg_repa.yml
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_seg_token_repa.yml
```

### 5.3 关键设置

```yaml
repa_type: seg
repa_seg_model: nvidia/segformer-b2-finetuned-ade-512-512
repa_seg_layer: -1
repa_weight: 0.5
repa_layer: 13
repa_target_dim: 512
repa_projector_hidden_dim: 1024
```

F2 推荐设置：

```yaml
repa_type: seg
repa_seg_model: nvidia/segformer-b2-finetuned-ade-512-512
repa_seg_layer: -2
repa_weight: 0.5
repa_layer: 13
repa_align_mode: token
repa_target_dim: 320
repa_projector_hidden_dim: 1024
```

含义：

- `repa_type: seg`：使用 frozen segmentation encoder hidden states 作为 REPA target；
- `repa_seg_model`：默认使用 HuggingFace SegFormer-B2 ADE20K 512x512；
- `repa_seg_layer: -1`：取分割 encoder 最后一层 hidden state；
- `repa_target_dim: 512`：SegFormer-B2 最后一层通道数；
- `repa_weight: 0.5`：与 DINO/OCR REPA 的初始权重保持一致，方便横向对比。
- `repa_align_mode: token`：F2 使用逐 token 对齐，避免 global pooling 抹掉空间结构。

如果服务器无法联网下载 HuggingFace 模型，可将 `repa_seg_model` 改成本地 SegFormer 模型目录。

### 5.4 训练命令

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh seg
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh seg_token
```

### 5.5 推理命令

```bash
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh seg
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh seg_token
```

实际输出图像目录通常为：

```text
preset/results/text_repa_ablation/F_seg_repa_step00020000/sd2_steps1_seed42_shortcut/
preset/results/text_repa_ablation/F2_seg_token_repa_step00020000/sd2_steps1_seed42_shortcut/
```

---

## 6. 推荐对比表

建议将结果与已有 Teacher / A / B / C 放在同一张表中：

| ID | 方法 | Target | Extra loss | 主要目的 |
|---|---|---|---|---|
| Teacher | multi-step teacher | - | - | 蒸馏前 teacher baseline |
| A | full target | `omega=1.0` | 无 | full target baseline |
| B | guided target | `omega=0.5` | 无 | 当前最优 guided target baseline |
| C | guided + RC | `omega=0.5` | RCGM consistency | 验证 RC trajectory compression |
| D | guided + DINO-REPA | `omega=0.5` | DINO representation alignment | 验证通用视觉表征对齐 |
| D2 | guided + DINO-Token-REPA | `omega=0.5` | token-wise DINO final feature alignment | 更接近原 REPA 的 DINO 对齐 |
| E | guided + OCR-REPA | `omega=0.5` | OCR representation alignment | 验证文本识别表征对齐 |
| F | guided + Seg-REPA | `omega=0.5` | segmentation representation alignment | 验证结构/语义分割表征对齐 |

主要指标：

```text
Primary:
  OCR-EM
  OCR-CER
  OCR-CharAcc

Secondary:
  PSNR
  SSIM
  LPIPS

Diagnostic:
  DISTS
  NIQE
  MUSIQ
  MANIQA
```

---

## 7. 预期分析方式

### 7.1 DINO-REPA vs B

如果 DINO-REPA 比 B 更好，说明：

> 通用视觉 representation alignment 可以进一步提升 guided-target one-step
> distillation。

如果 DINO-REPA 提升图像质量但 OCR 不明显，说明：

> DINO 表征更偏通用视觉语义或感知质量，不一定直接对应文字可识别性。

### 7.2 OCR-REPA vs B

如果 OCR-REPA 比 B 的 OCR 更好，说明：

> 在不直接使用 OCR recognition loss 的情况下，文本识别表征对齐可以进一步提升
> one-step student 的 OCR fidelity。

这会支持后续方法故事：

```text
guidance-aware distillation
    + text-aware representation alignment
```

### 7.3 OCR-REPA vs DINO-REPA

如果 OCR-REPA 在 OCR 指标上优于 DINO-REPA，说明：

> 对文本超分而言，recognizer-aware representation 比 generic visual
> representation 更适合作为蒸馏正则。

如果 DINO-REPA 和 OCR-REPA 都有效，则说明：

> REPA-style representation alignment 是一个通用可插拔的 one-step distillation
> 增强项，其中 OCR encoder 是文本任务下更针对性的实例。
