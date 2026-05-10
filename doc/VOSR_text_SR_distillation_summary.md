# 基于 VOSR 的文本超分一步蒸馏研究方案总结

## 1. 背景与核心问题

当前讨论围绕 VOSR（Vision-Only Generative Model for Image Super-Resolution）展开，目标是将其思想迁移到文本超分辨率（Scene Text Image Super-Resolution, STISR）任务中。

VOSR 的核心特点是：它不依赖 Stable Diffusion / SDXL / SD3 这类 Text-to-Image 预训练模型，而是从图像恢复任务本身出发，构建一个 vision-only 的生成式超分框架。其主要组成包括：

- LR 图像经过 VAE encoder 得到 structural latent condition；
- LR 图像经过 DINOv2 等视觉编码器得到 visual semantic condition；
- 使用 DiT / LightningDiT 作为扩散主干；
- 使用 restoration-oriented CFG，而不是传统 unconditional CFG；
- 先训练 multi-step teacher，再蒸馏成 one-step student。

本文档要总结的核心研究问题是：

> one-step 文本识别率提升，是否来自 guided teacher target 和 RC trajectory compression，而不是单纯 OCR loss？

也就是说，研究重点不是简单地把多步模型压缩成一步模型，而是探索：

> 能否设计一种面向文本可识别性的 guidance-aware / OCR-aware 蒸馏方式，把多步 teacher 中更适合文本恢复的方向蒸馏到 one-step student 中。

---

## 2. 对 VOSR 的关键理解

### 2.1 为什么 VOSR 使用 DiT 而不是 SD-UNet

VOSR 并不是简单地把 SD 的 U-Net 换成 DiT，而是改变了超分任务的建模范式。

传统 SD-based SR 方法通常从 T2I 生成模型出发，再通过 prompt、adapter、ControlNet、low-resolution condition 等方式约束生成结果。但 SR 本质上不是自由生成，而是 LR input-conditioned restoration。VOSR 认为 T2I 模型存在一个结构性冲突：

- T2I prior 强调图像生成真实性；
- SR 任务强调对 LR 输入的结构忠实；
- 文本超分中，字符身份正确比“看起来像字”更重要。

因此，VOSR 选择 vision-only 设计，直接以 LR 图像为条件训练恢复模型。

DiT 对该任务有几个潜在优势：

1. **更适合 token-level 视觉语义融合**：DINOv2 输出 patch/token 风格的视觉特征，DiT 本身也是 token-based transformer，因此二者融合更加自然。
2. **全局 attention 更适合文本结构恢复**：文本字符往往依赖上下文，例如 `B/8`、`O/0`、`I/l/1` 等混淆情况。
3. **减少 T2I prior 带来的字符幻觉**：SD 类模型可能生成“看起来合理但不忠实”的文字，而 VOSR 更强调 input-anchored restoration。

---

### 2.2 VOSR 为什么训练代价较小

VOSR 训练代价较小的原因不是模型本身特别小，而是它避开了 T2I-based SR 的大规模多模态训练负担。

主要原因包括：

- 不训练 text encoder；
- 不需要 web-scale image-text pair；
- 不依赖 prompt / caption / text-image alignment；
- VAE 和 DINOv2 主要作为冻结特征提取器；
- 核心训练对象是 DiT restoration backbone；
- one-step distillation 选择 memory-friendly 的 shortcut / recursive consistency 方案，而不是高显存的 JVP 或多网络耦合方案。

VOSR 论文中也指出，其训练成本低于代表性 T2I-based SR 方法的十分之一。

---

### 2.3 VOSR-0.5B 和 VOSR-1.4B 的主要差异

| 项目 | VOSR-0.5B | VOSR-1.4B |
|---|---:|---:|
| Transformer dim | 1024 | 1536 |
| depth | 28 | 36 |
| heads | 16 | 24 |
| VAE | SD2.1 4-channel VAE | Qwen 16-channel VAE |
| semantic encoder | DINOv2-Base | DINOv2-Large |
| latent channel | 4 | 16 |

主要参数差距来自：

1. DiT 主干宽度和深度；
2. VAE latent channel 数；
3. DINOv2 encoder 规模。

对文本超分而言，16-channel VAE 可能更有利于保留字符细节，但 VOSR README 中也提到，0.5B 的 lightweight decoder 在 text-rich / document-like images 上可能更友好。因此，文本超分不一定必须从 1.4B 起步，0.5B 是更实际的实验起点。

---

## 3. VOSR 的 Restoration-Oriented CFG

### 3.1 传统 CFG 的问题

传统 CFG 通常是：

```math
v_{cfg} = v_{uncond} + s(v_{cond} - v_{uncond})
```

其中 unconditional branch 完全不接收条件。

但在 SR 中，这种设计存在问题：

- 如果完全去掉 LR 条件，unconditional branch 要学习通用图像生成；
- conditional branch 独自承担 LR fidelity；
- 对从零训练的 restoration model 来说，这种角色分离很难优化；
- 如果 unconditional branch 学不好，guidance 方向会不稳定，甚至伤害恢复质量。

---

### 3.2 VOSR 的 Partial Conditioning

VOSR 将 unconditional branch 替换为 partially conditioned branch。

其设计为：

```text
full branch:
    strong LR structural condition + visual semantic condition

partial branch:
    weak LR structural condition + no visual semantic condition
```

因此 guided prediction 为：

```math
v_T^g = v_T^p + \omega (v_T^f - v_T^p)
```

其中：

- `v_T^f`：fully conditioned teacher prediction；
- `v_T^p`：partially conditioned teacher prediction；
- `omega`：distillation-time guidance weight。

这与传统 CFG 的区别是：

- 传统 CFG：从 unconditional generation 指向 conditional generation；
- VOSR CFG：从 weakly anchored restoration 指向 strongly anchored restoration。

因此，VOSR 的 CFG 是 restoration-oriented 的。

---

### 3.3 VOSR 关于 CFG 的消融证据

VOSR 论文中做了 partial conditioning 的消融实验，对比：

| Guidance design | LPIPS | MUSIQ |
|---|---:|---:|
| Full condition only | 0.3752 | 67.29 |
| Standard CFG | 0.4053 | 50.78 |
| Ours partial conditioning | 0.3772 | 69.26 |

结论：

- standard CFG 明显退化；
- partial conditioning 效果最好；
- input-anchored auxiliary branch 更适合 SR；
- guidance 设计本身对恢复质量非常关键。

---

## 4. VOSR 的 One-Step Distillation

### 4.1 不是 VSD

VOSR 的一步蒸馏不是传统意义上的 VSD。

VSD / SDS 类方法通常通过 pretrained diffusion teacher 的 score 或 distribution matching 来训练 student，使 student 输出落在 teacher 的生成分布中。这类方法强调 perceptual realism，但不一定保证文本字符身份正确。

VOSR 的蒸馏方式更接近：

```text
multi-step restoration teacher
        ↓
guided teacher prediction
        ↓
one-step student
```

它蒸馏的是 teacher 在 restoration-oriented guidance 下的恢复方向，而不是泛化的 T2I prior。

---

### 4.2 Guided Teacher Target

VOSR 的 distillation target 是：

```math
v_T^g = v_T^p + \omega (v_T^f - v_T^p)
```

而不是单纯的 `v_T^f`。

因此，student 学到的是：

> teacher 在 restoration-oriented guidance 下的 input-anchored generative restoration behavior。

这正是后续文本超分研究可以利用的关键点。

---

### 4.3 RC / RCGM Trajectory Compression

VOSR 比较了 shortcut-based 和 recursive-consistency-based 蒸馏方式，最终主实验采用 RC-based variant。

RC 的核心思想是：

- 不只是让 student 拟合一个单点 teacher prediction；
- 而是通过递归一致性，让 student 的大跨度一步预测更接近 teacher 的 denoising trajectory；
- 对于从多步压缩到一步的大时间跨度，RC 提供更强的 trajectory-level supervision。

VOSR 的实验结果显示：

| Method | LPIPS | MUSIQ |
|---|---:|---:|
| Teacher VOSR-0.5B-ms | 0.3069 | 68.93 |
| Shortcut-based distillation | 0.2913 | 68.21 |
| RC-based distillation | 0.2856 | 69.78 |

说明 RC-based distillation 在感知质量和结构保真之间更优。

---

## 5. 为什么 One-Step 可能对文本更好

当前实验观察是：

> VOSR one-step 在文本图像上可能比 multi-step 获得更好的文本识别效果。

可能原因包括：

### 5.1 多步采样容易累积字符漂移

多步扩散每一步都有生成自由度。对于自然图像，这有利于纹理丰富化；但对于文本，可能导致：

- `5` 被修成 `S`；
- `rn` 被修成 `m`；
- `O` 和 `0` 混淆；
- 字符边缘被补成错误但“合理”的形状。

一步模型由于没有长链式采样过程，可能减少字符漂移和幻觉累积。

---

### 5.2 Guided target 被压缩进 student

如果 student 学到的是 guided teacher target，而不是裸 teacher output，那么一步模型实际上被蒸馏成了一个更偏 input fidelity 的恢复器。

这可能解释：

> one-step 不是单纯更快，而是被 guided distillation 塑造成了更适合文本恢复的模型。

---

### 5.3 蒸馏可能抑制随机高频幻觉

文本高频不是普通纹理，而是有规则的笔画结构。  
多步扩散可能产生锐利但错误的纹理；一步蒸馏由于目标更直接，可能更稳定地保留字符结构。

---

## 6. 面向文本超分的研究方案

### 6.1 方法名候选

可以考虑以下命名：

- TextRoD-SR: Text-aware Restoration-Oriented Distillation for SR
- OCR-Guided One-Step VOSR
- Text-Guided Restoration-Oriented Distillation
- Guidance-Decomposed Text SR Distillation

---

### 6.2 总体思想

不是简单把 VOSR 用到 TextZoom，而是设计：

> 面向文本可识别性的 restoration-oriented guidance distillation。

总体框架：

```text
LR image
   │
   ├── VAE encoder → structural latent c_lq
   ├── DINO encoder → visual semantic feature c_vis
   ├── OCR encoder → text semantic feature c_ocr
   │
   ▼
Multi-step text-aware teacher
   │
   ├── full branch:
   │       c_lq + c_vis + c_ocr
   │
   ├── partial branch:
   │       α c_lq + β c_vis + dropped c_ocr
   │
   ▼
guided teacher target
   │
   ▼
one-step student / correction flow
   │
   ▼
OCR-friendly SR image
```

---

## 7. Text-Aware Restoration-Oriented Guidance

### 7.1 Full Condition

```math
c_f = \{c_{lq}, c_{vis}, c_{ocr}\}
```

其中：

- `c_lq`：LR VAE latent structural condition；
- `c_vis`：DINO / visual encoder feature；
- `c_ocr`：OCR recognizer / text encoder feature。

---

### 7.2 Partial Condition

```math
c_p = \{\alpha c_{lq}, \beta c_{vis}, 0 \cdot c_{ocr}\}
```

其中：

- `alpha`：LR structural retention factor；
- `beta`：visual semantic retention factor；
- OCR condition 在 partial branch 中可置零或随机 dropout。

建议初始设置：

```text
α ∈ [0.05, 0.25]
β = 0 或 0.3
OCR condition drop
```

---

### 7.3 Guided Prediction

```math
v_{text-cfg} = v_p + s(v_f - v_p)
```

其含义是：

> 从弱 LR restoration 方向，推向强 LR + visual semantic + OCR semantic 的文本保真恢复方向。

---

## 8. Guidance-Decomposed One-Step Distillation

为了明确验证 CFG 的作用，可以不只蒸馏最终 guided target，而是显式蒸馏 guidance residual。

### 8.1 Teacher Residual

```math
g_T = v_T^f - v_T^p
```

### 8.2 Student Residual

如果 student 输出两个分支：

```math
v_S^f, \quad v_S^p
```

则：

```math
g_S = v_S^f - v_S^p
```

### 8.3 Guided Student Prediction

```math
v_S^g = v_S^p + s(v_S^f - v_S^p)
```

---

## 9. Loss 设计

### 9.1 Guided Distillation Loss

```math
\mathcal{L}_{distill} = \|v_S^g - v_T^g\|_2^2
```

### 9.2 Branch-Level Distillation Loss

```math
\mathcal{L}_{branch}
=
\|v_S^f - v_T^f\|_2^2
+
\|v_S^p - v_T^p\|_2^2
```

### 9.3 Guidance Residual Loss

```math
\mathcal{L}_{guide}
=
\|(v_S^f - v_S^p) - (v_T^f - v_T^p)\|_1
```

### 9.4 RC Consistency Loss

```math
\mathcal{L}_{RC}
=
\|z_S^{0 \leftarrow t} - \text{stopgrad}(z_T^{0 \leftarrow t})\|_2^2
```

作用：

- 约束 one-step student 的大跨度预测；
- 让 student 轨迹更接近 multi-step teacher；
- 提升压缩稳定性。

### 9.5 OCR Recognition Loss

如果有文本标签 `y`：

```math
\mathcal{L}_{OCR}
=
\text{CTC}(R(\hat{x}_{SR}), y)
```

或对于 transformer recognizer：

```math
\mathcal{L}_{OCR}
=
\text{CE}(R(\hat{x}_{SR}), y)
```

### 9.6 OCR Feature Alignment Loss

```math
\mathcal{L}_{feat}
=
\|F_{ocr}(\hat{x}_{SR}) - F_{ocr}(x_{HR})\|_1
```

### 9.7 Edge / Stroke Consistency Loss

```math
\mathcal{L}_{edge}
=
\|E(\hat{x}_{SR}) - E(x_{HR})\|_1
```

如果有 text mask：

```math
\mathcal{L}_{edge}
=
\|m_{text} \odot (E(\hat{x}_{SR}) - E(x_{HR}))\|_1
```

### 9.8 总损失

```math
\mathcal{L}
=
\lambda_1 \mathcal{L}_{distill}
+
\lambda_2 \mathcal{L}_{branch}
+
\lambda_3 \mathcal{L}_{guide}
+
\lambda_4 \mathcal{L}_{RC}
+
\lambda_5 \mathcal{L}_{OCR}
+
\lambda_6 \mathcal{L}_{feat}
+
\lambda_7 \mathcal{L}_{edge}
+
\lambda_8 \mathcal{L}_{rec}
```

---

## 10. 与 Stage1 / Stage2 计划的对应

### 原始计划

```text
Stage 1:
    可控的一步恢复模型
    RGPA + LQFM + reconstruction / GAN

Stage 2:
    VSD + OCR-aware U-REPA
    从 Stage1 checkpoint 继续训练
```

---

### 基于 VOSR 的建议改造

## Stage 1：Text-aware Multi-step Restoration Teacher

目标：

> 训练或微调一个文本结构更稳定的 multi-step teacher。

输入：

```text
LR image x_lq
HR image x_hr
text label y
optional text bbox / mask
```

条件：

```text
c_lq  = VAE_Enc(x_lq)
c_vis = DINO(x_lq)
c_ocr = OCR_Encoder(x_lq)
```

full branch：

```text
c_lq + c_vis + c_ocr
```

partial branch：

```text
α c_lq + β c_vis + dropped c_ocr
```

Stage 1 loss：

```math
\mathcal{L}_{stage1}
=
\mathcal{L}_{fm}
+
\lambda_{ocr}\mathcal{L}_{OCR}
+
\lambda_{feat}\mathcal{L}_{feat}
+
\lambda_{edge}\mathcal{L}_{edge}
+
\lambda_{rec}\mathcal{L}_{rec}
```

---

## Stage 2：OCR-aware One-step Guidance Distillation

目标：

> 将 multi-step teacher 的 text-aware restoration guidance 蒸馏到 one-step student。

teacher target：

```math
v_T^g = v_T^p + \omega (v_T^f - v_T^p)
```

student 学习：

- guided teacher target；
- full / partial branch；
- guidance residual；
- RC trajectory consistency；
- 可选 OCR-aware loss。

---

## Stage 3：可选 Text Correction Flow

目标：

> 在 one-step student 输出基础上，增加轻量矫正模块，专门修正文字区域。

形式：

```math
z_{out} = z_{base} + \Delta z
```

其中：

```math
\Delta z = F_{corr}(z_{base}, c_{lq}, c_{vis}, c_{ocr}, m_{text})
```

用于修正：

- 笔画断裂；
- 字符边缘；
- OCR 不确定区域；
- 字符身份错误。

---

## 11. 当前本地实验设置总结

### 11.1 核心目标

当前实验目标是验证：

> one-step 文本识别率提升，是否来自 guided teacher target 和 RC trajectory compression，而不是单纯 OCR loss。

### 11.2 共同设置

三组实验使用同一个 text fine-tuned VOSR multi-step checkpoint：

```text
exp_vosr_text/.../checkpoint-00040000/clean_weights/ema_model.safetensors
```

该 checkpoint 同时作为：

```yaml
teacher_ckpt:
  冻结 teacher，产生 distillation target

pretrained_ckpt:
  student 初始化权重，从 multi-step checkpoint 继续蒸馏成 one-step
```

共同训练配置：

```yaml
模型: VOSR-0.5B
数据: configs/train_txt/text_hr_512_dataset.txt
分辨率: 512
infer_steps: 1
max_train_steps: 20000
batch: 4
gradient_accumulation_steps: 4
mixed_precision: bf16
output_dir: exp_vosr_text_distill_ablation/
```

启动方式：

```bash
bash scripts/train_text_distill_ablation.sh full_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_rc
```

连续运行：

```bash
bash scripts/train_text_distill_ablation.sh all
```

---

## 12. 当前三组实验设计

## 实验 A：Full Target Distillation

配置文件：

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_full_target_no_rc.yml
```

关键设置：

```yaml
cfg_scale: 1.0
distill_type: shortcut
u_weight: 0.0
```

目标形式：

```math
v_T^g = v_T^p + 1.0(v_T^f - v_T^p) = v_T^f
```

含义：

> 只蒸馏 full teacher prediction。

消融目的：

> 作为普通 one-step distillation baseline，验证只学 full teacher 能达到什么 OCR 效果。

---

## 实验 B：Guided Target Distillation

配置文件：

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc.yml
```

关键设置：

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
```

目标形式：

```math
v_T^g = v_T^p + 0.5(v_T^f - v_T^p)
```

消融目的：

> 单独验证 guided teacher target 是否比 full teacher target 更适合 one-step 文本超分。

关键比较：

```text
B vs A
```

如果：

```text
B OCR Acc > A OCR Acc
```

说明：

> guided target 本身有价值。

---

## 实验 C：Guided Target + RC

配置文件：

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_rc.yml
```

关键设置：

```yaml
cfg_scale: 0.5
distill_type: rcgm
u_weight: 1.0
rcgm_delta_t: 0.01
rcgm_n_steps: 2
```

目标形式：

```math
v_T^g = v_T^p + 0.5(v_T^f - v_T^p)
```

但额外加入 RC / RCGM consistency。

消融目的：

> 验证 one-step 模型文本优势是否来自 trajectory compression 的递归一致性约束。

关键比较：

```text
C vs B
```

如果：

```text
C OCR Acc > B OCR Acc
```

说明：

> RC 对 one-step 文本可识别性有额外贡献。

---

## 暂未实现实验 D：Guided Target + RC + OCR Loss

目标设置：

```yaml
target: v_T^g
RC: yes
OCR loss: yes
```

消融目的：

> 判断 OCR loss 是锦上添花，还是 OCR 提升的主要来源。

当前未实现原因：

1. dataset 尚未输出 text label；
2. 缺 frozen OCR recognizer；
3. 缺 OCR preprocessing；
4. 缺 CTC / CE loss；
5. 缺 OCR loss 权重配置。

当前不创建“看起来有 OCR loss 但实际不生效”的 YAML 是合理的。

---

## 13. 当前实验设置评价

### 13.1 整体评价

当前 A/B/C 三组实验设计是合理的，优点包括：

1. teacher 和 student 初始化一致；
2. 变量控制较清楚；
3. A/B/C 逐步验证 full target、guided target、RC；
4. 暂时不加 OCR loss，可以避免把机制贡献和 OCR 监督混在一起；
5. 后续 D 可以自然验证 OCR loss 是否只是锦上添花。

---

### 13.2 需要注意的问题一：A vs B 的解释

实验 A 中：

```math
v_T^g = v_T^f
```

实验 B 中：

```math
v_T^g = v_T^p + 0.5(v_T^f-v_T^p)
```

所以 A/B 的比较实际是在比较：

```text
omega = 1.0 的 full target
vs
omega = 0.5 的 guided target
```

因此更严谨的表述是：

> omega=0.5 的 guided teacher target 是否比 omega=1.0 的 full teacher target 更适合文本 one-step 蒸馏。

而不是绝对地说：

> guided residual 一定有效。

---

### 13.3 建议补充实验 A2：Partial Target

建议增加一个低成本实验：

```yaml
cfg_scale: 0.0
distill_type: shortcut
u_weight: 0.0
```

此时：

```math
v_T^g = v_T^p
```

这样就形成：

| ID | Target | omega |
|---|---|---:|
| A2 | partial target \(v_T^p\) | 0.0 |
| B | guided target | 0.5 |
| A | full target \(v_T^f\) | 1.0 |

如果结果是：

```text
B > A and B > A2
```

则可以说明：

> 文本 one-step 蒸馏不是越靠 full 越好，也不是越靠 partial 越好，而是存在更适合 OCR 的中间 guided target 区间。

---

### 13.4 建议补充实验 B2：Guidance Scale Sweep

如果算力允许，可加入：

```yaml
cfg_scale: 0.25
cfg_scale: 0.75
```

最终形成：

| Target | omega |
|---|---:|
| partial | 0.0 |
| guided-0.25 | 0.25 |
| guided-0.5 | 0.5 |
| guided-0.75 | 0.75 |
| full | 1.0 |

用于画出：

```text
distillation-time guidance weight vs OCR Acc / NED
```

这条曲线会非常有说服力。

---

### 13.5 需要注意的问题二：B vs C 的解释

B 使用：

```yaml
distill_type: shortcut
u_weight: 0.0
```

C 使用：

```yaml
distill_type: rcgm
u_weight: 1.0
```

因此 C vs B 不只差一个 RC loss，可能还差：

- 代码路径；
- target 构造方式；
- teacher 调用方式；
- intermediate rollout；
- timestep 采样方式。

因此更稳妥的表述是：

> 在相同 guided target 下，RCGM-style trajectory compression 比 no-RC shortcut baseline 更适合 one-step 文本超分。

如果想更严谨，可以加 C0：

```yaml
cfg_scale: 0.5
distill_type: rcgm
u_weight: 0.0
```

比较：

```text
C0 vs B:
    RCGM code path / recipe 差异

C vs C0:
    RC consistency loss 的真实贡献
```

---

## 14. 最终结果表设计

当前结果表：

| ID | 方法 | Target | RC | OCR Loss | 关键比较 |
|---|---|---|---|---|---|
| A | Full target | \(v_T^f\) | no | no | baseline |
| B | Guided target | \(v_T^g\) | no | no | B vs A |
| C | Guided target + RC | \(v_T^g\) | yes | no | C vs B |
| D | Guided target + RC + OCR | \(v_T^g\) | yes | yes | D vs C |

建议增强版结果表：

| ID | 方法 | Target | Distill type | RC weight | OCR Loss | 关键比较 |
|---|---|---|---|---:|---|---|
| A2 | Partial target | \(v_T^p\) | shortcut | 0 | no | A2/B/A sweep |
| A | Full target | \(v_T^f\) | shortcut | 0 | no | baseline |
| B | Guided target | \(v_T^g, omega=0.5\) | shortcut | 0 | no | B vs A |
| C0 | Guided target + RCGM path | \(v_T^g\) | rcgm | 0 | no | C0 vs B |
| C | Guided target + RC | \(v_T^g\) | rcgm | 1 | no | C vs C0 / B |
| D | Guided target + RC + OCR | \(v_T^g\) | rcgm | 1 | yes | D vs C |

---

## 15. 指标设计

建议最终指标包括：

### 文本识别指标

- Word Acc
- Char Acc
- Normalized Edit Distance
- CRNN Acc
- ASTER Acc
- MORAN Acc
- PARSeq / ABINet Acc
- 多 OCR recognizer 平均 Acc

### 图像质量指标

- PSNR
- SSIM
- LPIPS
- DISTS
- NIQE
- MUSIQ
- MANIQA

### 效率指标

- Runtime
- GPU memory
- inference steps
- FPS

### 结构指标

- Text-region LPIPS
- Edge F-score
- Stroke consistency
- Text mask region PSNR / SSIM

---

## 16. 评估协议建议

### 16.1 固定推理设置

A/B/C 必须使用相同推理设置：

```text
infer_steps = 1
same seed
same VAE decoder
same align_method
same input resize / crop
same checkpoint step
same EMA / non-EMA choice
```

否则 OCR 差异可能来自推理流程。

---

### 16.2 多 checkpoint 评估

不要只评估最后 20000 step。建议保存并评估：

```text
5000
10000
15000
20000
```

观察：

```text
training step vs OCR Acc
```

原因：

- guided target 可能中期最好；
- RC 收敛可能更慢；
- full target 后期可能追上；
- 训练后期可能图像指标更好但 OCR 变差。

---

### 16.3 多 OCR recognizer 评估

建议至少使用：

```text
CRNN
ASTER
MORAN
PARSeq / ABINet
```

如果只用单一 OCR，结果可能偏向某个 recognizer 的识别偏好。

---

### 16.4 加 teacher 上限对照

建议评估：

| 模型 | 说明 |
|---|---|
| Teacher-ms-25step | text fine-tuned multi-step teacher |
| Teacher-ms-1step | 如果代码允许，直接 1 step sampling |
| A/B/C student | 三个 one-step student |

这样可以判断：

- student 是否超过 teacher；
- student 是否只是接近 teacher；
- one-step 是否真的比 multi-step 更 OCR-friendly。

---

## 17. 理想证据链

理想结果：

```text
B > A:
    omega=0.5 guided teacher target 比 full teacher target 更适合 one-step 文本蒸馏。

C > B:
    在相同 guided target 下，RCGM trajectory compression 进一步提升文本结构稳定性。

D > C:
    OCR loss 可以进一步增强文本可识别性，但如果 A/B/C 已有差异，则 one-step 优势不能完全归因于 OCR loss。
```

更完整证据链：

```text
A2 < B > A:
    存在 OCR-friendly 的中间 guidance target。

C > B:
    trajectory-level compression 对文本结构有额外贡献。

D > C:
    OCR loss 是增强项，而不是全部原因。
```

---

## 18. 不同实验结果的解释

### 情况 1：B > A，C > B

最理想。

说明：

- guided target 有效；
- RC 进一步有效；
- 无需 OCR loss 也能提升文本识别。

---

### 情况 2：B > A，但 C ≈ B

说明：

- guided target 是主要因素；
- RC 对文本提升不明显；
- RC 可能主要改善图像质量或训练稳定性。

---

### 情况 3：B ≈ A，但 C > B

说明：

- guided target 单独不明显；
- trajectory compression 是关键；
- 需要进一步分析 teacher trajectory 和字符漂移。

---

### 情况 4：A ≈ B ≈ C

说明当前机制没有明显影响。可能原因：

1. teacher text fine-tune 不够强；
2. 20000 steps 不够；
3. cfg_scale=0.5 不是最佳；
4. TextZoom 退化和 VOSR degradation 不匹配；
5. OCR 评估噪声较大；
6. full / partial target 在文本区域差异不够大。

应补做：

```text
cfg_scale sweep: 0.0 / 0.25 / 0.5 / 0.75 / 1.0
```

---

### 情况 5：A > B

说明：

- full branch 对文本更可靠；
- partial branch 可能引入过多生成自由度；
- 文本 SR 可能需要 stronger LR / OCR anchoring。

可以尝试：

```text
cfg_scale = 1.2 / 1.5
```

看是否更强 fidelity 有利于 OCR。

---

## 19. 后续实验建议

### 19.1 当前优先级

优先跑：

```bash
bash scripts/train_text_distill_ablation.sh full_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_rc
```

即 A/B/C。

---

### 19.2 强烈建议补充

补 A2：

```yaml
cfg_scale: 0.0
distill_type: shortcut
u_weight: 0.0
```

作用：

> 形成 partial / guided / full 三点对照。

---

### 19.3 有算力时补充

补 C0：

```yaml
cfg_scale: 0.5
distill_type: rcgm
u_weight: 0.0
```

作用：

> 区分 RCGM 代码路径和 RC consistency loss 的真实贡献。

---

## 20. 可写成论文的核心创新点

### 创新点 1：Text-aware Restoration-Oriented CFG

将 VOSR 的 partial conditioning 扩展到文本超分：

```text
full branch:
    LR structure + visual semantic + OCR semantic

partial branch:
    weak LR structure + weak/no visual semantic + no OCR semantic
```

核心：

> 构造从弱恢复到文本保真恢复的 guidance direction。

---

### 创新点 2：Guidance-Decomposed One-Step Distillation

不是只蒸馏 final guided prediction，而是显式蒸馏：

- full branch；
- partial branch；
- guidance residual；
- guided prediction。

核心：

> 让 one-step student 学到 restoration-oriented guidance field。

---

### 创新点 3：OCR-aware Text Preservation

将 OCR recognizer 用于：

- OCR recognition loss；
- OCR feature alignment；
- stroke / edge consistency；
- correction flow。

核心：

> 让 one-step SR 模型直接对文本可识别性负责。

---

## 21. 可写进论文的 Method 草稿

### Motivation

Existing one-step diffusion SR methods mainly focus on accelerating iterative denoising, but for scene text image super-resolution, acceleration alone is insufficient. Text restoration is highly sensitive to character identity and stroke structure. We observe that conventional score distillation or VSD-style training does not necessarily improve text readability, while restoration-oriented guidance can significantly affect OCR-related metrics. This motivates us to investigate whether the guidance behavior itself can be distilled into a one-step text SR model.

### Text-aware Restoration-Oriented Guidance

Given an LR image `x_LR`, we extract three complementary conditions: a spatial structural latent `c_lq` from a VAE encoder, a visual semantic feature `c_vis` from a pretrained vision encoder, and a text-aware semantic feature `c_ocr` from a frozen OCR recognizer. The full condition is defined as:

```math
c_f = \{c_{lq}, c_{vis}, c_{ocr}\}
```

To construct an input-anchored auxiliary branch, we define a partial condition:

```math
c_p = \{\alpha c_{lq}, \beta c_{vis}, 0 \cdot c_{ocr}\}
```

The teacher predictions under full and partial conditions are denoted as `v_T^f` and `v_T^p`. We obtain the text-aware guided teacher prediction:

```math
v_T^g = v_T^p + \omega(v_T^f - v_T^p)
```

Unlike standard CFG, the auxiliary branch is not unconditional but weakly anchored to the LR input. Thus, the guidance residual represents a restoration-oriented direction from weakly conditioned restoration to text-aware faithful restoration.

### Guidance-Decomposed One-Step Distillation

Instead of directly distilling the final output of the teacher, we explicitly distill the restoration-oriented guidance field. The student predicts both full and partial velocities, `v_S^f` and `v_S^p`, and forms:

```math
v_S^g = v_S^p + s(v_S^f - v_S^p)
```

The distillation objective contains a guided prediction loss:

```math
\mathcal{L}_{distill} = \|v_S^g - v_T^g\|_2^2
```

a branch-level loss:

```math
\mathcal{L}_{branch}
=
\|v_S^f - v_T^f\|_2^2
+
\|v_S^p - v_T^p\|_2^2
```

and a guidance residual loss:

```math
\mathcal{L}_{guide}
=
\|(v_S^f - v_S^p) - (v_T^f - v_T^p)\|_1
```

To improve large temporal compression, we further adopt recursive consistency regularization following the teacher trajectory.

### OCR-aware Text Preservation Loss

Given the predicted SR image `x_SR`, a frozen OCR recognizer `R`, and the text label `y`, we use:

```math
\mathcal{L}_{OCR}
=
\text{CTC}(R(\hat{x}_{SR}), y)
```

or cross-entropy loss depending on the recognizer. We also align intermediate OCR features:

```math
\mathcal{L}_{feat}
=
\|F_{ocr}(\hat{x}_{SR}) - F_{ocr}(x_{HR})\|_1
```

Finally, to preserve stroke structures, we add a text-region edge loss:

```math
\mathcal{L}_{edge}
=
\|m_{text} \odot (E(\hat{x}_{SR}) - E(x_{HR}))\|_1
```

The overall objective is:

```math
\mathcal{L}
=
\lambda_1 \mathcal{L}_{distill}
+
\lambda_2 \mathcal{L}_{branch}
+
\lambda_3 \mathcal{L}_{guide}
+
\lambda_4 \mathcal{L}_{RC}
+
\lambda_5 \mathcal{L}_{OCR}
+
\lambda_6 \mathcal{L}_{feat}
+
\lambda_7 \mathcal{L}_{edge}
```

---

## 22. 最终总结

当前实验设计的基本判断：

> A/B/C 是合理的，可以先跑。

其研究价值在于：

1. 不引入 OCR loss，先验证 guided target 和 RC 是否本身就影响 OCR；
2. 用同一个 text fine-tuned multi-step checkpoint 同时作为 teacher 和 student initialization，变量控制较干净；
3. A/B/C 分别对应 full target、guided target、guided target + RC，能够形成初步因果链；
4. 后续 D 再加入 OCR loss，可以判断 OCR loss 是锦上添花还是主要来源。

最推荐的补充是：

```text
A2: cfg_scale=0.0 partial target
C0: cfg_scale=0.5, rcgm, u_weight=0.0
```

如果最终结果满足：

```text
A2 < B > A
C > B
D > C
```

则可以形成非常完整的论文证据链：

> 文本 one-step 优势不是简单来自 OCR loss，而是来自 restoration-oriented guided teacher target、RC trajectory compression，以及后续 OCR-aware supervision 的共同作用。

最终研究主线可以概括为：

> 将 VOSR 的 restoration-oriented guidance 从通用 SR 扩展到文本超分，通过 guided teacher target、guidance residual distillation 和 RC trajectory compression，把多步 teacher 中更适合文本结构恢复的行为蒸馏到 one-step student，从而实现 OCR-friendly one-step text super-resolution。
