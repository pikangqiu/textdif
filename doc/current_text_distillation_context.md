# 当前文本超分蒸馏研究上下文

更新时间：2026-05-11

本文档用于给之后继续接手本仓库的 agent / researcher 快速同步上下文。它不是普通项目 README，而是记录当前研究目的、已经完成的实验、代码改动、阶段结论和下一步方向。

## 1. 当前研究目的

本项目当前不是单纯复现 VOSR，也不是单纯做 OCR loss。当前核心研究问题是：

> 在 VOSR 的 one-step restoration distillation 框架下，是否可以通过更合适的 teacher target construction，使 one-step 文本超分模型获得更好的文字可识别性？

更具体地说，研究从 VOSR 中观察到一个现象：

> text fine-tuned multi-step teacher 蒸馏成 one-step student 后，OCR 指标反而可能优于蒸馏前的 multi-step teacher。

因此当前想研究的是：

1. **是什么蒸馏机制在提升文字效果。**
   - 是 guided teacher target 起作用？
   - 是 RC / RCGM trajectory compression 起作用？
   - 是 one-step 模型减少多步采样字符漂移？
   - 还是 DiT / DINO 条件结构本身带来的优势？

2. **这种蒸馏方式还能不能进一步提升文字效果。**
   - 不急着直接上 OCR recognition loss。
   - 更希望先验证一种通用的 text-aware / representation-aware distillation 方式。
   - 目前下一步重点是基于 B baseline 加入 DINO-REPA 和 OCR-REPA。

3. **这种发现是否具有可迁移性。**
   - 之前讨论过 OSEDiff / VSD。
   - 需要注意：OSEDiff 的 `cfg_vsd` 和 VOSR 的 restoration-oriented CFG 不是同一种设计，不能简单用扫 `cfg_vsd` 来对应 VOSR A/B。
   - 若迁移到 VSD，应研究 VSD teacher target / score target 的构造方式，而不是只改 CFG 数值。

## 2. 关键概念：VOSR 的 CFG 不是传统 CFG

VOSR 中的 guided target 来自 restoration-oriented partial conditioning：

```text
full branch:
    strong LR structural condition + DINO visual semantic condition

partial branch:
    weak LR structural condition + dropped DINO visual condition
```

teacher target 构造为：

```text
v_T^g = v_T^p + omega * (v_T^f - v_T^p)
```

这里的 `omega` 在配置中对应 `cfg_scale`。这不是传统 diffusion unconditional CFG，也不是 OSEDiff 里的 `cfg_vsd`。它的含义是：

> 从弱条件 restoration 方向，推向强条件 restoration 方向。

因此当前 A/B/C 的 `cfg_scale` 不是普通“调 guidance 强度”的实验，而是在测试不同 teacher target construction 对 one-step student 的 OCR 效果。

## 3. 当前主实验：A/B/C 文本蒸馏消融

所有 A/B/C 都使用同一个 text fine-tuned 0.5B multi-step checkpoint：

```text
exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000/clean_weights/ema_model.safetensors
```

该 checkpoint 同时作为：

- frozen teacher：产生 distillation target；
- student initialization：从 multi-step text teacher 继续蒸馏成 one-step。

| ID | 实验含义 | `cfg_scale` | `distill_type` | `u_weight` | RC/RCGM | 目的 |
|---|---|---:|---|---:|---|---|
| Teacher MS | 蒸馏前 multi-step teacher | 推理 0.5 | FM | - | 否 | baseline |
| A | full target no RC | 1.0 | shortcut | 0.0 | 否 | 直接学习 `v_T^f` |
| B | guided target no RC | 0.5 | shortcut | 0.0 | 否 | 学习 guided target `v_T^g` |
| C | guided target + RC | 0.5 | rcgm | 1.0 | 是 | 测试 RC/RCGM 对 OCR 是否有益 |

重要解释：

- A/B 虽然 `distill_type=shortcut`，但 `u_weight=0.0`，所以实际只有 direct teacher-student target matching，即 `v_loss`。
- C 和 B 的前半部分相同，都是 guided teacher target；区别是 C 额外启用 RCGM consistency。
- C 的目标不是“更强 B”，而是测试原始 VOSR 中有利于通用 SR 的 RC/RCGM 是否也有利于 OCR。

## 4. 已完成结果

评测集：847 个有效图像对，1945 个 OCR regions。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | OCR-EM ↑ | OCR-CER ↓ | OCR-CharAcc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher MS | **24.8620** | 0.7883 | 0.2692 | 0.2087 | 6.6045 | **54.9036** | 0.4040 | 0.5620 | 0.3622 | 0.6378 |
| A full target | 24.5173 | 0.8329 | **0.2538** | 0.2189 | 10.7384 | 53.4998 | 0.4156 | 0.5789 | 0.3418 | 0.6582 |
| B guided target | 24.6187 | **0.8334** | 0.2541 | 0.2200 | 10.9392 | 54.7819 | **0.4295** | **0.5841** | **0.3401** | **0.6599** |
| C guided + RC | 24.2763 | 0.7864 | 0.2550 | **0.2078** | **6.5167** | 54.8537 | 0.3819 | 0.5548 | 0.3599 | 0.6401 |

阶段性结论：

1. **A/B 相比 Teacher MS 提升了 OCR。**
   - 说明 one-step distillation 在当前文本超分任务中不只是加速，也可能减少多步采样的字符漂移。

2. **B 是当前 OCR 最好的设置。**
   - B 相比 A 的 OCR-EM 更高，OCR-CER 更低。
   - 支持“guided teacher target 比 full teacher target 更适合文本 OCR 蒸馏”。

3. **C 改善部分感知/无参考指标，但损害 OCR。**
   - C 的 DISTS、NIQE 更好，但 OCR-EM / CER / CharAcc 明显差于 B。
   - 当前不应把原始 VOSR 的 RC/RCGM recipe 直接当作文本 OCR 最优设置。

4. **当前最合理主线不是“所有蒸馏都提升文字”，而是：**
   - teacher target construction 对 one-step text SR 的 OCR fidelity 很关键；
   - text-aware teacher signal + guided target + one-step student 可能共同产生文字提升。

## 5. 收敛性检查

A/B/C 都完整训练到 `checkpoint-00020000`：

```text
A: 20 checkpoints, checkpoint-00001000 ... checkpoint-00020000
B: 20 checkpoints, checkpoint-00001000 ... checkpoint-00020000
C: 20 checkpoints, checkpoint-00001000 ... checkpoint-00020000
```

现有 TensorBoard 标量只记录 `loss/lr/v_loss`。注意当前代码中 `v_loss` 实际记录为 `loss.item()`：

- A/B 因为 `u_weight=0`，`loss == v_loss`。
- C 中日志里的 `v_loss` 实际是总 loss，不是单独的 teacher-student velocity loss。

loss 窗口均值：

| 实验 | 0-1k | 1k-5k | 5k-10k | 10k-15k | 15k-20k | 19k-20k |
|---|---:|---:|---:|---:|---:|---:|
| A loss | 8.33e-5 | 9.79e-5 | 1.14e-4 | 1.26e-4 | 1.30e-4 | 1.28e-4 |
| B loss | 1.53e-3 | 1.54e-3 | 1.45e-3 | 1.34e-3 | 1.43e-3 | 1.40e-3 |
| C total loss | 5.05e-1 | 3.96e-1 | 4.30e-1 | 4.02e-1 | 3.84e-1 | 3.73e-1 |

判断：

- 不能把 A/B/C 效果简单归因于“20k 没收敛”。
- A/B 没有明显持续下降趋势。
- C 的 total loss 仍在学习一致性项，但 OCR 已经差于 B，更像目标方向与 OCR 不一致。
- 若要进一步确认，优先补测 B/C 的 `10k/15k/20k`，而不是盲目延长所有实验。

## 6. 当前代码实现地图

### 6.1 `vosr.py`

关键位置：

- `VOSR.__init__`：保存 `u_weight/cfg_scale/cfg_ratio/a/b/t_start/t_end` 等蒸馏参数。
- `_prepare_cfg_conditions_distill()`：构造 distillation 用的 full / weak conditions。
- `_teacher_target()`：构造 teacher guided target：

```python
v = v_weak_pred + omega * (v_cond - v_weak_pred)
```

- `loss_fm_distill_shortcut_improved()`：
  - 先构造 guided teacher target；
  - student 拟合 `v_current -> v`；
  - 还计算 shortcut `u_loss`，但 A/B 中 `u_weight=0`，所以不生效。
- `loss_fm_distill_rcgm_improved()`：
  - 和 B 一样先计算 guided `v_loss`；
  - 额外通过 `_rcgm_consistency()` 计算 RCGM consistency；
  - C 中 `loss = v_loss + u_loss`。

### 6.2 `train_vosr_distill.py`

关键位置：

- 加载 DINOv2 作为 VOSR visual encoder。
- 构造 frozen teacher `model_tea`，无 `auxiliary_time_cond`。
- 构造 one-step student `model`，有 `auxiliary_time_cond=True`。
- 训练时在线 RealESRGAN 退化 HQ 得到 LQ。
- 通过 VAE encoder 得到 latent。
- 根据 `args.distill_type` 调用：
  - `loss_fm_distill_shortcut_improved`
  - `loss_fm_distill_rcgm_improved`
- 目前已经加入 REPA 支持：
  - `repa_type: dino`
  - `repa_type: ocr`

已知日志问题：

```python
logs["v_loss"] = loss.item()
```

这会让 C 的 `v_loss` 不是纯 `v_loss`，后续应改成分别记录 `v_loss/u_loss/repa_loss`。

### 6.3 `models/lightningdit.py`

已经支持：

```python
forward(..., return_hidden_at=None)
forward_flexible(..., return_hidden_at=None)
```

当 `return_hidden_at` 不为空时，返回：

```python
(model_output, hidden_tokens)
```

这是 REPA 对齐 student DiT hidden tokens 的基础。

### 6.4 `models/repa_align.py`

新增：

- `RepaProjector`
- `repa_cosine_loss`

第一版 REPA 使用 global pooled cosine alignment：

```text
student hidden tokens -> projector -> mean pool -> normalize
target encoder tokens -> mean pool -> normalize
loss = 1 - cosine
```

选择 global pooling 是为了避免 DiT token 数与 DINO/OCR token 数不一致。

## 7. REPA 方向当前进度

当前计划是在 B baseline 上加入 representation alignment：

```text
L = L_guided_distill + lambda_repa * L_repa
```

已添加配置：

```text
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml
```

### D：B + DINO-REPA

关键设置：

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
repa_type: dino
repa_weight: 0.5
repa_layer: 13
repa_dino_layer: 8
repa_target_dim: 768
```

当前本地已有 checkpoint：

```text
checkpoint-00001000
checkpoint-00002000
checkpoint-00003000
checkpoint-00004000
```

早期曾遇到过：

```text
AttributeError: 'VOSR' object has no attribute 'a'
```

原因是 `VOSR.__init__` 中 `self.a/self.b` 初始化缺失或被移动到了不可达位置；现在已修复。

### E：B + OCR-REPA

关键设置：

```yaml
cfg_scale: 0.5
distill_type: shortcut
u_weight: 0.0
repa_type: ocr
repa_ocr_model: microsoft/trocr-base-printed
repa_weight: 0.5
repa_layer: 13
```

当前配置和代码已存在，但本地没有看到对应实验目录。若服务器不能联网下载 HuggingFace 模型，需要把 `repa_ocr_model` 改成本地 OCR encoder 路径。

## 8. 运行脚本

训练 A/B/C：

```bash
bash scripts/train_text_distill_ablation.sh full_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_no_rc
bash scripts/train_text_distill_ablation.sh guided_target_rc
bash scripts/train_text_distill_ablation.sh all
```

训练 REPA：

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr
```

推理 A/B/C：

```bash
GPUS=0,1,2 STEP=00020000 bash scripts/infer_text_distill_ablation_multigpu.sh
```

推理蒸馏前 teacher：

```bash
GPU=0 bash scripts/infer_text_teacher_0.5b_step40000.sh
```

通用多实验推理调度：

```bash
python scripts/infer_experiments_multigpu.py --include '*text_ablation*' --checkpoint latest --gpus 0,1,2 --force-rerun
```

## 9. 关于 OSEDiff / VSD 的当前理解

用户之前在 OSEDiff 上做过 VSD，两阶段之间文字效果没有明显区别。这个观察不能直接否定 VOSR 的蒸馏效果，因为二者蒸馏信号不同：

| 项 | VOSR guided distillation | OSEDiff VSD |
|---|---|---|
| teacher | text fine-tuned restoration flow teacher | frozen diffusion teacher / score provider |
| target | guided velocity target `v_T^g` | score / x0 gradient target |
| CFG 设计 | restoration-oriented partial/full condition | standard diffusion CFG in VSD |
| student | one-step DiT restoration model | one-step SR generator / LoRA |
| 当前观察 | B 提升 OCR | 之前 VSD 两阶段文字无明显差异 |

重要约束：

> OSEDiff 的 `cfg_vsd` 和 VOSR 的 `cfg_scale` 不是同一机制。不能用简单扫 OSEDiff `cfg_vsd` 来声称复现 VOSR A/B/C。

如果后续要迁移到 OSEDiff，应该研究：

```text
VSD teacher score / x0 target construction
```

而不是只改 CFG 参数。

## 10. 当前最重要的研究解释

当前最稳妥的解释不是“DiT 架构单独导致文字提升”，也不是“所有蒸馏都会提升文字”。

更合理的解释是：

> 在 VOSR 的 one-step flow distillation 框架中，text-aware multi-step teacher 提供了有文本恢复能力的 teacher signal；guided teacher target 为 one-step student 提供了更适合学习的中间 restoration direction；one-step 采样减少了多步采样中的字符漂移和高频幻觉。因此 B 在 OCR 上优于 multi-step teacher 和 full-target A。

DiT / DINO 可能是有利承载结构：

- DiT token 表征适合全局结构和后续 REPA 对齐；
- DINOv2 visual condition 为 restoration-oriented guidance 提供语义特征；
- 但 A/B/C 在同一 DiT 架构下差异明显，因此当前证据更支持“target construction 是主要变量”。

## 11. 下一步建议

优先级从高到低：

1. **继续 DINO-REPA 训练到可比较 checkpoint。**
   - 至少评估 `10k/20k`，若资源允许到 `20k`。

2. **启动 OCR-REPA，但先确认 OCR encoder 本地可用。**
   - 如果 HuggingFace 不能联网，先准备本地 TrOCR 或其他 transformer OCR encoder。

3. **补测 B/C 中间 checkpoint。**
   - 评估 `10k/15k/20k`，判断 B 是否平台化，C 是否后期被 RC 拉偏。

4. **修改日志记录。**
   - 把 `v_loss`、`u_loss/rc_loss`、`repa_loss` 分开记录。
   - 这对解释机制非常重要。

5. **谨慎设计跨模型验证。**
   - OSEDiff 可作为后续验证对象，但不能简单调 `cfg_vsd`。
   - 需要设计与 VOSR “teacher target construction”同构的 VSD target ablation。

## 12. 已知未完成 / 风险点

- OCR-REPA 依赖 HuggingFace OCR 模型，可能受网络或模型结构兼容性影响。
- 当前 REPA 第一版是 global pooled cosine，对文本局部区域可能不够敏感。
- 当前数据训练仍是 RealESRGAN 在线退化，不一定完全匹配真实文本退化。
- A/B/C 的训练日志没有中间 OCR 评估曲线。
- C 的 `v_loss` 日志命名不准确，机制分析需要更细粒度日志。
- 当前还没有 text mask / bbox 加权 distillation。
- 还没有证明该现象与 DiT 架构无关，只能说当前证据更支持 target construction。

