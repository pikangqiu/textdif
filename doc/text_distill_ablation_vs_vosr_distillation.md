# 文本蒸馏消融实验与原始 VOSR 蒸馏设置的关系

## 1. 文档目的

本文档总结当前 A/B/C 文本蒸馏消融实验与原始 VOSR one-step
distillation 设置之间的关系，并分析当前实验结果能够支持的结论。

当前实验的目的不是简单复现 VOSR 的 one-step 蒸馏，而是进一步分析：

> 对文本超分和 OCR 可识别性而言，one-step 蒸馏的提升主要来自
> guidance-aware teacher target，还是来自原始 VOSR 中使用的
> RC/RCGM-style trajectory compression？

换句话说，这组实验想回答：

> 文本超分的一步蒸馏是否需要完全沿用 VOSR 原始的 RC 蒸馏 recipe，
> 还是需要针对文本可识别性重新选择更合适的 teacher target？

---

## 2. 实验设置对比

当前 A/B/C 三组实验都使用同一个 text fine-tuned multi-step teacher：

```text
exp_vosr_text/.../checkpoint-00040000/clean_weights/ema_model.safetensors
```

该 checkpoint 同时作为：

- 冻结 teacher，用来产生蒸馏 target；
- student 初始化权重，用于从 multi-step checkpoint 继续蒸馏成 one-step。

| 模型 / 实验 | Teacher / Init | 蒸馏 target | `cfg_scale` / omega | Distill type | `u_weight` | RC / RCGM |
|---|---|---|---:|---|---:|---|
| Teacher MS | text fine-tuned 0.5B multi-step | 无 | 推理时 0.5 | multi-step FM | 无 | 否 |
| A | 同一 teacher/init | full teacher target `v_T^f` | 1.0 | shortcut | 0.0 | 否 |
| B | 同一 teacher/init | guided target `v_T^p + 0.5(v_T^f-v_T^p)` | 0.5 | shortcut | 0.0 | 否 |
| C | 同一 teacher/init | guided target `v_T^p + 0.5(v_T^f-v_T^p)` | 0.5 | rcgm | 1.0 | 是 |
| 原始 VOSR OS | VOSR multi-step teacher | guided target | 论文/默认设置 | RC-based variant | 启用 | 是 |

需要注意：

- A/B/C 是面向文本超分的受控消融实验；
- 原始 VOSR one-step distillation 主要面向通用图像超分质量；
- B 并不是原始 VOSR 最终 recipe 的直接复现，而是一个去掉 RC 后的
  text-oriented guided-target baseline。

---

## 3. 实验结果表

评测集：847 个有效图像对，1945 个 OCR regions。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | OCR-EM ↑ | OCR-CER ↓ | OCR-CharAcc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher MS | **24.8620** | 0.7883 | 0.2692 | 0.2087 | 6.6045 | **54.9036** | 0.4040 | 0.5620 | 0.3622 | 0.6378 |
| A: full target | 24.5173 | 0.8329 | **0.2538** | 0.2189 | 10.7384 | 53.4998 | 0.4156 | 0.5789 | 0.3418 | 0.6582 |
| B: guided target | 24.6187 | **0.8334** | 0.2541 | 0.2200 | 10.9392 | 54.7819 | **0.4295** | **0.5841** | **0.3401** | **0.6599** |
| C: guided + RC | 24.2763 | 0.7864 | 0.2550 | **0.2078** | **6.5167** | 54.8537 | 0.3819 | 0.5548 | 0.3599 | 0.6401 |

---

## 4. 主要观察

### 4.1 One-step distillation 相比 multi-step teacher 提升了 OCR

与蒸馏前的 text fine-tuned multi-step teacher 相比，A 和 B 都提升了 OCR：

| 对比 | OCR-EM Δ | OCR-CER Δ | OCR-CharAcc Δ |
|---|---:|---:|---:|
| A - Teacher | +0.0169 | -0.0204 | +0.0204 |
| B - Teacher | **+0.0221** | **-0.0221** | **+0.0221** |
| C - Teacher | -0.0072 | -0.0023 | +0.0023 |

这说明 one-step distillation 在当前文本超分任务中不只是加速采样。
它还可能减少 multi-step sampling 过程中的字符漂移和高频幻觉，从而提升
OCR 稳定性。

Teacher MS 的 PSNR 和 MUSIQ 最高，但 OCR 并不是最高。这说明对于文本超分，
传统图像质量指标或无参考质量指标并不能完全代表字符可识别性。

---

### 4.2 Guided target 比 full target 更适合文本蒸馏

B 相比 A 在主要 OCR 指标上更好：

| 对比 | OCR-EM Δ | OCR-CER Δ | OCR-CharAcc Δ |
|---|---:|---:|---:|
| B - A | +0.0052 | -0.0017 | +0.0017 |

A 的设置等价于：

```math
v_T^g = v_T^p + 1.0(v_T^f - v_T^p) = v_T^f
```

也就是直接蒸馏 full teacher prediction。

B 的设置为：

```math
v_T^g = v_T^p + 0.5(v_T^f - v_T^p)
```

也就是在 partial branch 和 full branch 之间选择一个中间 guided target。

当前结果说明：

> 对文本 one-step 蒸馏而言，直接学习 full teacher target 并不是最优。
> 一个适中的 restoration-guided target 更有利于 OCR fidelity。

这支持当前方法故事的核心：

> 文本提升来自更合适的 guidance-aware distillation target，而不是 OCR loss。

---

### 4.3 RC / RCGM 改善部分感知指标，但损害 OCR

C 相比 B 在 DISTS 和 NIQE 上明显更好，MUSIQ 也略高：

| 对比 | DISTS Δ | NIQE Δ | MUSIQ Δ |
|---|---:|---:|---:|
| C - B | -0.0122 | -4.4225 | +0.0718 |

但是 C 在 SSIM 和 OCR 指标上明显更差：

| 对比 | SSIM Δ | OCR-EM Δ | OCR-CER Δ | OCR-CharAcc Δ |
|---|---:|---:|---:|---:|
| C - B | -0.0470 | -0.0293 | +0.0198 | -0.0198 |

这说明 RC/RCGM-style trajectory compression 在当前文本超分设置下并没有提升
文字可识别性。它更像是在改善某些无参考或感知分布指标，但会破坏字符身份、
笔画结构或局部文本保真。

因此 C 的结果可以作为一个有意义的负结果：

> 原始 VOSR 中对通用图像 SR 有利的 RC/RCGM trajectory compression，
> 并不能直接迁移为文本 OCR 最优的蒸馏方式。

---

## 5. 与原始 VOSR 蒸馏设置的关系

原始 VOSR 论文中的 one-step distillation 证明了 RC-based distillation
可以提升通用图像超分的感知质量和一步采样性能。

但当前文本实验显示：

| 维度 | 原始 VOSR distillation | 当前文本 SR 观察 |
|---|---|---|
| 主要目标 | 通用图像 SR 质量 | OCR / text recognizability |
| 核心有效机制 | guided target + RC trajectory compression | guided target，不使用强 RC |
| RC 效果 | 在 VOSR 报告中提升 LPIPS / MUSIQ | 当前提升 NIQE / DISTS，但损害 OCR |
| 当前最佳变体 | 原始设置中 RC-based one-step | B: guided target, shortcut, `u_weight=0` |

因此，更准确的结论是：

> VOSR 的 restoration-oriented guidance 思想可以迁移到文本超分；
> 但原始 VOSR 的 RC/RCGM trajectory compression 不应直接视为文本 OCR
> 最优设置。

换句话说，当前实验不是否定 VOSR 的蒸馏设计，而是说明：

> 文本超分对字符结构和可识别性更敏感，因此需要重新选择蒸馏 target 和正则项。

---

## 6. 当前阶段结论

当前最好的模型是 B：

```text
guided target, cfg_scale=0.5, shortcut distillation, no RC
```

可以形成如下阶段性结论：

> Guidance-aware teacher target can improve one-step text SR over both the
> multi-step teacher and full-target distillation.

中文表述为：

> 基于 restoration-oriented guidance 的 teacher target 能够提升 one-step
> 文本超分的 OCR 表现。该提升不是由 OCR loss 带来的，因为 A/B/C 中均未使用
> OCR loss，而是来自对 teacher target 的更合适选择。

RC 的结论应谨慎表述为：

> RC/RCGM-style trajectory compression 在当前设置下改善了部分无参考感知指标，
> 但显著损害 OCR fidelity。因此，文本敏感的 one-step SR 可能更需要
> guidance-aware target design 和 text-aware representation alignment，
> 而不是强 trajectory consistency。

---

## 7. 下一步方向

下一步方法应以 B 为基础继续扩展：

```text
B + representation alignment
```

可以设计两组 REPA-style 实验：

1. **DINO-REPA**

   将 student DiT 的中间 hidden tokens 对齐到 frozen DINOv2 从 HR 图像中提取的
   visual representation。

2. **OCR-REPA**

   将 student DiT 的中间 hidden tokens 对齐到 frozen OCR recognizer 从 HR 图像中
   提取的 text-aware representation。

这样可以保持主方法的通用性：

```text
guidance-aware one-step distillation
    + optional representation alignment
```

同时通过 OCR representation alignment 实现文本超分场景下的增强，而不是直接依赖
OCR recognition loss 来硬性优化评测指标。

---

## 8. 推荐论文故事线

当前更适合的讲法是：

> We study text SR as a sensitive testbed for one-step restoration distillation.
> Instead of relying on direct OCR supervision, we first show that selecting a
> guidance-aware teacher target improves OCR fidelity over both the multi-step
> teacher and full-target distillation. We further find that the RC-based
> trajectory compression used in generic VOSR does not directly translate to
> OCR-sensitive restoration, motivating text-aware representation alignment.

对应中文表达：

> 我们将文本超分作为检验 one-step restoration distillation 的敏感场景。
> 实验表明，在不使用 OCR loss 的情况下，选择合适的 guidance-aware teacher target
> 就可以提升 one-step student 的文本可识别性；而原始 VOSR 中对通用图像有效的
> RC trajectory compression 在文本 OCR 指标上并不占优。这说明文本超分蒸馏需要
> 更关注 teacher target 的选择和文本表征对齐。
