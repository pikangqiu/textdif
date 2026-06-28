# PRISM 阅读笔记：对 VOSR/TextDiff 的参考意义

> 论文：PRISM: Prior Rectification and Uncertainty-Aware Structure Modeling for Diffusion-Based Text Image Super-Resolution  
> arXiv: 2605.13027，提交日期 2026-05-13  
> 本地资料：`paper/sources/PRISM_2605.13027.pdf`，抽取文本 `paper/sources/txt/PRISM_2605.13027.txt`  
> 结论先行：PRISM 是 **text-line / cropped Text-SR**，不能作为我们 full-image text SR 的主任务对齐对象；但它在训练期先验、局部结构、不确定性、Real-CE 诊断和 eval 工程上非常值得借鉴。

---

## 1. 这篇论文到底做什么

PRISM 做的是中文-英文 **text-line super-resolution**。它的典型输入输出不是整张场景图，而是裁好的条状文本图：

```text
LR text-line: 32 x 128
-> SD2.1 latent diffusion based PRISM
-> HR text-line: 128 x 512
```

模型基座：

| 部分 | PRISM 设计 |
|---|---|
| 基础生成模型 | Stable Diffusion 2.1-base |
| VAE | SD2.1 VAE，冻结 |
| UNet | SD2.1 UNet，LoRA rank 16 微调 |
| 推理 | single-step restoration，固定 timestep 399 |
| 输入形态 | cropped text-line / text crop |
| 主要新增模块 | FMPR + SURE |
| OCR eval | PP-OCRv5 ACC/NED |
| 训练数据 | BTL: 100K bilingual text-line corpus |

它不是从 TeReDiff 改出来的。TeReDiff 在 PRISM 里只是 baseline。

---

## 2. 不能直接借的地方

最重要的边界：**PRISM 的任务定义不是我们的任务定义。**

我们做的是：

```text
full/global text-rich image SR
-> restore full image
-> evaluate text regions / spotting afterward
```

PRISM 做的是：

```text
cropped text-line SR
-> OCR on the restored text line
```

因此：

- 不能把 PRISM-style text-line eval 当我们的主表。
- 不能让 baseline 直接吃裁好的 text-line，否则我们的 full-image/global SR 任务会被带偏。
- 不能为了 Real-CE 数字把方法改成 text-line module as main method。
- 不能直接 claim 与 PRISM 进行公平主对比，除非单独开一个 “line-level diagnostic” 附录。

合适定位：

> PRISM 是最新 text-line diffusion Text-SR 的强参考，不是我们 full-image text SR 的主对手。它能帮助我们诊断 Real-CE、中文笔画和训练数据问题。

---

## 3. 值得借鉴的模型设计

### 3.1 FMPR：training-time privileged prior

PRISM 的 FMPR 做了一个很重要的拆分：

1. 训练时同时看 LQ/HQ latent，构造一个 privileged prior `c*`。
2. `c*` 只在训练时可见，用来定义“好的 text-aware condition 应该长什么样”。
3. 推理时只能看 LQ，通过 LQ-only prior encoder + flow matching，把 degraded prior 拉向 privileged prior space。

它真正有价值的不是具体 MLP 或 Euler step，而是这个思想：

> 文字先验不要只从退化输入硬抽；可以用训练期 HR/GT 信息定义一个更可靠的目标先验，再让推理模型学习如何从 LQ 自己恢复这个先验。

和我们的关系：

- 我们已经有 “student-side supervision 才能穿透蒸馏均衡器” 的 thesis。
- FMPR 可以被改写成我们的语言：**training-time privileged OCR/text prior as student-side target**。
- 推理期仍然不需要 OCR、不需要 GT、不需要额外文字条件。

可尝试的改动：

| PRISM | 我们的可迁移版本 |
|---|---|
| LQ/HQ latent privileged prior | HR/GT text-region OCR feature target |
| Flow matching to recover prior | student feature alignment / OCR-REPA target recovery |
| one-step backbone warmed by privileged prior | distillation student directly supervised by privileged text target |

这个可以成为一个小实验，但不建议变成主贡献，以免分散“蒸馏均衡器”主线。

### 3.2 SURE：uncertainty-aware local structure

PRISM 的 SURE 解决另一个问题：

- 即使全局 text prior 对了，局部笔画边界也可能不确定。
- LQ edge/boundary 不一定可靠，直接用边缘监督可能把错误笔画强化。
- 所以它预测 structural cue 的 mean 和 variance，用 uncertainty 控制哪些局部线索可信。

和我们的关系：

- Real-CE 中文复杂字符差，很可能不只是语义错，而是断笔、粘连、闭合结构错。
- 这类错误用整图 IQA 看不出来，用 spotting 也会被 detection 混淆。
- 可以在 GT text regions 上加一个轻量结构监督，验证是否改善中文 CharAcc/NED。

可尝试的改动：

| 版本 | 改动 | 目的 |
|---|---|---|
| `B + OCR` | 当前 student-side OCR supervision | 主线 |
| `B + OCR + region_aux` | 训练时裁 GT text regions 加密集 OCR supervision | 补局部文字监督 |
| `B + OCR + region_aux + boundary` | 加 Sobel/edge/stroke boundary loss | 看中文笔画是否改善 |
| `B + OCR + region_aux + uncertainty_boundary` | 对边界监督加 uncertainty / confidence weighting | 避免错误边缘误导 |

注意：这应该是 ablation/diagnostic，不要让方法变成 PRISM-style line model。

### 3.3 Progressive freezing 的工程思路

PRISM 的训练不是一锅端：

1. privileged-prior construction 训练 100K。
2. LQ-only prior recovery 训练 100K。
3. 冻结 FMPR pathway 和 restoration backbone，再训练 SURE 50K。

我们可以借鉴“分阶段定位问题”的工程方式：

- 先固定主 distillation student，验证 OCR-REPA/CTC 是否稳定。
- 再只加 region auxiliary batch，看是否改善 Real-CE。
- 最后再加 boundary/uncertainty 小分支。

这样能避免实验互相污染，也更容易讲清楚每个模块解决什么。

---

## 4. 值得借鉴的数据工程

PRISM 构造 BTL 的动机是：没有一个现成数据集同时满足中文、英文、高质量、text-line SR。

BTL 数据来源：

| 来源 | 用途 |
|---|---|
| CTR | 大量中文 text-line crops |
| SA-Text | 高质量英文 text crops |
| synthetic rendering | 字体/布局/文本多样性 |

筛选规则：

- annotations valid。
- resize height = 128。
- aspect ratio 2 到 8。
- transcript length <= 24。
- MUSIQ / MANIQA / CLIP-IQA 做 no-reference quality ranking。

最终：

| 数据 | 数量 |
|---|---:|
| curated real HQ crops | 50K |
| synthetic HQ text-line images | 50K |
| total BTL | 100K |
| train / test | 80K / 20K |

对我们的启发：

- TAIR/SA-Text 中文样本可能严重不足。PRISM 统计中，SA-Text usable Chinese candidates 远少于 CTR。
- 我们不应把 BTL 当主训练集照搬，但可以构造 **text-region auxiliary pool**。
- 这个 pool 应该服务 full-image student training，而不是把任务改成 line SR。

建议构造：

```text
full-image distillation data
+ GT text-region crops from TAIR / SA-Text / Real-Texts / Real-CE train
+ Chinese oversampling
+ quality filtering
+ OCR/CTC/feature supervision only during training
```

---

## 5. 值得借鉴的 eval 工程

PRISM 的 eval 值得借的不是任务形式，而是工程透明度：

- 明确 RealCE-val 经过过滤。
- 过滤理由：misalignment、color mismatch、annotation errors。
- 手工修正错误标注。
- 固定 OCR：PP-OCRv5。
- 同时报 ACC/NED、PSNR/LPIPS/FID、runtime。
- 速度比较固定输入输出尺寸。
- TeReDiff 作为 baseline 被 retrain/fine-tune on BTL-train，但仍然 RealCE-val 差。

对我们的 eval 要求：

| 要求 | 我们怎么做 |
|---|---|
| 不改变任务 | 所有方法先输出完整 SR 图 |
| 固定 region | OCR 只裁 GT text regions / GT lines，不动态换框 |
| 固定 OCR | PP-OCRv5 + Real-CE official CRNN 附表 |
| 固定 split | Real-CE valid split 文件化 |
| sanity check | 报 LR 与 HR oracle |
| 语言拆分 | Chinese / English / digit 分别报 |
| 不混 published | 自跑同协议表和 as-reported context 分开 |

主表应叫：

> Real-CE global-region readability evaluation

而不是：

> Real-CE text-line SR benchmark

---

## 6. PRISM 对我们贡献写法的影响

PRISM 进一步说明：现在“text prior + one-step diffusion + Real-CE”已经很拥挤。我们不能把贡献写成：

- first one-step text SR。
- first diffusion text prior。
- first Real-CE PP-OCR eval。
- stronger text-line SR。

更好的贡献写法：

1. **Mechanism**：发现 one-step full-image text restoration distillation 中存在 distillation equalizer：teacher/input/generic prior 容易被抹平，student-side OCR-specific supervision 才稳定穿透。
2. **Method**：提出 full-image one-step student 的双空间 OCR supervision，训练期使用 OCR/GT text signal，推理期不需要 OCR、不需要 text condition、不改变 global input/output。
3. **Evaluation**：提出 full-image text SR 的 bridge evaluation：TAIR spotting + Real-CE global-region OCR + region IQA + NFE + language split。
4. **Diagnosis**：解释 Real-CE 中文弱项来自中文覆盖、局部笔画结构和 full-image/line-level task gap，而不是单纯缺一个 text prior。

可用表述：

> Unlike recent line-level Text-SR methods such as PRISM, our setting restores full text-rich images. We therefore evaluate Real-CE by applying SR globally and measuring readability on fixed GT text regions, preserving the full-image restoration setting while isolating text-region legibility.

---

## 7. 对论文整体结构的影响

建议结构：

1. **Introduction**
   - 真实图像 SR 不等于文字可读。
   - 扩散时代已有很多 text-aware prior。
   - 高效部署要求 one-step，但 one-step distillation 下哪些 text signals survive 仍不清楚。
   - 我们发现 distillation equalizer，并设计 student-side supervision。

2. **Related Work**
   - Cropped/Text-line STISR：TextZoom, DiffTSR, PRISM。
   - Full-image text-aware restoration：TAIR, TADiSR, TIGER, TEXTS-Diff。
   - One-step diffusion SR：OSEDiff, TSD-SR, SinSR, DFO。
   - Evaluation protocols：Text-line recognition vs full-scene spotting vs region OCR。

3. **Method**
   - Base one-step distillation。
   - Distillation equalizer diagnostic。
   - Student-side OCR-REPA / GT-text CTC。
   - Optional region auxiliary / boundary supervision as robustness extension。

4. **Evaluation Protocol**
   - TAIR full-scene spotting。
   - Real-CE global-region OCR。
   - Region IQA。
   - Language split。
   - NFE/runtime。

5. **Experiments**
   - Main TAIR table。
   - Real-CE table。
   - Mechanism ablation。
   - Real-CE diagnosis and auxiliary training。

---

## 8. Real-CE 效果差的原因与办法

### 8.1 可能原因

| 原因 | 解释 | 怎么验证 |
|---|---|---|
| 中文样本覆盖不足 | TAIR/SA-Text 中文样本少，复杂中文笔画没学足 | Chinese/English split |
| full-image supervision 太稀 | 小文字区域在全图 loss 中占比低 | region OCR vs full IQA |
| one-step distillation 抹掉细节 | teacher/input prior 被均衡，局部结构丢失 | teacher/B/2a/2d/2a+2d 对比 |
| 局部笔画结构缺监督 | 中文断笔/粘连/闭合结构错 | boundary/stroke diagnostic |
| Real-CE protocol 噪声 | misalignment、标注错误、颜色不一致 | valid split + HR/LR sanity |
| 不只是数据问题 | PRISM 中 TeReDiff 即使用 BTL 设置仍差 | 引作 motivation，不作直接主比 |

### 8.2 可能办法

P0:

- 固定 Real-CE valid split。
- 报 LR/HR oracle。
- Real-CE 指标按 Chinese/English/digit 拆。
- 对 B、2a、2d、2a+2d 同协议重跑。

P1:

- 加 text-region auxiliary batch。
- 对中文区域 over-sample。
- 加 PP-OCRv5/CRNN 双 evaluator。

P2:

- 加 uncertainty-weighted boundary/stroke loss。
- 做 PRISM-style text-line diagnostic，但不进入主 claim。

---

## 9. 最后判断

PRISM 给我们的真正参考意义是：

> 复杂文本可读性不是靠“更强扩散先验”自然出现的。它需要可靠的训练期文字先验、局部结构约束、干净的 eval split，以及明确的任务边界。

对我们来说，最好的吸收方式不是跟着 PRISM 改成条状文字 SR，而是：

```text
保持 full-image SR 任务
+ 训练期 text-region auxiliary supervision
+ student-side OCR-specific targets
+ Real-CE global-region OCR eval
+ Chinese/English split diagnosis
```

这样既能回应 Real-CE 弱项，也不会牺牲我们自己的论文定位。
