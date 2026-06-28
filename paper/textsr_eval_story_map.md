# Text-SR 前人状态图谱：eval、训练与 story 切入口

> 建立：2026-06-27。目的：把 Zotero `text/text SR` 与 `paper/sources` 中的文本超分论文按“时代/任务/评测/训练叙事”重新梳理，服务我们论文的 intro、related work、eval protocol 和 contribution 表述。
>
> 读法提醒：这里不按“谁用了什么模块”排序，而按**评测形态如何定义问题**、**训练信号如何进入模型**、**论文故事如何 claim 贡献**排序。

---

## 0. 结论先行

文本超分现在不是一个单一赛道，而是两个时代、两种任务形态叠在一起：

1. **TextZoom / crop-STISR 时代**：问题被定义为“给定裁好的文本行/词图，是否能让 recognizer 读对”。评测以 CRNN/MORAN/ASTER/TransOCR 的 ACC/NED 为中心，检测问题被剥离。论文故事通常是“更好的文字先验/语义监督/结构损失能提升识别”。
2. **扩散 / real-scene 时代**：问题变成“真实退化场景里的文字区域是否既能被下游系统找到、又能被 OCR 读对，同时不牺牲图像质量”。评测口径开始分裂：TAIR 走 spotting，Real-CE 走 CRNN 行级识别，TADiSR/TIGER/TEXTS-Diff 走 PP-OCR 区域识别，但 OCR-A 定义、参照物、测试子集都不同。

这给我们的切入口很清楚：

- 不要把故事写成“我们又提出一个 text-aware SR 方法”。这个位置已经被 TextSR、TADiSR、TeReDiff、TIGER、DualTSR 等占满。
- 不要把贡献写成“首个 one-step text SR”。TEXTS-Diff 已经冲掉这个 claim。
- 最稳的主线应该是：**扩散时代的文本超分已经有很多文字先验，但没有回答“一步蒸馏里什么文字信号真正能存活”。我们的贡献是把问题从 text-aware restoration 推到 distillation-aware text restoration：teacher/input/generic priors 会被蒸馏均衡器抹平，只有 student 侧、细粒度、OCR-specific 的监督稳定穿透。**
- eval 贡献不要包装成“发明新指标”，而应写成：**跨阵营统一再评测 / bridge evaluation**。我们把 L1 spotting、L2 given-region OCR、区域 IQA、NFE 放在同一批图和同一条冻结管线下，使两个时代、两个阵营的结论能被同时观察。

---

## 1. 两个时代的状态差异

### 1.1 TextZoom / crop-STISR 时代：recognizer-centric

代表：TextZoom, TPGSR, C3-STISR, DPMN, TSTSRN, PEAN, DiffTSR 的行级设置，以及 Zotero `text SR` 中大量 2020-2024 的 STISR 工作。

这一时代的默认设定：

- 输入常是裁好的 word/line patch，TextZoom 典型尺寸如 32x128，Real-CE 也强调从 TextZoom 扩展到更复杂中英长文本。
- 评测问题被简化成“文字是否更容易被识别器识别”，常用 CRNN/MORAN/ASTER/TransOCR 等外部 recognizer 的 ACC/NED。
- 检测、版面、真实整图退化和区域定位不在主问题里；换句话说，OCR 任务链里的 detection 被移除。
- 训练故事多围绕 recognizer guidance、text prior、stroke/edge/semantic structure、attention alignment。
- 贡献话术通常是“利用文本语义/结构先验改善可读性”，这在当时是有效叙事，但在扩散时代已经变成公共语言。

它对我们的作用：

- 作为 historical root：说明文本 SR 一直以 recognizability 为核心，而不是单纯 PSNR/SSIM。
- 作为边界：我们的主 benchmark 不应退回 TextZoom。TextZoom 是 crop-STISR 子任务，不足以支撑扩散时代的真实场景 restore 叙事。
- 作为 related work 分类：可以把这些方法归为“recognizer-guided cropped STISR”，强调它们大多不处理 full-scene spotting 和 one-step distillation。

### 1.2 扩散 / real-scene 时代：eval fractured

代表：TAIR/TeReDiff, Real-CE, DiffTSR, TADiSR, TIGER, TEXTS-Diff, TextSR, DualTSR, PRISM/NCAP/DCDM 等扩散或生成式文本 SR 工作。

这一时代的变化：

- 问题从“裁好文本行读不读得出”升级为“真实退化整图/区域中的文字是否可信”。扩散模型能生成更自然图像，但也会 hallucinate 字形，导致“看起来像字但读错”的风险变成主矛盾。
- 训练数据开始混合 synthetic text data、Real-CE train、Real-Texts、SA-Text、segmentation masks、OCR 预测文本或 GT transcription。
- 方法故事从 handcrafted recognizer prior 扩展到 diffusion prior、text-conditioned generation、segmentation decoder、two-stage text-first/image-later、discrete text diffusion、OCR-conditioned iterative refinement。
- 评测没有统一，反而更碎：
  - TAIR：全图 spotting，TESTR/ABCNet，det F1 + E2E None/Full。
  - Real-CE：官方 CRNN，ACC/NED，强调中英真实配准与长文本。
  - DiffTSR：裁剪文本行，TransOCR，ACC/NED。
  - TADiSR：PP-OCR 区域 OCR-A，但 Real-CE 只用过滤子集，且参照物和定义与别人不同。
  - TIGER：PP-OCR OCR-A + Delta OCR-A + 区域 IQA，开始显式讨论 OCR vs IQA trade-off。
  - TEXTS-Diff：PP-OCRv5 整行精确匹配，Real-Texts/Real-CE/RealSR，并已经是 one-step text-aware diffusion。

它对我们的作用：

- 扩散时代真正缺的不是“再加一个文字模块”，而是**搞清楚在高效一步蒸馏中，哪些文字信号不会被洗掉**。
- eval 的痛点不是没有指标，而是**指标之间不可互搬**：OCR 引擎、OCR-A 定义、参照物、区域子集、是否检测、是否报 NFE 都不同。
- 我们应顺着这个痛点写：我们不是替领域发明一个全新 benchmark，而是把已有口径对齐、并暴露它们对方法排序的差异。

---

## 2. 前人 eval 的真实状态

### 2.1 任务形态分裂

| 形态 | 问的问题 | 代表论文 | 常用指标 | 对我们的启发 |
|---|---|---|---|---|
| Cropped line/word STISR | 给定文本裁剪图，字清不清、recognizer 读不读对 | TextZoom, Real-CE official, DiffTSR, DualTSR 部分设置 | ACC, NED, PSNR, SSIM, LPIPS | 适合证明字形可读性，但无法证明下游 spotting |
| Full-scene spotting | SR 后整图中的文字能否被检测并识别 | TAIR/TeReDiff | det F1, E2E None, E2E Full | 适合证明真实部署价值，但检测和识别纠缠 |
| Given-region OCR | 给 GT 区域/框，剥离检测后看字形是否可读 | TADiSR, TIGER, TEXTS-Diff | OCR-A, EM, CharAcc/CER, NED | 适合补足 spotting 的混淆，是我们 L2 的核心 |
| Region IQA | 只在文字区域算质量 | TIGER | PSNRcr, SSIMcr, LPIPScr, DISTScr | 比整图 IQA 更能解释“图好但字错/字好但图差” |

核心判断：**没有任何单一指标能承载我们的故事。** 我们需要 L1+L2+IQA 并排，才可以说清楚“一步蒸馏如何影响 detection、recognition、visual fidelity”。

### 2.2 OCR-A 这个词本身不统一

目前至少有三种 OCR-A：

- TADiSR/TIGER 一类：更接近 Levenshtein ratio / normalized edit similarity。
- TEXTS-Diff 一类：PP-OCRv5 整行精确匹配，硬得多。
- 参照物也不同：有的对 GT transcription，有的对 OCR-of-HR 伪标签。

所以主文里要避免“我们比某某 OCR-A 0.xx 高/低”的直接横跳。更稳的写法：

- published numbers 只作为 “as-reported, protocol differs”。
- ranking 和 headline 只来自我们的 frozen re-evaluation。
- 如果必须引用前人数，要把 engine、reference、test subset 写在表注里。

### 2.3 Delta OCR-A 不是我们的原创，但适合被我们继承

TIGER 已经使用 `Delta OCR-A vs LR`，并指出很多通用扩散/生成式 SR 会让 OCR 下降。这个对我们有价值，但不能 claim first。

推荐表述：

- “Following TIGER, we report the OCR gain over LR to quantify whether restoration actually improves machine readability.”
- 中文理解：我们借 TIGER 的 Delta OCR-A，把它放入统一管线，证明我们的 one-step student 不只是图像变好，而是 OCR 也相对 LR 正增益。

### 2.4 NFE/steps 是扩散时代应该强制出现的列

前人常强调效果，但不稳定报告 NFE。对扩散 SR 来说这不合理，因为 1 step、5 step、20 step、50 step 的应用代价完全不同。

我们应把 `NFE` 放进所有主表：

- 不是单独当 novelty，而是作为公平性维度。
- 让 reviewer 看到：我们的精度不是用多步采样换来的。
- 对 TEXTS-Diff/TADiSR 这类 one-step 或 near-one-step 方法，NFE 列也能帮助避免“你只是更快”或“别人也一步”的误读。

---

## 3. 前人训练策略的真实状态

### 3.1 TextZoom 时代：recognizer/structure supervision

典型训练信号：

- reconstruction loss：L1/L2/Charbonnier。
- perceptual/adversarial loss：提升视觉自然度。
- recognition loss / text prior：用 recognizer feature、predicted text、semantic embedding 指导 SR。
- structure loss：edge、stroke、segmentation mask、spatial attention。

故事方式：

- “SR should serve downstream recognition.”
- “Text has semantic/structural priors absent in natural image SR.”
- “A recognizer provides high-level guidance for legible restoration.”

对我们来说，这些已经是旧共识。我们的新意不能停在“加 OCR supervision 有用”，而要落到**在 distillation setting 里，监督放在哪里才有用**。

### 3.2 扩散时代：input-side / teacher-side / external-prior 很拥挤

典型训练或推理信号：

- OCR/text prompt condition：TextSR、TEXTS-Diff 等。
- segmentation mask 或 joint decoder：TADiSR。
- spotting network inside diffusion：TAIR/TeReDiff。
- two-stage text-first restoration：TIGER。
- continuous image diffusion + discrete text diffusion：DualTSR。
- generic representation alignment：REPA/OSEDiff/TSD-SR 这类通用生成或一步蒸馏根基。

它们共同相信：把文本条件、结构条件、分割条件、识别器条件输入模型，模型就会更 text-aware。

我们的实验可以形成反命题：

- 在多步 diffusion restoration 里，input-side 或 teacher-side prior 可能有效。
- 但在 one-step distillation 里，这些信号可能被 student 的蒸馏目标平均掉、投影掉、或变成不可被最终映射使用的弱信号。
- 因此关键不再是“有没有文字先验”，而是**文字先验是否直接约束 student 的可读输出/中间表征**。

这就是“蒸馏均衡器”能站住的地方。

### 3.3 Real-CE/Real-Texts 训练的时代变化

Real-CE 原论文强调 TextZoom 对中文复杂字符和长文本泛化不足；后续扩散工作大多不再纯靠 TextZoom，而会混入真实域数据：

- TADiSR：合成/FTSR + 过滤后的 Real-CE train。
- TIGER：合成 + Real-CE 重标子集 + UZ-ST。
- TEXTS-Diff：Real-Texts + Real-CE train cropped regions + LSDIR。
- Real-CE 官方：自带 GT box、text line transcription、edge-aware supervision。

对我们的训练设置建议：

- 主文应区分两种 setting：
  - **Zero-shot / no Real-CE-train**：支撑“跨域一致性”和“蒸馏监督机制”的故事。
  - **Mixed real-domain training**：若要竞争 Real-CE 数字，需要公平承认前人也混入 Real-CE train，并把我们对应训练行单列。
- 不要把“未混 Real-CE train”下的 Real-CE 数字和前人“混 Real-CE train”数字直接比较。
- 若我们混入 Real-CE train，最好说成“following diffusion-era practice, we include real-domain paired text regions for competitive evaluation”，而不是“fine-tune trick”。

---

## 4. 前人 contribution 写法与我们的避雷

### 4.1 已被占掉的位置

这些 claim 不适合当我们的主贡献：

- “text-aware diffusion for text SR”：TextSR、TADiSR、TEXTS-Diff、TIGER、TAIR 都已经覆盖。
- “first one-step text SR”：TEXTS-Diff 已经是 one-step text-aware diffusion，TADiSR 也有单步/近单步设定。
- “new OCR metric / Delta OCR”：TIGER 已经用 Delta OCR-A，TEXTS-Diff/TADiSR/TIGER 都有 PP-OCR 变体。
- “Real-CE SOTA”作为唯一主张：Real-CE 各论文协议严重不一致，而且现有项目文档也提示当前 Real-CE 主要支撑跨域一致性，不宜过载。

### 4.2 仍然有空间的位置

更稳的 contribution 角度：

1. **Distillation principle**：首次系统诊断 one-step text restoration distillation 中文字监督的有效位置，发现 teacher/input/generic priors 被均衡，student-side OCR-specific supervision 才穿透。
2. **Training design**：提出双空间 student supervision：区域 OCR-REPA 约束特征空间，GT-text CTC 约束输出可读性；推理期不需要 OCR、不需要文字标注、不增加 NFE。
3. **Evaluation bridge**：统一 L1 spotting、L2 region OCR、region IQA 与 NFE，在同一批输出上观察方法排序，避免跨论文 OCR-A 混比。
4. **Efficiency as constraint, not only selling point**：我们不是“为了快牺牲识别”，而是在 NFE=1 条件下通过 student-side text supervision 保住/提升可读性。

### 4.3 推荐的 contribution 草稿

可以写成三条，不要四条堆太满：

1. We identify a distillation equalization phenomenon in one-step text image restoration: text priors injected through the teacher, input conditions, or generic representation alignment are largely washed out by distillation, whereas fine-grained OCR-specific supervision applied to the student remains effective.
2. We propose a dual-space student supervision scheme that combines localized OCR representation alignment with GT-text CTC supervision, improving machine readability without requiring OCR/text conditions at inference and keeping NFE=1.
3. We build a unified evaluation bridge for diffusion-era text SR by reporting end-to-end spotting, given-region OCR, text-region IQA, OCR gain over LR, and NFE under frozen evaluation pipelines, exposing protocol gaps in prior as-reported comparisons.

中文内核：

- 贡献 1 是“发现/诊断”，最像 paper 的科学性。
- 贡献 2 是“方法”，但要服务于发现，不要像堆模块。
- 贡献 3 是“测量/再评测”，要克制，不要说成 invented protocol。

---

## 5. 我们论文 story 的切入口

### 5.1 建议主线

扩散模型把真实场景 SR 推到了“更真实”的阶段，但文字 SR 有一个特殊问题：**真实不等于可读**。生成式模型可以补出视觉上合理的字形，却让 OCR 和下游 spotting 变差。于是领域开始引入各种文本先验：OCR prompts、segmentation decoders、text spotting branches、discrete text diffusion、two-stage text-first restoration。

但是，高效应用要求 one-step restoration。这里出现一个新问题：**多步/teacher 中有效的文字先验，蒸馏到一步 student 后是否还有效？** 现有工作基本没有回答这个问题。

我们的发现是：one-step distillation behaves like an equalizer。它会把许多 teacher-side/input-side/generic alignment 的差异抹平；真正留下来的，是直接施加在 student 上、局部文本区域内、OCR-specific 的细粒度监督。

因此我们的工作不是“又一个 text-aware diffusion model”，而是“告诉扩散时代文本 SR：在 one-step distillation 里，文字监督应该放在哪里、如何评估它是否真的改善可读性”。

### 5.2 Intro 可用的三段逻辑

第一段：文本 SR 的目标变了。

- 过去 TextZoom-style STISR 主要看 crop text recognition。
- 现在 diffusion restoration 进入真实整图/真实退化阶段，核心风险变成 hallucinated but unreadable text。
- 因此评测必须同时看 spotting、given-region OCR 和 image quality。

第二段：现有 text-aware diffusion 的盲点。

- 现有方法通过 OCR/text condition、segmentation、spotting branch、text-first stage 等增强文字。
- 这些多在 multi-step 或 heavily conditioned restoration 里成立。
- 但高效部署依赖 one-step distillation，而目前缺少对“哪些 text priors survive distillation”的系统研究。

第三段：我们的发现和方法。

- 通过 orthogonal probes 发现 teacher/input/generic priors 被均衡。
- Student-side fine-grained OCR supervision 才能穿透。
- 基于此设计 dual-space supervision，并用 unified bridge eval 证明 OCR/readability 的提升不是某个 recognizer 或某个 protocol 的偶然。

### 5.3 Related Work 的组织建议

不要按“GAN/Transformer/Diffusion”机械排序。建议四段：

1. **Recognizer-guided cropped STISR**：TextZoom、TPGSR、C3-STISR、DPMN、TSTSRN 等。强调它们建立了 recognizability 目标，但任务形态是 cropped text。
2. **Diffusion-era text-aware restoration**：DiffTSR、TextSR、TAIR/TeReDiff、TADiSR、TIGER、DualTSR、TEXTS-Diff。强调它们把 text priors 注入 diffusion，但大多依赖多步、输入条件、外部 OCR 或额外结构。
3. **One-step diffusion SR and distillation**：OSEDiff、TSD-SR、SinSR、REPA。强调通用 SR 里已有一步蒸馏/representation alignment，但没有回答文字可读性和 OCR-specific supervision 的问题。
4. **Evaluation protocols for text SR**：TextZoom/Real-CE/TAIR/TADiSR/TIGER/TEXTS-Diff。强调口径分裂，导出我们的 bridge evaluation。

---

## 6. Eval 设计建议

### 6.1 主表不要只做一个 leaderboard

建议主表列：

- Method group：General SR / Generic diffusion SR / Text-aware diffusion SR / Ours。
- NFE。
- L1 spotting：det F1, E2E None, E2E Full。
- L2 region OCR：EM, CharAcc/CER, OCR-A(Levenshtein)。
- Delta OCR vs LR。
- Region IQA：PSNRcr, SSIMcr, LPIPScr, DISTScr。
- Full-image IQA：PSNR, SSIM, LPIPS/FID 可视篇幅决定主表/附录。

表述重点：

- L1 表明下游系统能不能“找到并读对”。
- L2 表明剥离检测后字形是否真的可读。
- IQA 表明有没有为了 OCR 牺牲图像。
- NFE 表明效率约束。

### 6.2 对比方式要分 published 与 self-run

推荐三层：

1. **Frozen self-run table**：只用同一管线重跑的方法参与主排名。
2. **As-reported context table**：收录 TADiSR/TIGER/TEXTS-Diff/DualTSR 等，但显式标注 protocol differs，不直接排名。
3. **Ablation/mechanism table**：B、teacher-side、input-side、generic REPA、student OCR-REPA、GT-text CTC、2a+2d，证明“有效位置”。

这样可以避免两个 reviewer 攻击：

- “你混用了不同 OCR-A。”
- “你的 eval 是为了自己设计的。”

### 6.3 必做的 mechanism eval

为了让 story 真正成立，eval 不应只有最终 SOTA，还要把“蒸馏均衡器”测出来：

- teacher-side enhanced teacher: teacher improved, distilled student returns near baseline。
- input-side text/OCR condition: condition exists, but student gain weak or gate shuts it down。
- generic representation alignment: DINO/seg/detection-style alignment near baseline。
- student-side OCR-specific feature alignment: stable gain。
- output-space GT-text CTC: stable gain。
- combined student supervision: complementary or Pareto improvement。

这些实验比多加一个 baseline 更重要，因为它们直接回答前人没回答的问题。

### 6.4 Real-CE 的写法

Real-CE 必须克制：

- 它是跨域/中文复杂字符/真实配准的重要 evidence。
- 但前人 Real-CE 子集和 OCR 引擎太乱，不适合直接 claim broad SOTA。
- 若我们未混 Real-CE train，写“zero-shot generalization / cross-domain consistency”。
- 若我们混 Real-CE train，写“competitive setting following prior practice”，并和 zero-shot 分开。

---

## 7. 论文措辞：可以说与不要说

### 可以说

- “We study which text priors survive one-step diffusion distillation.”
- “We find that student-side fine-grained OCR supervision is critical, while teacher-side, input-side, and generic alignment signals are largely equalized after distillation.”
- “Our evaluation bridges full-image spotting and given-region OCR under a frozen pipeline.”
- “Following TIGER, we report OCR gain over LR, but evaluate it together with NFE and region IQA.”
- “Unlike inference-time OCR-conditioned methods, our OCR supervision is only used during training.”

### 不要说

- “First one-step text SR.”
- “We introduce Delta OCR.”
- “We introduce the first OCR-based evaluation protocol.”
- “We beat all Real-CE methods” unless all baselines are rerun under exactly the same subset/engine/reference/training setting。
- “TextZoom results prove real-scene performance.”

### 建议的 title/abstract 关键词

可以围绕：

- one-step text image restoration
- distillation equalization
- student-side OCR supervision
- diffusion-era text SR evaluation
- machine readability under efficient restoration

不要围绕：

- generic text-aware diffusion
- simply OCR-guided SR
- benchmark invention

---

## 8. 需要补的阅读/实验动作

1. Zotero 去重：`text SR` 集合里有大量重复条目，后续正式 related work 可按 title/year 去重后生成 bib checklist。
2. 精读 TEXTS-Diff：重点确认 one-step 训练、PP-OCRv5 OCR-A 定义、Real-Texts/Real-CE 数据构成，避免 C1 表述踩雷。
3. 精读 TIGER：确认 Delta OCR-A、region IQA、baseline grouping 的表注，C4 必须 credit。
4. 精读 TADiSR：确认 Real-CE 189 子集、OCR-of-HR 或 GT reference、单步设置和 JSD/seg decoder 的 contribution 写法。
5. 精读 PRISM：确认它的 text-line 任务边界、BTL 数据构造、RealCE-val clean split、PP-OCRv5 ACC/NED、TeReDiff 负例，以及 FMPR/SURE 对我们 training-time prior 和局部结构监督的启发。详见 `PRISM_reading_notes.md`。
6. 精读 Real-CE：确认官方 ACC/NED、test split、line transcription、TextZoom limitation 的原文证据。
7. 实验上优先补 frozen eval 的脚本和输出，而不是扩 baseline 数量。story 现在缺的不是更多方法名，而是同口径证据。

---

## 9. 可直接迁移到现有文件的改动建议

### `writing_strategy.md`

- 把 C1 改成“one-step distillation for text restoration under efficiency constraint”，不要当 first。
- C2 放到 thesis 位：distillation equalizer。
- C4 继续降级为“bridge evaluation / reproducible re-evaluation”，明确不是新指标。

### `eval_protocol.md`

- 保留 UTEP 工作名，但正文避免“protocol invention”语气。
- 增加 “as-reported vs frozen self-run” 的表格规则。
- 在 Delta OCR 部分明确 `following TIGER`。
- 在 OCR-A 部分把 EM、CharAcc/CER、Levenshtein OCR-A 三个口径同时报，减少与 TEXTS-Diff/TIGER/TADiSR 的冲突。

### `related_work_scan.md`

- 补一个 “TextZoom-era cropped STISR vs diffusion-era real-scene restoration” 的开场框。
- 把 TEXTS-Diff 单独标成 “one-step text-aware diffusion competitor”，用于约束我们 C1。
- 把 DiffTSR 放在 “open-source cropped/line-level diffusion STISR” 而不是 full-scene 主对标。

---

## 10. 一句话版本

前人已经证明“文本先验对 SR 有用”，也已经进入“扩散/一步/真实场景”的阶段；我们最好的论文切口不是再证明 text-aware，而是证明：**在 one-step distillation 这个新约束下，绝大多数 text-aware 信号并不会自然留下来，必须把 OCR-specific 细粒度监督直接压到 student 上，并用跨 spotting、区域 OCR、区域 IQA、NFE 的统一评测证明这种监督真正提升了可读性。**
