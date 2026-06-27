# 相关工作扫描 + 参考论文结构借鉴（three-way-scan）

> 建立 2026-06-22。扩散时代文本/场景文字图像超分的文献地图，服务于：① Related Work 写作；② 对标定位；③ novelty 边界确认；④ 借鉴 3 篇参考论文的结构与对比组织。
> PDF 在 `sources/`。每条 WHY=解决什么、HOW=方法、WHAT=结果/基准/口径。

---

## 0. 一句话地图

| 维度 | 现状 | 我们的位置 |
|---|---|---|
| 文字先验注入方式 | 输入侧 OCR 条件（TextSR/TADiSR/TeReDiff，多步或迭代） | **student 侧训练期监督，推理零开销/不需 OCR** |
| 步数 | 几乎全多步；TADiSR/我们单步 | **NFE=1 蒸馏** |
| 蒸馏 × 文本 SR | 通用一步蒸馏(OSEDiff/TSD-SR)均非文字向 | **白地：一步蒸馏 for text-SR 无人做** |
| 关键 insight | 默认"加文字条件就有用" | **蒸馏均衡器：输入/teacher 侧被抹平，唯 student 侧细粒度 OCR 穿透**（无人报告） |

---

## 1. 基准线

### TAIR / TeReDiff（ICCV25，我们的 codebase 基准）
- **WHY** 扩散复原幻觉出"看似合理但错误"的文字（text-image hallucination）。
- **HOW** 多任务扩散：扩散内部特征→text-spotting，中间文字预测反向 condition 去噪；放出 SA-Text 10万张密集标注。多步。
- **WHAT** TAIR/SA-Text SOTA；**明确称 TextZoom 是 TAIR 的子任务** → 支撑我们"弃 TextZoom 转 TAIR"。

### Real-CE（ICCV23，第二基准）
- **WHY** 已有偏英文（结构简单），缺中文复杂笔画真实基准。
- **HOW** 13/26/52mm 真实配准对；edge-aware 图像+特征域结构监督。
- **WHAT** 1935/783 对；**自带中/英 CRNN 评 ACC/NED**（=我们协议③）。

---

## 2. 三篇用户指定参考（不对比，借鉴结构）

### TADiSR（2025-06）—— Real-CE 最大威胁 + 主表/消融范本
- **WHY** 生成式 SR 牺牲结构、文字笔画畸变/错字。
- **HOW** LoRA 微调 cross-attn 聚焦"text"+取注意力图；**双解码器（图像+文字分割）+ 跨解码器交互 CDIB**；modified focal loss 耦合。**单扩散步（t=200）**。
- **WHAT 结构（可直接借鉴）**：
  - 章节：Intro / Related / Method(3.1 总架构 / 3.2 Text-Aware Cross-Attn / 3.3 Joint Seg Decoders / 3.4 Loss / 3.5 数据合成) / Exp(4.1 实现 / 4.2 性能 / 4.3 消融) / Conclusion / Appendix A-D。
  - 基线分组：**GAN(R-ESRGAN,HAT) / 通用扩散(SeeSR,SupIR,OSEDiff) / 文字向(MARCONet,DiffTSR)**。
  - 指标列：PSNR/SSIM/LPIPS/FID/OCR-A。
  - 主表 FTSR ×4：TADiSR 25.49/0.736/0.152/32.13/**0.662**（逐列粗体最优）。
  - **Real-CE-val(189 对, PP-OCR, Levenshtein ratio, ×4)：24.02/0.829/0.100/38.01/0.882**。
  - 消融：w/o JSD、w/o TACA、w/o MF-Loss（逐组件移除）。
  - 贡献=3 条：框架 + 数据合成管线 + 实验 SOTA。
- **⚠️ 对我们**：单步 + Real-CE 0.882，是 Real-CE 这块最硬竞品。**但其 Real-CE 协议=189 过滤对 + Levenshtein(≈NED) ≠ 我们 valid_list 260 + 官方 ACC**——数字不可直接搬比。

### TIGER（2025-10）—— 区域裁剪指标 + 大基线表范本
- **WHY** SR 在"图像质量 vs 文字可读性"间 trade-off。
- **HOW** 两阶段"先修字、后增图"（text-first, image-later）+ 自建 UZ-ST（×14.29 极端变焦中文集）。
- **WHAT 结构（强烈借鉴）**：
  - 基线 **13 个**：GAN(R-ESRGAN,HAT)/扩散(SeeSR,SupIR,DiffBIR,OSEDiff,DreamClear,TSD-SR,DiT4SR)/文字向(MARCONet,DiffTSR,**TADiSR**)。
  - **指标三组**：整图(PSNR/SSIM/LPIPS/DISTS/FID) + **文字区域裁剪(PSNR_cr/SSIM_cr/LPIPS_cr/DISTS_cr)** + 文字(OCR-A)。← **这套"区域裁剪 _cr 指标"正是我们的 masked-PSNR/SSIM，应采用其命名与表结构**。
  - 主表：双 benchmark 并排(Real-CE | UZ-ST)，每边 12 列。
  - **OCR 依赖消融(Table 5)**：无 OCR 预测文本时仍 OCR-A 40.4% → 鲁棒性卖点。
  - 贡献=3：首个两阶段 text-first 框架 + UZ-ST 数据集 + 实验 SOTA。
- **⚠️ 数字打架（关键）**：TIGER 表里 TADiSR Real-CE OCR-A=**64.7%**，而 TADiSR 自报 **0.882** → **同方法跨论文 Real-CE OCR-A 差悬殊，证明 Real-CE 识别口径不统一、不可跨论文搬数**。我们 Real-CE 必须**自己同协议重测**所有对标项。

### DualTSR（2026-03）—— 与 VOSR 同源 CFM，最新
- **WHY** STISR 依赖外部 OCR 或多分支架构，难训难复现。
- **HOW** 单一多模态 transformer + **双扩散目标：连续(Conditional Flow Matching)建图像 + 离散扩散建文本**，每层交互（MM-DiT 式），去除外部 OCR。
- **WHAT 结构**：基准 CTR-TSR(合成中文)+RealCE(**300 子集**)；基线 通用(ESRGAN/MSRResNet/SwinIR/SRFormer)+文字向(MARCONet/MARCONet++/DiffTSR)；指标 **PSNR/LPIPS/FID/ACC/NED**(×2 与 ×4 两表)；消融增量 (a)Joint-MG→(b)+离散文本→(c)全目标 + 采样步数/CFG 扫描。贡献=3。
- **对我们**：CFM 与 VOSR/LightningDiT 同源，可在 Related Work 作"同为 flow-matching 但他们多分支多步、我们单步蒸馏"的对照；ACC/NED 口径更接近 Real-CE 官方。

---

## 3. 方法家族 + 根基

- **OSEDiff(NeurIPS24)/SinSR(CVPR24)/TSD-SR(CVPR25)**：通用一步蒸馏 real-ISR，**无一文字向** → 我们白地。
- **TextSR(2025-05)**：字符 ByT5→cross-attn 注入 + CFG 双条件 + **迭代 OCR 重条件**；5 步 DDIM。**与 E1 同源**，但靠多步+迭代才奏效；我们在单步蒸馏下发现该路被抹平（gate→负）→ E1 是反衬对照而非卖点。
- **DiffTSR(CVPR24)**：IDM+TDM+MoM，中文，多步；常见 baseline（TADiSR 已超）。
- **REPA(ICLR25 Oral)**：通用 SSL 特征(DINOv2等)对齐 DiT 隐藏层加速生成训练。**2a=其 OCR 域特化 + 区域局部化 + 用于蒸馏/可读性**；"通用 DINO-REPA 无效、OCR-REPA 有效"是对 REPA 的实质扩展。必引划界。

---

## 4. 给我们论文的硬结论

1. **差异化主轴=一步蒸馏 + 蒸馏均衡器**（不是"OCR 条件/文字感知"，那已被 TextSR/TADiSR/TeReDiff/TIGER/DualTSR 占满）。
2. **novelty 三块白地**：① 一步蒸馏 × text-SR；② 蒸馏均衡器现象（无人报告）；③ 通用 REPA 对文字无效需 OCR 特化（无人验证）。
3. **表结构借鉴**：基线按 GAN/通用扩散/文字向/Ours 分组；指标整图 + **区域裁剪 _cr** + OCR；双 benchmark 并排；消融增量式（2a→2d→2a+2d 正好契合）。
4. **必做的口径动作**：Real-CE 上**所有对标项自己同协议(valid_list 260 + 官方 CRNN ACC/NED)重测**，绝不跨论文搬 OCR-A 数（TADiSR 0.882 vs TIGER 记 64.7% 即铁证）。
5. **我们独有卖点要写进 intro/消融**：推理期**不需要 OCR/不需要文字标注**（TextSR/TIGER 推理依赖 OCR；我们 OCR 只在训练期监督），且 **NFE=1**（~50× 快）。

---

## 5. 对比设计（核心：怎么比 Real-CE）—— 2026-06-22 补

### 两件已查清的事实
1. **文本 SR 有两种不兼容任务形态**，口径天然分裂：
   - (a) 行级 STISR：裁好单行(≤24字,128×512)整条过识别器。代表 TextZoom / **DiffTSR** / CTR-TSR / TSRN/TBSRN/TATT/MARCONet。
   - (b) 整图场景复原：整张退化图,逐区域/spotting。代表 **TAIR / TADiSR / TIGER / TEXTS-Diff / 我们**。
   - **我们在 (b);DiffTSR 在 (a)** → 直接同表是 apples-to-oranges。
2. **每篇 Real-CE 协议各自定,数字互不可比**：DiffTSR(1531裁剪/TransOCR/ACC·NED) · TADiSR(189/PP-OCR/Levenshtein) · DualTSR(300/ACC·NED) · TEXTS-Diff(2718/PP-OCRv5整行精确) · 我们(260/官方CRNN/ACC·NED)。铁证 TADiSR 0.882 vs TIGER 记 64.7%。

### TEXTS-Diff(ICASSP26) 情报 —— 冲击 C1
- **已是一步文本 SR**（"follow the one-step diffusion technique of OSEDiff … in one step"）+ 输入侧文字线索(abstract concept + concrete region)。基线 Real-ESRGAN/StableSR(4)/DiffBIR(50)/FaithDiff(20)/SeeSR(50)/SUPIR(50)/OSEDiff(1);指标 OCR-A(PP-OCRv5整行)+PSNR/SSIM/LPIPS/DISTS/FID+NIQE/MANIQA/MUSIQ/CLIPIQA;数据 Real-CE(2718)/Real-Texts(自建1000)/RealSR。
- **影响**："首个一步文本 SR" claim 失效 → **一步/高效=入场券,护城河移到 C2(蒸馏均衡器)+C3(双空间student监督,推理不需OCR)**。
- 利好:PP-OCRv5整行匹配=我们协议①同源 → 自建 PP-OCR 内轴更站得住。
- 投稿时未开源 → 不对比,Related Work 点名。

### DiffTSR(CVPR24) —— 唯一开源文字向,但形态不同
- 开源(权重+推理+训练,GoogleDrive/BaiduDisk);Real-CE 用 1531 裁剪对 + TransOCR;**行级形态**。
- 处理:要么注明形态差异仅在 Related Work 引;要么按区域裁剪喂入我们统一协议跑一行(标注"行级方法,裁剪输入")。

### 领域如何处理 Real-CE 训练(2026-06-22 从 PDF 查实)
**Real-CE 自带 GT box + 文字转录**(train 1935对/23,547行) → 可上框级/文本级监督。两阵营:
- **阵营A 零样本泛化**:DiffTSR —— "all methods not trained on Real-CE, 评 generalization"。
- **阵营B 混入 Real-CE train(最新最强全在此)**:
  - **TADiSR**:FTSR(合成45k,带mask)+Real-CE train 337;滤错配1935→337;基线也在 FTSR+Real-CE 微调。
  - **TIGER**:合成+Real-CE 337(重标注)+UZ-ST,两阶段(阶段1合成+真实裁剪区域);基线在训练集微调。
  - **TEXTS-Diff**:Real-Texts(33,875)+Real-CE train 14,312裁剪(带标注)+20k。
  - **Real-CE RRDB**:纯 Real-CE train 训练。
- **共性**:全是把 Real-CE train **混进**自己合成/mask 数据**联合训练**(非顺序微调);合成供 mask+量,Real-CE 供真实域;滤错配/部分裁剪到文字区域。
- **对我们**:正解=把 Real-CE train(带框/转录)混入蒸馏训练,上 2a-GT+2d 监督(主场优势);出两 setting(S1零样本泛化=一致性故事,S2混训=竞争数字)。详 `ToDoExp/03`。

### 我们的对比设计(定调,2026-06-22 修订为"验证-引用"轻量策略)
> 用户拍板:**不跑全部基线;验证表可信度后引用 published 数字**。"验证 SOTA 再引用"**只对单一协议的已发表表成立**。
- **TAIR(主场)=原生 TESTR 协议,直接引 TAIR Table 3**(DiffBIR/SeeSR/SUPIR/FaithDiff/TeReDiff)+我们的行。**验证已完成**:HQ 校准 + SOTA TeReDiff 已复现 → **0 新跑**。
- **Real-CE(客场)=引 Real-CE 论文官方协议基线(RRDB/GAN)+我们的行**。**验证已完成**:HR oracle 0.4836≈0.4807、LR 0.275≈0.2759 → **0 新跑**。
  - 协议(我们自评 + 任何补跑统一):valid_list 260 · 官方CRNN ACC/NED(+附PP-OCRv5整行) · 整图PSNR/SSIM/LPIPS(+可选区域裁剪_cr)。
  - **扩散SOTA(TADiSR/TIGER/DualTSR/TEXTS-Diff)不开源+协议各异→无法验证后引用→仅Related Work点名**。
  - **唯一可选新跑**:补 1-2 个开源扩散对照点(OSEDiff首选/SeeSR),因 Real-CE 论文 2023 无扩散基线。详 `ToDoExp/01_*.md`。
- **公平性**:扩散基线零样本同台;我们微调行单独列,勿混。

**Sources**：见 `sources/README.md`。
