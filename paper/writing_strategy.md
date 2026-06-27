# 论文写作主思路（AAAI · 文本超分一步蒸馏）

> 建立 2026-06-22。本文是论文写作的"主线 + 骨架 + 待决策"，与 `related_work_scan.md`（文献定位）、`experiments_for_paper.md`（结果素材）配套。
> 实验单一事实来源仍是 `../doc/result_exp.md`。

---

## 1. 基调

**带 insight 的方法论文**：以"方法"为外壳（可复现一个东西），以"发现"为内核（reviewer 记住的是 insight）。
intro 不从"我们提出 XXX 模块"开头，而从**反常现象**开头——审稿人对"+1.3 点"免疫，对"大家都在 teacher 上下功夫其实白费"会停下来读。

---

## 2. 叙事主线（intro 漏斗，五步）

1. **任务定位**：文本超分目标是**可读性**而非像素保真；扩散是可读性 SOTA，但**多步慢 + 多步采样引入字符漂移**（伏笔）。
2. **反直觉①**：多步蒸成单步，本以为掉点——**识别率反升**（teacher 63.78 → 单步 B 65.99）。机理：单步消除多步采样的字符随机漂移。→ "单步不是妥协，是 OCR 杠杆"。
3. **反直觉②**：要更进一步，自然想法是"加强 teacher / 注入 OCR 语义 / 上通用 REPA"——**逐一试，全被"蒸馏均衡器"抹平**（结构增强①② teacher 涨→蒸馏后回 baseline；E1 的 gate 自学到负主动关闭；dino/seg/检测图一致性全卡 baseline）。这是论文靶心。
4. **我们的答案**：唯一穿透均衡器的是 **student 侧、细粒度、OCR 特异**监督，且在**两个互补空间**同时做——特征空间(区域 OCR-REPA) + 输出空间(真 GT 文本 CTC)，构成"识别+定位 / 端到端+保真"两个 Pareto 冠军。
5. **结果**：NFE=1 下 TAIR(TESTR) + Real-CE(CRNN) 双基准可读性 SOTA，快 ~50×，优势跨数据集/跨识别器保持。

> 力量点：第 3 步的"四个反例"既是 motivation 又是 ablation——**台账里的 null result 全部变正资产**。

---

## 3. 贡献列表（三个不同抽象层级，防"塌缩成1.5个"）

> **2026-06-22 重大修订**：TEXTS-Diff(ICASSP26) 已是一步文本 SR(OSEDiff 式)，**"首个一步文本SR"claim 失效**。一步/高效降为**入场券**，护城河移到 C2+C3。C1 改为"蒸馏路线 + 字符漂移发现"，不再吹"首个一步"。

- **C1（框架级，已弱化）** 用 **flow-matching teacher 的 shortcut 蒸馏**得到单步文本 SR，并揭示"单步蒸馏=字符漂移消除器"（提速 ~50× 同时**提升**可读性）。差异化 vs TEXTS-Diff：我们是**蒸馏多步teacher**而非OSEDiff式直训，且**推理不需OCR/文字标注**。
- **C2（原理级，主护城河）** "蒸馏均衡器"现象与诊断：teacher/输入/通用表征对齐的文字先验均被蒸馏抹平，**唯 student 侧细粒度 OCR 监督穿透**；四个正交反例严格界定特异性。
- **C3（方法级）** 双空间 OCR 监督（特征空间 OCR-REPA + 输出空间 GT-text CTC）及其互补性，给出两个 Pareto 最优解。
- **C4（基准/测量级贡献，2026-06-27 降级定调）** **跨阵营统一的可复现再评测**（详 `eval_protocol.md`；**不包装成"发明新协议/新指标"**——ΔOCR-A 属 TIGER、OCR-A/区域IQA/PP-OCR 皆前人现成，硬卖"新协议"会被审稿人以 TIGER/TADiSR 打"自肥+不新"）。我们**独有、别人没有的三样**：
  1. **跨阵营桥接**：扫描前人发现领域裂成 **TAIR spotting 阵营** 与 **Real-CE 区域识别阵营**，**无人在同一批图同跑两套**；我们在同图同管线上对齐 L1(TESTR spotting) + L2(PP-OCR 区域识别)，第一次让两套口径可比。
  2. **可复现审计 + 自评纪律**：TeReDiff published E2E 49.39 不可复现的溯源链(md5 逐字节 + IQA 三方对拍 + HQ 校准准)——领域里没人做的"published 数不可信"实证；所有方法冻结管线自评，published 仅作标注旁注。
  3. **NFE 列 + det 与 E2E 同报**：摆上效率与"低召回→E2E 虚高假象"。
  揭示三点 finding：(i) published text-SR 数不可复现；(ii) 两阵营指标对方法排序不一致；(iii) 通用扩散 ΔOCR-A<0(引 TIGER 已证)而我们一步法>0。ΔOCR-A/OCR-A **借用并 credit 前人**。参考实现 `/data/ywk/eval`，重心放在「桥接演示 + 可复现审计」。**定位**：分量轻于方法级但**真**，使论文成「方法(C2) + 基准(C4) + 效率」三条腿，化解"只有一个 loss"。

> 备选叙事：以 C2 为唯一强 thesis、C1/C3 为支撑（**现更推荐**，因 C1 被 TEXTS-Diff 削弱）。**【待决策1】见 §6。**

> **2026-06-22 证据审视后定调（见 `story_evidence_audit.md`）**：**C2 = 承重墙(thesis)**，证据最满(四正交反例+结构增强 before/after)。**C1/C3 不承重**：C1 只讲"效率入场券+均衡器载体"，不强吹漂移机制(未隔离)、scaling 修好或砍；C3 改写为"特征空间+输出空间监督建设性叠加(det 双新高)"，**精排序锚 det 不锚 CharAcc**(2a+2d−2d=+0.46、2d−2a峰=+0.62 均<1分自定噪声线)。粗结论"OCR-aware >> baseline 与所有非OCR对照(>1分)"很稳。**Real-CE 写成跨域一致性,非第二SOTA**。

---

## 4. 章节骨架

```
1 Introduction      — §2 五步漏斗 + §3 三贡献
2 Related Work      — ①STISR(TextZoom)为何不再适用→切割 ②扩散文本SR/可读性(TextSR/TADiSR/DiffTSR/TeReDiff/DualTSR) ③一步蒸馏(OSEDiff/TSD-SR,均非文字向) ④REPA/OCR监督
3 Preliminary       — flow-matching teacher + 一步蒸馏底座(B baseline)
4 Method
  4.1 一步蒸馏框架 + 字符漂移分析        (C1)
  4.2 蒸馏均衡器：什么穿不透(含四反例设计) (C2)  ← 反例写成方法的一部分,不是堆在实验里
  4.3 双空间 OCR 监督 OCR-REPA + GT-CTC   (C3)
5 Experiments
  5.1 Setup：双benchmark + 三层评测口径
  5.2 主表：TAIR(TESTR) + Real-CE(CRNN)   ← 基线按 GAN/通用扩散/文字向/Ours 分组;指标整图+区域裁剪_cr+OCR
  5.3 消融：四反例 + w扫描 + 2a/2d/2a+2d互补性(增量式)
  5.4 跨域一致性(PP-OCR CharAcc 同轴跑847+Real-CE) + 蒸馏均衡器 before/after 机制图
  5.5 Scaling(1.4B,同口径) + 效率(NFE/时延)
6 Conclusion
```

---

## 5. 评测口径（三层，写进 5.1）—— 不统一,要"原生协议对外 + PP-OCR 内轴"

| 度量 | 代码 | 识别器 | 用在 | 角色 |
|---|---|---|---|---|
| ①PP-OCR CharAcc | 199 OSEDiff `metric_*.py` | PP-OCRv5 + GT-mask | Real-Text 847 | **内部排序主轴 + 跨域一致性轴** |
| ②TESTR E2E None/Full | `TAIR/realtext_eval/eval_spotting.py` | TESTR | Real-Text 847 | **可引 TAIR 表的对外口径** |
| ③Real-CE CRNN ACC/NED | `scripts/eval_realce.py` | Real-CE 官方 CRNN | Real-CE val | **可引 Real-CE 表的对外口径** |

铁律：**每个 benchmark 用其原生协议对外（可对published表）；PP-OCR CharAcc 作跨 benchmark 内轴证"优势与识别器无关"。不要追求一份统一 eval。**
Real-CE 雷：TADiSR 0.882 vs TIGER 记 TADiSR 64.7% → **Real-CE 数字跨论文不可搬，所有对标项必须自己同协议重测**。

---

## 6. 待决策（开始写前要拍板）

- **【待决策1】重心/标题**：A 效率重心《One-Step Diffusion for Readable Scene Text SR》(稳妥平庸) / B insight 重心《What Survives Distillation? ...》(记忆点强,要求 C2 写硬) / C 混合(主标题方法+副标题 insight)。
- **【待决策2】C2 强度**：仅经验观察(四反例+机制图,够 AAAI) vs 补机理小分析(蒸馏"平均化"为何抹平 teacher 信号)。
- **【待决策3】贡献编排**：三正交层级(C1/C2/C3) vs 单强 thesis(C2 为主)。

---

## 7. 关联实验取舍（用户 2026-06-22 拍板，详见根 memory paper-experiment-priority）

- **要做**：#2 Real-CE 真实配对微调(同口径对 published)、#3 跨数据集 PP-OCR 一致性表、#4 1.4B 同口径(shortcut)、#5 E2 收口、#6 蒸馏均衡器机制图。
- **不做**：#1 冠军噪声带(已知 cherry-pick 风险)、#7 SA-Text(污染,仅附录)。
