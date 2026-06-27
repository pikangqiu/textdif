# 实验素材（论文向）

> 建立 2026-06-22。本文是**面向论文表格/正文**的结果蒸馏：headline 数字 + 每个结果进哪张表。
> **完整设计/配置/机制/失败实验的单一事实来源 = `../doc/result_exp.md`**（本文只做指针 + 论文向裁剪，不重复细节）。

---

## 1. Headline 数字（NFE=1，Real-Text 847）

| 方法 | 角色 | CharAcc(①) | Det F1(②) | E2E None/Full(②) | PSNR(①) |
|---|---|---|---|---|---|
| Teacher (多步) | 上游 | 63.78 | — | — | 24.86 |
| **B: guided target** | **一步蒸馏 baseline** | 65.99 | — | — | 24.62 |
| 2a: local OCR-REPA(w0.5) | 特征对齐冠军(无需GT文本) | 66.9–67.7 | 76.3±0.3 | 42.8±0.8 / 52.4 | 24.86–24.97 |
| 🏆 2d: 真 GT 文本 CTC | E2E/保真冠军 | 68.28 | 77.03 | 44.60 / 54.68 | **25.08** |
| 🏆 2a+2d 联合 | 识别/检测双冠军 | **68.74** | **78.32** | 42.59 / 52.71 | 24.67 |

四反例(均无增益,反衬 2a 特异性)：E1 TrOCR-cond 65.80 / 2c detmap 65.76 / dino_repa 66.00 / seg_token 64.86；协议② det 全卡 71–72。

对标(协议②,同口径)：TeReDiff 复现 None 33.53/Full 44.62(论文宣称 49.39/56.45,官方权重未复现)；FaithDiff None 41.64/Full 47.97 → **2a/2b 超全部 published 常规扩散基线且 NFE=1**。

---

## 2. 结果 → 表/图映射

| 论文位置 | 内容 | 数据来源(result_exp.md 章节) |
|---|---|---|
| 主表 Table A (TAIR 协议②) | TESTR Det F1 / E2E None+Full,基线分组 | §3.2 |
| 主表 Table A 辅列 (协议①) | CharAcc/EM + PSNR/SSIM/LPIPS/感知 | §3.1 |
| 主表 Table B (Real-CE 协议③) | CRNN ACC/NED + (区域裁剪)PSNR/SSIM + LPIPS | 跨域评测节(零样本+真实配对微调) |
| 消融 Table C | 四反例 + w 扫描(倒U) + 2a/2d/2a+2d 增量 | §2 各节 + w扫描节 |
| 跨域一致性 Table D | PP-OCR CharAcc 同轴 847 vs Real-CE,证排序保持 | 跨域评测节 |
| 机制图 Fig (C2) | teacher 侧 metric 增益→蒸馏后回落 baseline | 结构增强①②节 |
| Scaling Table | 0.5B vs 1.4B 同口径(shortcut) | 1.4B 节(待补同口径) |
| 效率 | NFE=1 vs 50,~50× | §3.3 |

---

## 3. 待补实验状态（对应 writing_strategy §7）

| # | 实验 | 状态(2026-06-22) | 卡点 |
|---|---|---|---|
| 2 | Real-CE 真实配对微调(同口径对 published RRDB ACC 0.3093) | 在线版无灾难性遗忘;真实配对 GT-box 版排队/等卡 | 本机/199 等卡 |
| 3 | 跨数据集 PP-OCR 一致性表 | 零样本排序已验证一致(B<2a-GT<2d<2a+2d) | 微调后补行 |
| 4 | 1.4B 同口径(shortcut)放大 | 1.4B simple 蒸馏中(~14h);shortcut 同口径未跑 | 算力;口径决策 |
| 5 | E2(E1+2a)收口 | 199 训练中 | — |
| 6 | 蒸馏均衡器 before/after 机制图 | 数据已有(结构增强①②),待画 | 画图 |

> **已知风险**：2d/2a+2d 主表冠军为单 ckpt(20000),用户判不补噪声带——若审稿质疑 cherry-pick 再回补(#1)。

---

## 4. 对比表策略（2026-06-22 修订：验证-引用，不跑全部基线）

- **不可跨论文搬 Real-CE 扩散方法数**(铁证:TADiSR 0.882 vs TIGER 记同方法 64.7%)。
- **TAIR 表**：直接引 TAIR Table 3 published(已验证:HQ 校准 + TeReDiff 复现)→ **0 新跑**。
- **Real-CE 表**：引 Real-CE 论文官方协议基线(RRDB/GAN,已验证 HR/LR 复现)+ 我们的行 → **0 新跑**；扩散 SOTA 不开源仅 Related Work 点名。
- **唯一可选新跑**：补 1-2 个开源扩散对照点(OSEDiff/SeeSR)入 Real-CE 表，因 Real-CE 论文无扩散基线。详 `ToDoExp/01_realce_baseline_reeval.md`。
- 统一协议(自评+补跑)：valid_list 260 · 官方CRNN ACC/NED(`scripts/eval_realce.py`)+附PP-OCRv5整行 · 整图PSNR/SSIM/LPIPS(+可选_cr)。
- 公平性：扩散基线零样本同台；我们微调行单独列，勿混。

> DiffTSR 仓库**不含 Real-CE eval**(仅 inference+train+自带行级 CTR-TSR testset)；形态为行级,不进整图主表。
