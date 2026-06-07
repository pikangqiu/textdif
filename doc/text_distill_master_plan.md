# Text-SR 一步蒸馏研究：全局计划 + 实验记录（主文档）

更新时间：2026-06-07 ｜ 分支：`text-distill-ablation`

> 本文是**总览/索引**：研究故事、创新点、分阶段计划、每个实验的状态与历史结果。
> 可执行命令（训练/推理）见配套 `doc/text_distill_all_experiments_runbook.md`。
> 概念推导见 `doc/VOSR_text_SR_distillation_summary.md`。

---

## 0. 一句话故事

> 在 VOSR 的 vision-only 一步 flow 蒸馏框架里，**teacher target 的构造方式**（而非单纯
> 加 OCR loss）是文本可识别性的主要来源；在此基础上把表征对齐**局部化**、再升级为**直接
> 文字监督**，进一步突破 OCR 天花板。

核心问题：one-step 文本识别率提升，是否来自 **guided teacher target**（+ 可选 RC），
而不是单纯 OCR loss？—— 当前证据支持「target construction 是主变量」。

---

## 1. 三个创新点（论文骨架）

1. **Restoration-oriented guided-target 蒸馏更利于文本**：teacher target
   `v_T^g = v_T^p + ω·(v_T^f − v_T^p)`（ω=`cfg_scale`，partial→full 条件，非标准 CFG）。
   实验证明中间 guided target（ω=0.5, B）OCR 优于 full target（ω=1.0, A）和多步 teacher。
2. **局部化的 OCR 表征对齐**（2a）：把 REPA 从全局 mean-pool 改成**文字框区域序列**对齐，
   论证「REPA 用于文本 SR 必须局部化」。
3. **HR 伪标签 + CTC 直接文字监督**（2b）：无需人工标注，用冻结识别器在 HR 上自生成伪标签，
   把"可读性"直接写进蒸馏梯度。

辅助支线：FD-Loss 分布对齐、RC/RCGM 一致性、（未来）VAE encoder LoRA / CODSR 噪声加权。

---

## 2. 评测口径（务必统一）

- 评测集：**847 个有效图像对，1945 个 OCR regions**；输入 `/data/ywk/datasets/real_test/LQ`。
- 三个 OCR 指标：**OCR-EM**（exact match / word acc）、**OCR-CER**（字符错误率↓）、
  **OCR-CharAcc**（字符准确率↑）。用户口语说的 **“65.99”就是 B 的 CharAcc=0.6599**。
- **全方法统一用固定步数 last checkpoint（默认 20k）；绝不在测试集上挑 checkpoint。**
  小数点差异先用「最后 3–5 个 checkpoint 的 mean±std」卡噪声，差距 < std 视为平手。

---

## 3. 共同设置（所有实验单一变量）

| 项 | 值 |
|---|---|
| teacher / student init（同一权重） | `exp_vosr_text/...checkpoint-00040000/clean_weights/ema_model.safetensors` |
| backbone / VAE / 视觉条件 | 0.5B DiT(d1024/b28/h16/ps2) / SD2.1 4ch f8 / DINOv2-b L8（vision-only，无文字条件） |
| 分辨率 ×上采样 / 训练步数 | 512 ×4 / 20000（每 1000 存） |
| 蒸馏方式 | 一律 `distill_type: shortcut`；**B 及所有加项 `u_weight=0`（纯 guided-target，shortcut 一致性未生效）** |
| 退化 | 在线 RealESRGAN + RandomCrop |

> ⚠️ 局限（写进 limitation）：所有结论 conditional on **同一 init / 同一 backbone**。
> 主机制主张建议至少换一个 init 复现一次。

---

## 4. 分阶段路线图（每阶段在前一阶段冠军上叠加）

```
Stage 0  多步 teacher（蒸馏前基准/上界）
   │
Stage 1  蒸馏目标消融  A(full) / B(guided) / C(guided+RC)        ← 找到主变量
   │      结论：B 是 OCR 冠军；guided target > full > RC
   ▼
Stage 2  REPA 表征对齐（B + dino/ocr/seg，全局 pool）            ← 全局对齐已饱和
   │      结论：REPAocr 全局 pool 65.81 ≈ B 65.99，对 OCR 无增益
   ▼
Stage 3  局部 OCR 监督（本轮重点，B 之上）
   │      2a  local-crop OCR-REPA（区域序列特征对齐）   ← 进行中
   │      2b  HR 伪标签 + CTC 识别 loss（直接文字监督） ← 进行中
   │      2c  文字检测图一致性                          ← 待定
   ▼
Stage 4  分布对齐 FD-Loss（B / shortcut 之上）                   ← 支线
```

### 2a / 2b / 2c 三级阶梯（Stage 3 细节）

| 级 | 名称 | 监督信号 | 文字粒度 | 与上一级的递进 |
|---|---|---|---|---|
| — | REPAocr（旧） | 整图 OCR 特征 **全局 mean-pool** cosine | 1 向量 | 丢失局部结构 → 饱和 |
| **2a** | local-crop OCR-REPA | 文字框区域**序列特征** cosine | 每框 1 段序列 | 把对齐**局部化** |
| **2b** | HR 伪标签 + CTC | pred crop logits 对 HR 伪标签算 **CTCLoss** | glyph 级 | 从"对齐特征"到"直接监督识别结果" |
| **2c** | 检测图一致性 | 文字**检测响应图**一致 | 像素级 mask | 进一步空间约束（未实现） |

2a 损失：`L = L_guided_distill + ocr_repa_weight · (1 − cos(feat(pred_crops), feat(HR_crops).detach()))`
2b 损失：`L = L_guided_distill + ocr_ctc_weight · CTCLoss(rec_logits(pred_crops), greedy_decode(HR_crops))`

> 2a/2b **隔离**：独立配置键、纯加法、默认关；CTC 用确定性贪心解码不耗 RNG；
> `tests/test_ocr_repa.py::test_2b_does_not_perturb_2a` 钉死同输入下 2a loss 不变。
> 识别器/检测器全程冻结，梯度只**穿过**它们回流到预测图（再到 student）。

---

## 5. 历史实验结果（Stage 0–1，已完成）

评测集：847 对 / 1945 regions。**粗体=该列最优。**

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | NIQE↓ | MUSIQ↑ | MANIQA↑ | **OCR-EM↑** | **OCR-CER↓** | **OCR-CharAcc↑** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher MS（多步，上界） | **24.8620** | 0.7883 | 0.2692 | 0.2087 | 6.6045 | **54.9036** | 0.4040 | 0.5620 | 0.3622 | 0.6378 |
| A: full target (ω=1.0) | 24.5173 | 0.8329 | **0.2538** | 0.2189 | 10.7384 | 53.4998 | 0.4156 | 0.5789 | 0.3418 | 0.6582 |
| **B: guided target (ω=0.5)** | 24.6187 | **0.8334** | 0.2541 | 0.2200 | 10.9392 | 54.7819 | **0.4295** | **0.5841** | **0.3401** | **0.6599** |
| C: guided + RC (rcgm) | 24.2763 | 0.7864 | 0.2550 | **0.2078** | **6.5167** | 54.8537 | 0.3819 | 0.5548 | 0.3599 | 0.6401 |

**结论**
1. A/B 的 OCR 均 **优于多步 teacher** → 一步蒸馏不只是加速，还减少多步采样的字符漂移。
2. **B = OCR 冠军**（EM/CER/CharAcc 全胜 A）→ guided target > full target。
3. C 改善 DISTS/NIQE 但 **损害 OCR** → VOSR 通用 SR 的 RC/RCGM recipe 不能直接照搬到文本。

**收敛性**：A/B/C 均训到 20k；A/B 的 loss 无持续下降，差异不能归因于"没收敛"；
C 的一致性项仍在学但 OCR 已差于 B（目标方向与 OCR 不一致）。

### Stage 2（REPA 表征对齐）已知点
- **REPAocr（全局 pool）CharAcc 65.81 ≈ B 65.99** → 全局表征对齐对 OCR 已饱和，
  正是 2a 局部化的动机。
- DINO/seg REPA：已有 config 与部分 checkpoint，结果待统一评测（见 runbook）。

---

## 6. 实验状态总表

| Stage | tag | 含义 | config 后缀 | 状态 |
|---|---|---|---|---|
| 0 | Teacher MS | 多步基准 | （teacher ckpt） | ✅ 已测 |
| 1 | A | full target ω=1.0 | `_full_target_no_rc` | ✅ 已测 |
| 1 | **B** | guided target ω=0.5（冠军） | `_guided_target_no_rc` | ✅ 已测 |
| 1 | C | guided + RCGM | `_guided_target_rc` | ✅ 已测 |
| 1 | shortcut | u_weight=1.0 真 shortcut | `_guided_target_shortcut` | ✅ 已训 |
| 2 | dino / dino_token | DINOv2 REPA | `_dino_repa` / `_dino_token_repa` | ⏳ 待统一评测 |
| 2 | ocr | OCR 全局 pool REPA | `_ocr_repa` | ✅ 65.81 |
| 2 | seg / seg_token | 分割 REPA | `_seg_repa` / `_seg_token_repa` | ⏳ |
| 3 | **ocr_local (2a)** | 局部区域序列对齐 | `_ocr_local_repa` | ▶️ 本轮训练 |
| 3 | **ocr_ctc (2b)** | HR 伪标签 + CTC | `_ocr_ctc` | ▶️ 本轮训练 |
| 3 | 2c | 检测图一致性 | — | 📝 待定 |
| 4 | no_rc_fd / shortcut_fd | FD-Loss 分布对齐 | `_fd` / `_shortcut_fd` | ⏳ |

---

## 7. 下一步（优先级）

1. 跑完并评测 **2a / 2b**（本轮重点），与 B 对比 OCR-EM/CER/CharAcc。
2. 关键方法补测最后 3–5 个 checkpoint 的 **mean±std**，把小数点差异卡进噪声。
3. 统一评测 Stage 2 的 dino/seg REPA，补齐 locality ablation 一行（global vs region vs CTC）。
4. 修日志：分开记录 `v_loss / u_loss / repa_loss / ocr_repa_loss / ocr_ctc_loss`（部分已加）。
5. 换一个 init/backbone 复现主机制主张（否则列 limitation）。

---

## 8. 风险 / 已知问题

- OCR 评估本身有噪声；务必统一口径、报 mean±std。
- 旧 REPA 第一版是全局 pool，对文本局部不敏感（2a 已修正）。
- 在线 RealESRGAN 退化不一定完全匹配真实文本退化。
- C 的 `v_loss` 日志历史上等于 total loss（命名不准），机制分析看新日志。
- 结论目前 conditional on 单一 init / 单一 DiT 架构。

---

## 9. 相关文档

| 文档 | 内容 |
|---|---|
| `doc/text_distill_all_experiments_runbook.md` | 全实验训练/推理命令 |
| `doc/text_ocr_local_repa_experiment.md` | 2a/2b 原理与隔离契约细节 |
| `doc/VOSR_text_SR_distillation_summary.md` | 概念推导 / Stage1-3 / 损失公式 |
| `doc/text_fd_loss_experiment.md` | FD-Loss 实验 |
| `doc/current_text_distillation_context.md` | A/B/C 上下文与收敛性原始记录 |
| `doc/text_distill_ablation_vs_vosr_distillation.md` | 与原 VOSR 蒸馏的对比 |
