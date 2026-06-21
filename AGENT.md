# AGENT.md — 项目上下文与文档索引

> 给所有 AI / 协作者的第一入口。读完本文即可建立项目全貌、找到所有关键文档、并知道在共享服务器上操作的硬约束。
> 主代码仓：`/data/ywk/VOSR`（分支 `text-distill-ablation`）。最后更新：2026-06-21。

---

## 1. 项目是什么

**VOSR 文本图像超分 · 一步蒸馏（text-SR one-step shortcut distillation）。** 目标：把多步 flow-matching 的文本 SR 教师模型蒸馏成 **NFE=1（单步）** 的学生，在保持/提升**文字可读性**的前提下做实时超分，并形成一篇论文。

- **学生/教师骨干**：LightningDiT（0.5B = d1024/b28/h16 + SD2 f8c4 VAE；1.4B = d1536/b36/h24 + Qwen-Image f8c16 VAE）。
- **核心方法线（消融阶梯）**：在一步蒸馏基线上叠加「文字感知」监督/先验：
  - **2a**：局部裁剪 OCR-REPA（用 OCR recognizer 特征对齐，在线 DBNet 检测框）→ **当前识别冠军**。
  - **2a-GT**：同 2a 但用数据集 GT 框（`restoration_dataset.json` 预索引）——检测器 vs GT 的干净消融。
  - **2b**：GT-box CRNN 伪标签 CTC（无突破）。
  - **2c**：detection-map 监督。
  - **2d**：真实 GT 文本 CTC → **检测冠军**。
  - **2a+2d 联合**：识别/检测双新高（两个互补冠军）。
  - **E1**：OCR 语义条件注入（输入侧先验，DINO 式 cross-attn，推理时保留）。
  - **E2**：E1 + 2a-GT 联合（**当前在 199 训练中**）。
- **评测（双协议，缺一不可）**：
  - **协议①** PP-OCR + pyiqa：OCR-CharAcc / EM / CER + PSNR/SSIM/LPIPS/DISTS/NIQE/MUSIQ/MANIQA。
  - **协议②** TESTR 官方 spotting（TAIR 论文协议）：Det F1 / E2E-None / E2E-Full。
  - **铁律**：E2E 必须与 Det F1 一起读（低检测方法会出现虚高的 E2E 伪影）。
- **基准**：Real-Text 847 对（已逐字节核对 = TAIR arXiv 2506.09993 公开测试集，可直接引用论文表）。训练集 119,495 个 `sa_*` 裁剪与其零重叠；SA-Text-test 因母图污染弃用。

---

## 2. 文档索引（按用途）

### 📐 论文思路 / 方法设计
| 文档 | 内容 |
|---|---|
| `doc/项目说明.md` | 项目总体说明（最全的中文背景）。 |
| `doc/VOSR_text_SR_distillation_summary.md` | 文本 SR 蒸馏方法总结。 |
| `doc/text_distill_master_plan.md` | 实验主计划（阶梯/路线图）。 |
| `doc/current_text_distillation_context.md` | 当前蒸馏上下文与设计取舍。 |
| `doc/text_distill_ablation_design.md` | 消融实验设计。 |
| `doc/text_distill_ablation_vs_vosr_distillation.md` | 本消融线 vs 原 VOSR 蒸馏的对比定位。 |
| `doc/ocr_cond_injection_plan.md` | **E1/E2** OCR 条件注入方案设计。 |
| `doc/text_ocr_local_repa_experiment.md`、`doc/text_distill_2a_breakthrough_analysis.md` | **2a** 局部 OCR-REPA 的机制与突破分析（最深）。 |
| `doc/text_fd_loss_experiment.md`、`doc/vae_lora_ocr_repa_experiment.md`、`doc/2026-06-05-vae-lora-ocr-repa-implementation-plan.md` | 其他支线（FD loss / VAE-LoRA）。 |

### 🧪 实验进展 / 结果（**单一事实来源**）
| 文档 | 内容 |
|---|---|
| **`doc/result_exp.md`** | ⭐ **实验总记录**：每个实验的动机/配置/机制/训练设置/结果 + 统一对比总表（§0 评测协议、§2 各实验、§3 总表与结论）。**每做完一个实验必须更新此文**。 |
| `doc/text_distill_all_experiments_runbook.md` | 全实验运行手册（命令/流程）。 |
| `doc/text_distill_ablation_commands.md`、`doc/text_repa_experiment_commands.md` | 训练/推理/评测命令集。 |
| `doc/text_distill_convergence_check.md` | 收敛性检查。 |
| `doc/realce_benchmark.md`、`doc/2a_ocr_repa_figure_prompt.md` | RealCE 基准 / 论文配图提示。 |

### 🖥️ 计算资源（服务器/数据/路径）
| 文档 | 内容 |
|---|---|
| **`doc/compute_resources.md`** | ⭐ **算力总览**：3 台服务器的地址/登录/GPU/代码仓/数据/当前占用 + 通用铁律 + E2→199 迁移模板。 |
| `doc/node199_拓扑与配置原理.md` | 199 的 NVLink 拓扑与多卡（有效 batch 16）配置原理与经验。 |

---

## 3. 服务器使用原则（硬约束，必读）

> 本项目运行在 **3 台多人共享服务器**上，资源纪律是第一原则。完整细节见 `doc/compute_resources.md` §0。

1. **绝不杀任何别人的进程。** 只能停自己启动的 screen/进程；缺显存就**等**，不抢不杀。
2. **数据/产物只写各机指定目录**：本地 `/data/ywk/`、199 `/data2/wyw/ywk/`、226.31 `/data07/dt_data/ywk/`（其他盘视为满/只读）。
3. **不改远端 conda 环境**（不装包、不改依赖）。
4. **长任务用 `screen`，不用 `nohup`**；启动器带重试循环。
5. **远端无外网**：HF 用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；dinov2 用本地缓存仓 `source='local'`。
6. **远程操作助手**：199 用 `/data/ywk/claude_tools/{rexec,rput,rget}.py`；226.31 用 `{rexec_dt,rput_dt,rget_dt}.py`。
7. 三台机的代码会**分叉**（本地是 OCR-REPA/OCR-cond 超集；226.31 base main 无该栈；199 VOSR_textsr 有 2a 栈）。迁移要 diff + 备份，不盲目覆盖。

## 4. 当前状态（2026-06-21）

- **199**：E2（E1+2a-GT 联合）训练中，screen `e2_199`，GPU1-4，20000 步 ~1.5 天。
- **226.31**：1.4B teacher 从零重训中，screen `ms_bs16`，GPU1-4，100k 步 ~1.7 天（修复旧 teacher 欠拟合）。
- **本地**：主开发 / 评测。

> 维护约定：服务器占用、实验结论变化时，请同步更新 `doc/compute_resources.md` 与 `doc/result_exp.md`，并在此 §4 更新一句话状态。
