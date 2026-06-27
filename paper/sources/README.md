# paper/sources —— 文献 PDF 索引

> 本目录存放 AAAI 文本超分论文相关的参考文献 PDF（arXiv 公开版）。
> **PDF 不入 git**（见根 `.gitignore` 的 `paper/sources/*.pdf`）；本 README 入 git 作为索引。
> 下载日期：2026-06-22。逐条 WHY/HOW/WHAT + 结构分析见 `../related_work_scan.md`。

## 分类

### 我们的基准（必对标）
| 文件 | 论文 | 用途 |
|---|---|---|
| `TAIR_2506.09993.pdf` | Text-Aware Image Restoration with Diffusion Models (ICCV25, TeReDiff/SA-Text) | **codebase 基准**；协议②对标对象 |
| `RealCE_2308.03262.pdf` | A Benchmark for Chinese-English Scene Text Image SR (ICCV23) | **第二基准**；协议③(CRNN ACC/NED) 来源 |

### 用户指定：不开源、不对比，仅借鉴结构与对比组织
| 文件 | 论文 | 借鉴点 |
|---|---|---|
| `TADiSR_2506.04641.pdf` | Text-Aware Real-World Image SR via Diffusion + Joint Seg Decoders (2025-06) | 单步级 + Real-CE 强结果；主表/消融结构；**Real-CE 协议(189/PP-OCR/Levenshtein)** |
| `TIGER_2510.21590.pdf` | Restore Text First, Enhance Image Later (两阶段, 2025-10) | **区域裁剪指标 PSNR_cr/SSIM_cr/LPIPS_cr/DISTS_cr**；13 基线大表；OCR 依赖消融 |
| `DualTSR_2603.14207.pdf` | Unified Dual-Diffusion Transformer for STISR (2026-03) | **CFM + 离散扩散**(与 VOSR 同源 flow-matching)；ACC/NED 口径；消融增量结构 |

### 方法家族对照（通用一步蒸馏 SR；可对标/可引）
| 文件 | 论文 | 关系 |
|---|---|---|
| `OSEDiff_2406.08177.pdf` | One-Step Effective Diffusion Network (NeurIPS24) | 我们评测环境同名；一步蒸馏母方法之一 |
| `TSD-SR_2411.18263.pdf` | One-Step Diffusion w/ Target Score Distillation (CVPR25) | 一步蒸馏通用 SR；**均非文字向**=我们的白地 |

### 对比对象 / 对比设计参照（开源，可同协议重测）
| 文件 | 论文 | 关系 |
|---|---|---|
| `TEXTS-Diff_2601.17340.pdf` | TEXTS-Diff: TEXTS-Aware Diffusion for Real-World Text SR (ICASSP26) | **已是一步文本SR(OSEDiff式)**→冲击C1;识别口径=PP-OCRv5整行匹配(同我们协议①);投稿时未开源故不对比,Related Work点名 |
| `DiffTSR_2312.08886.pdf` | Diffusion-based Blind Text Image SR (CVPR24) | **开源(权重+推理+训练)**;但**行级STISR形态**(≤24字/128×512),与我们整图形态不同;Real-CE用TransOCR/1531裁剪对 |

### 直接竞品 / 方法根基
| 文件 | 论文 | 关系 |
|---|---|---|
| `TextSR_2505.23119.pdf` | Diffusion SR w/ Multilingual OCR Guidance (2025-05) | **与 E1 同源**(输入侧 OCR 条件注入, 多步)；反衬对照 |
| `REPA_2410.06940.pdf` | Representation Alignment for Generation (ICLR25 Oral) | **2a 的母方法**；2a=OCR 域特化 REPA，划界必引 |
