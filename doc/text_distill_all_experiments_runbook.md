# Text-SR 蒸馏消融全实验 Runbook（设置 / 训练 / 推理）

更新时间：2026-06-07

本文把 `text-distill-ablation` 分支上的**全部实验**汇总成一份可执行清单：每个实验
的设置、训练命令、推理命令。所有实验都共享同一个 teacher / 初始化权重、同一份数据、
同一套评测输入，单一变量法，便于横向比较。

> 评测指标（Word/Char Acc、NED、PSNR/SSIM/LPIPS）由外部脚本在输出目录上计算，
> 不在本仓库内；推理只负责把 SR 结果写到 `OUTPUT_ROOT/<tag>_step<STEP>/`。

---

## 0. 共同设置（所有实验一致）

```bash
# conda 环境（脚本会自动 activate；手动跑时也先切）
source /home/ywk/anaconda3/etc/profile.d/conda.sh && conda activate vosr
```

**Teacher / student 初始化权重（全实验相同）**

```text
exp_vosr_text/ldit_fm_bs008_sd2f8c4_size512_ps2_d1024_b28_h16_cfgs0.5-r0.1-wc0.05-0.25_edr3_tduni_typetxt_text_hr/checkpoints/checkpoint-00040000/clean_weights/ema_model.safetensors
```

| 项目 | 值 |
|---|---|
| backbone | 0.5B DiT（dim 1024 / depth 28 / heads 16 / patch 2） |
| VAE | SD2.1 (`ae_type: sd2`, 4ch, f8) |
| 视觉条件 | DINOv2-b 第 8 层（vision-only，无文字条件） |
| 分辨率 / 上采样 | 512 / ×4 |
| 训练步数 | 20000，每 1000 存一次 |
| 训练根目录 | `exp_vosr_text_distill_ablation/` |
| 评测输入 | `/data/ywk/datasets/real_test/LQ`（GT 在 `.../HR`） |

**推荐环境变量**

```bash
export INPUT_DIR=/data/ywk/datasets/real_test/LQ
export OUTPUT_ROOT=preset/results/text_repa_ablation   # 各实验组可换
export STEP=00020000          # 评测用的 checkpoint 步数
export NPROC_PER_NODE=4       # 训练 GPU 数
export GPUS=0                 # 推理 GPU（可 0,1,2 并行多实验）
```

**"B baseline" 的精确含义（写论文务必区分）**

| 名称 | cfg_scale | distill_type | u_weight | 含义 |
|---|---|---|---|---|
| **B（distill / no_rc）** | 0.5 | shortcut | **0.0** | 纯 guided-target 速度匹配，shortcut 自一致项**未生效** |
| **shortcut** | 0.5 | shortcut | **1.0** | 真正打开 shortcut 一致性 |
| **C（rc）** | 0.5 | rcgm | 1.0 | guided-target + RCGM 递归一致 |
| **A（full）** | 1.0 | shortcut | 0.0 | 全条件 target（cfg=1.0） |

所有 REPA / FD / OCR 加项实验都建立在 **B（u_weight=0）** 之上，加项是唯一变量。

---

## 1. 蒸馏目标基线组（A / B / C + shortcut）

研究"蒸馏方式"本身对 OCR 的影响。脚本：`scripts/train_text_distill_ablation.sh`。

| tag | 实验 | 关键设置 | config |
|---|---|---|---|
| A | full target | cfg_scale 1.0 | `VOSR_0.5B_text_full_target_no_rc.yml` |
| B | guided target（OCR 冠军基线） | cfg_scale 0.5, u_weight 0 | `VOSR_0.5B_text_guided_target_no_rc.yml` |
| C | guided + RCGM | distill_type rcgm, u_weight 1 | `VOSR_0.5B_text_guided_target_rc.yml` |
| — | shortcut consistency | u_weight 1.0 | `VOSR_0.5B_text_guided_target_shortcut.yml` |

**训练**

```bash
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh full_target_no_rc
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_no_rc
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_rc
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_shortcut
# 或一次跑 A/B/C/(rc)：
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh all
```

**推理**（A/B/C 三连，多 GPU 并行）

```bash
GPUS=0,1,2 STEP=00020000 bash scripts/infer_text_distill_ablation_multigpu.sh
```

---

## 2. REPA 表征对齐组（在 B 上叠加）

研究"对齐到什么表征"对 OCR 的影响。
脚本：`scripts/train_text_repa_ablation.sh` / `scripts/infer_text_repa_ablation_multigpu.sh`。
总损失：`L = L_guided_distill + λ_repa · L_repa`（第一版用 global pooled token cosine）。

| tag | 对齐目标 | 粒度 | config 后缀 |
|---|---|---|---|
| dino | DINOv2 | global pool | `_dino_repa` |
| dino_token | DINOv2 | per-token | `_dino_token_repa` |
| ocr | OCR(TrOCR) 特征 | **global pool** | `_ocr_repa` |
| seg | 分割特征 | global pool | `_seg_repa` |
| seg_token | 分割特征 | per-token | `_seg_token_repa` |

**训练**

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh dino_token
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh seg
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh seg_token
```

**推理**

```bash
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh dino
# ... dino_token / ocr / seg / seg_token
GPUS=0,1 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh all
```

---

## 3. 局部 OCR 监督组（本轮重点：2a / 2b）

把"全局表征对齐"推进到"局部文字监督"。检测器/识别器用仓库内 vendored
PaddleOCR2Pytorch（`ch_ptocr_server_v2.0_{det,rec}`，CRNN，全程冻结，不联网）。
文字框**在线**检测于当前 HR crop（训练有 RandomCrop + 在线退化，离线缓存会错位）。
原理细节见 `doc/text_ocr_local_repa_experiment.md`。

### 2a — local-crop OCR-REPA（区域序列特征对齐）

```text
HR --det(no_grad)--> bbox；pred/HR 裁同框 -> resize -> 冻结 rec backbone+neck -> 序列特征
L_ocr = 1 - cosine(feat(pred_crops), feat(HR_crops).detach())
```

config：`VOSR_0.5B_text_guided_target_no_rc_ocr_local_repa.yml`
（B 基线 + `ocr_repa_weight: 0.5`, `ocr_repa_loss_type: cosine`, `ocr_repa_interval: 1`；不设 `ocr_ctc_weight`）

### 2b — HR 伪标签 + CTC 识别 loss（直接监督可读性）

```text
HR crop -> 冻结 rec 贪心 CTC 解码 -> 伪标签序列（确定性，不耗 RNG）
pred crop -> rec logits -> CTCLoss(伪标签) = L_ctc
```

config：`VOSR_0.5B_text_guided_target_no_rc_ocr_ctc.yml`
（B 基线 + `ocr_repa_weight: 0.0`（关 2a）+ `ocr_ctc_weight: 0.5`, `ocr_ctc_interval: 1`）

> **2a/2b 隔离保证**：2b 纯加法、默认关，独立键 `ocr_ctc_*`；`ocr_active` 带
> `ocr_repa_weight>0` 守卫，CTC-only 不会触发 2a。
> `tests/test_ocr_repa.py::test_2b_does_not_perturb_2a` 钉死同输入下 2a loss 不变。
> 两者也可在同一 yaml 同时 >0 做组合（检测各跑一次，开销翻倍）。

**训练**

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_local   # 2a
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_ctc     # 2b
```

**推理**（已补进 repa 推理脚本）

```bash
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh ocr_local
GPUS=0 STEP=00020000 bash scripts/infer_text_repa_ablation_multigpu.sh ocr_ctc
```

**单元测试**（CPU，不抢训练 GPU）

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_ocr_repa.py -q
```

---

## 4. FD-Loss 组（分布对齐，在 B / shortcut 上叠加）

`L += fd_loss_weight · L_FD`（冻结 judge 的 Fréchet 距离分布对齐）。
脚本：`scripts/infer_text_fd_ablation_multigpu.sh`。

| tag | base | 关键设置 | config |
|---|---|---|---|
| no_rc_fd | B（u_weight 0） | fd_loss_weight 0.01 | `VOSR_0.5B_text_guided_target_no_rc_fd.yml` |
| shortcut_fd | shortcut（u_weight 1） | fd_loss_weight 0.01 | `VOSR_0.5B_text_guided_target_shortcut_fd.yml` |

**训练**

```bash
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_no_rc_fd
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_shortcut_fd
```

**推理**

```bash
GPUS=0 STEP=00020000 bash scripts/infer_text_fd_ablation_multigpu.sh no_rc_fd
GPUS=0 STEP=00020000 bash scripts/infer_text_fd_ablation_multigpu.sh shortcut_fd
GPUS=0,1 STEP=00020000 bash scripts/infer_text_fd_ablation_multigpu.sh all
```

---

## 5. 多步 teacher 基准（对照上界 / 速度对照）

```bash
INFER_STEPS=25 bash scripts/test_text_0.5b_step40000.sh
```

用于回答："一步蒸馏（B/C/加项）能否逼近甚至超过 25 步 teacher 的 OCR"。

---

## 6. 通用直推命令（任意 checkpoint）

包装脚本本质都是这一条；评测非默认步数 / 中间 checkpoint 时直接用：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_vosr_onestep.py \
  -c exp_vosr_text_distill_ablation/<EXP_DIR>/checkpoints/checkpoint-<STEP> \
  -i /data/ywk/datasets/real_test/LQ \
  -o preset/results/<tag>_step<STEP> \
  -u 4 --infer_steps 1 --align_method nofix --force_rerun
```

`<EXP_DIR>` 由训练参数自动拼成，例如 2b：

```text
ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_ctc
```

---

## 7. 评测协议建议（消除"最后一权重"噪声）

1. **全方法统一**用固定步数 last checkpoint（默认 `STEP=00020000`）；**绝不在测试集上挑 checkpoint**。
2. 关键方法补测**最后 3–5 个 checkpoint**（如 16k/18k/20k）报 **mean±std**，把
   "小数点差异"用噪声水平卡死：差距 < std 视为平手。
   ```bash
   for S in 00016000 00018000 00020000; do
     STEP=$S bash scripts/infer_text_repa_ablation_multigpu.sh ocr_ctc
   done
   ```
3. 主机制主张（"一步蒸馏提升 OCR"）建议至少换一个 init/backbone 复现一次，否则列为 limitation。

---

## 实验 → config 速查表

| 组 | tag | config 文件（`configs/train_yml/one_step/text_distill_ablation/`） |
|---|---|---|
| 基线 | A | `VOSR_0.5B_text_full_target_no_rc.yml` |
| 基线 | B | `VOSR_0.5B_text_guided_target_no_rc.yml` |
| 基线 | C | `VOSR_0.5B_text_guided_target_rc.yml` |
| 基线 | shortcut | `VOSR_0.5B_text_guided_target_shortcut.yml` |
| REPA | dino | `VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml` |
| REPA | dino_token | `VOSR_0.5B_text_guided_target_no_rc_dino_token_repa.yml` |
| REPA | ocr | `VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml` |
| REPA | seg | `VOSR_0.5B_text_guided_target_no_rc_seg_repa.yml` |
| REPA | seg_token | `VOSR_0.5B_text_guided_target_no_rc_seg_token_repa.yml` |
| OCR-局部 | ocr_local (2a) | `VOSR_0.5B_text_guided_target_no_rc_ocr_local_repa.yml` |
| OCR-局部 | ocr_ctc (2b) | `VOSR_0.5B_text_guided_target_no_rc_ocr_ctc.yml` |
| FD | no_rc_fd | `VOSR_0.5B_text_guided_target_no_rc_fd.yml` |
| FD | shortcut_fd | `VOSR_0.5B_text_guided_target_shortcut_fd.yml` |
