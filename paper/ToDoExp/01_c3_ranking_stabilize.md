# 任务 01 —— 稳住 C3：双空间监督排序（堵自伤噪声）

> 风险：冠军精排序落在自定噪声带内——2a+2d(68.74)−2d(68.28)=+0.46、2d−2a峰(67.66)=+0.62，均 <1 分(§0 规则"≥1分才算信号")；且 2d/2a+2d 单 ckpt 无 std。详 `../story_evidence_audit.md` C3。
> 目标：让 C3 不靠不可靠的 CharAcc 精排序承重。

## 做什么（两路并行，都要）

### (a) 补噪声带（反悔此前 #1，因审视证明它对 C3 是必需）
- 对 **2d** 与 **2a+2d** 各补评 ckpt **18000 / 19000**（已有 20000），凑 3 点出 mean±std，与 2a 噪声带(66.92–67.66)同口径。
- 协议①：用 199 OSEDiff `metric_*.py` 模板(sed 改 SR_PATH/ckpt)；协议②：本机 TESTR `eval_spotting.py`。
- 产物：2d、2a+2d 的 3-ckpt CharAcc/EM/det/E2E mean±std，写回 `../../doc/result_exp.md` 对应节。

### (b) 重写 C3 措辞（无论 (a) 结果如何都做）
- 不写"2a+2d > 2d > 2a 谁最强"；写：**特征空间(OCR-REPA) 与输出空间(GT-text CTC) 监督各自相对 baseline 显著(>1分)、且叠加无冲突(det F1 双新高 78.32, +1.29 over 2d)**。
- 精排序锚到**更稳的 det/E2E**，CharAcc 只说"OCR-aware 全部 >> baseline 与非 OCR 对照"。

## 判据
- 若 (a) 显示 2a+2d/2d 排序在 std 外稳定 → C3 可保留弱排序表述；
- 若仍在噪声内 → 严格按 (b) 写，headline 不主张精排序。

## 工作量
- 纯评测：每 ckpt 协议① ~20min + 协议② ~30min，共 4 个新 ckpt(2d/2a+2d × 18k/19k)。**~半天**。推理图若缺需先补推理(NFE=1,u4)。

## 结果（回填）
> 2d 3-ckpt: ______  2a+2d 3-ckpt: ______  排序是否稳: ______
