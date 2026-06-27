# 任务 04 —— TAIR 主表对齐 + TeReDiff 缺口处理

> 目标：主表与 TAIR Table 3 同口径可引；TeReDiff 宣称数缺口诚实闭环。详 `../story_evidence_audit.md` TeReDiff/对齐。

## Part A —— TAIR 主表对齐（多为引用，少量可选跑）
- TAIR Table 3 基线：**Real-ESRGAN / SwinIR / ResShift / StableSR / DiffBIR / SeeSR / SUPIR / FaithDiff / TeReDiff**；两 spotter(**ABCNet v2 + TESTR**)各 Det(P/R/F1)+E2E(None/Full)；Table 4 图像质量(PSNR/SSIM/LPIPS/DISTS/FID/NIQE/MANIQA/MUSIQ/CLIPIQA)。
- 我们：**直接引 TAIR Table 3 的 TESTR 列** + 我们的行(已验证:HQ 校准 + TeReDiff 复现)。**0 必需跑**。
- **对齐缺口(可选)**：TAIR 报两个 spotter，我们只有 TESTR。若 reviewer 要 ABCNet v2 → 在**已有 SR 输出**上跑 ABCNet v2 spotting(纯 eval,无需重推理)。先不做，列为 rebuttal 储备。

## Part B —— TeReDiff 缺口（行动项,非实验）
- 现状：官方 stage3 权重复现 None 33.53/Full 44.62 vs 论文宣称 49.39/56.45；det 反高(76.14>74.88)、HQ 校准准 → 缺口纯识别端;已排除 prompt_style(TAG 更差)。
- **行动**：
  1. **邮件联系 TeReDiff 作者**要"论文同版权重/复现说明"(缺口若因权重版本，拿到即可堵)。
  2. 正文写法：对标"可复现 TeReDiff + 全部其余 published(FaithDiff/DiffBIR/SeeSR/SUPIR…)"我们 NFE=1 胜；对 TeReDiff 宣称数取保留(标注"published, 官方权重未能复现")。**诚实立场写到滴水不漏。**
  3. 储备：若拿到论文权重复现出 ~49 → 调整对标措辞;若仍复现不出 → 保留现写法 + 把复现细节放附录。

## 工作量
- Part A 引用=0;ABCNet 可选 eval ~半天(储备)。
- Part B 邮件 + 写作,~0 算力。

## 结果（回填）
> 作者回复: ______ 主表对齐确认: ______ ABCNet(如做): ______
