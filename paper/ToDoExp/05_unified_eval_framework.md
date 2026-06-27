# 05 · 统一评测框架 /data/ywk/eval + UTEP 协议落地 + 机理探针

> 建立 2026-06-27。落地 C4(`../eval_protocol.md`) 与 C2 机理。决策已定（用户 2026-06-27）：
> - baseline 第一批 = **核心 4 个**：SeeSR / OSEDiff / DiffBIR / FaithDiff。
> - 协议层 = **L1 TESTR spotting + L2 PP-OCR 区域识别(EM+CharAcc+OCR-A Levenshtein) + IQA(全图+区域级 PSNRcr)**。
> - **C4 降级定位(2026-06-27 扫描前人后)**：不卖"发明新协议/新指标"——ΔOCR-A 属 TIGER、OCR-A/区域IQA/PP-OCR 皆前人现成；C4 = **基准/测量贡献**，重心放 **①跨阵营桥接(L1+L2 同图) ②可复现审计(TeReDiff 不可复现溯源) ③NFE+det/E2E 同报**；ΔOCR-A/OCR-A 借用并 credit。详 `../eval_protocol.md` §0.5/§2。
> - **机理探针与 eval 并行做**。
> 铁律：原样 GitHub 权重（一字不改），只写 `/data/ywk/` 下；先记录后执行；结果进 `../../doc/result_exp.md`。

## Part A — 建 `/data/ywk/eval`（参考实现）
目录约定（每个 baseline 一子目录，代码原原本本来自 GitHub）：
```
/data/ywk/eval/
  repos/{seesr,osediff,diffbir,faithdiff}/   # 各自 clone + 各自 env（本机可建新 env）
  weights/                                     # 官方权重（留 md5）
  inputs/{realtext847_LR, realce_val_LR}/      # 统一输入
  outputs/<method>/<dataset>/                  # 各方法 SR 输出（PNG）
  protocol/                                    # UTEP 评测脚本（冻结）
    l1_testr.py  l2_ppocr.py  iqa.py  delta_ocr.py  aggregate.py
  results/                                      # 汇总表 csv/md
```
1. clone 4 个仓库 + 下官方权重（本机有外网），逐一 smoke：在 5 张 847 LR 上出 SR。
2. **统一输入**：所有方法吃同一批 Real-Text 847 LR（与我们 ×4 到 512 同口径）。
3. 各方法 env 隔离（绝不改现有 env）；长任务 `screen`。

## Part B — UTEP 协议脚本（冻结后不再改）
- `l1_testr.py`：复用本机已验证的 TESTR 管线（config 1600/1824/0.45、官方权重），出 `det F1 / E2E None / Full`。Full = 用 GT 重建词表离线重评分既有识别串。
- `l2_ppocr.py`：PP-OCRv5 喂 GT 框，出 `CharAcc / EM / CER`（复用 199 OSEDiff 那套 metric 思路，但冻结一份在 eval/ 内自包含）。
- `iqa.py`：PSNR/SSIM/LPIPS/FID(+DISTS/NIQE/MUSIQ/MANIQA 进附录)。
- `delta_ocr.py`：先对 **bicubic-LR 输入**跑 L2(+L1)，再 `ΔOCR = SR − LR`。
- `aggregate.py`：出主表（NFE | det F1 | E2E None/Full | CharAcc | **ΔOCR** | PSNR/SSIM/LPIPS/FID）。

## Part C — 可信度背书：复现 TAIR Table 3
- 4 baseline 在我们管线下的 `det F1 / E2E None` 与 TAIR Table 3 published 对拍：
  - 参照（TAIR Table 3，TESTR）：DiffBIR 68.35/39.27、SeeSR 67.87/40.34、FaithDiff 70.57/41.64、SUPIR 48.39/27.25。
- **验收**：我们 re-run ≈ published（容差内）→ 管线中立可信，UTEP 立得住。对不上则先排查（输入口径/spotter config/权重版本），排查记录进台账。
- OSEDiff 不在 TAIR Table 3，但开源、是一步扩散对照，按同协议补一行。

## Part D — 出三张论文表
1. **主表(847, UTEP)**：4 baseline + 我们(B/2a/2d/2a+2d) + TeReDiff(复现+as-reported旁注) + LQ/HQ 锚。首列 ΔOCR。
2. **Real-CE 跨域表**：同 4 baseline + 我们零样本/微调，**重点看 SeeSR/OSEDiff 是否负 ΔOCR、我们是否正**。依赖 `03` Part 0（Real-CE GT 转写）出 L2；若转写未就绪先只出 IQA + L1。
3. **效率列**：NFE/steps（我们=1 vs 多步），并入主表。

## Part E（并行）— C2 机理探针（支撑"蒸馏均衡器"）
目标：把 C2 从"一个 loss"抬成机理性发现。两个探针：
1. **均衡器位移探针**：对 teacher 侧/input 侧干预（结构增强 teacher、E1 cond）测**学生单步输出的特征/像素位移**应≈0，对照 student 侧监督（2a/2d）位移显著 → 证"只有 student 侧能改输出"。
   - 实现：同一批 LQ，分别用「baseline 学生」「teacher侧增强重蒸学生」「2a/2d 学生」出 SR，测两两输出 LPIPS/特征距离；teacher侧应≈baseline、student侧应明显偏移。
2. **增益定位探针**：2d 相对 baseline 的 CharAcc 增益是否**集中在 teacher OCR 原本就错的区域** → 证 student 侧 CTC 注入了 teacher 缺的信息。
   - 实现：按区域分桶（teacher OCR 对/错），比较 2d−baseline 的 ΔCharAcc 在两桶的分布。
- 产出：1–2 张图 + 一段机理论证，写进 C2。

## DoD（完成定义）
- [ ] eval/ 4 baseline 各出 847 SR，md5 留痕；env 隔离不污染。
- [ ] UTEP 5 个脚本冻结、自包含、可一键复跑。
- [ ] 复现 TAIR Table 3：4 baseline det/E2E ≈ published（差异有记录解释）。
- [ ] 主表 + Real-CE 表 + 效率列出齐，ΔOCR 为首列；published 仅 as-reported 旁注。
- [ ] 机理探针 2 个出图 + 结论，写入 C2。
- [ ] 全程记录进 `../../doc/result_exp.md` 与根 `AGENT.md` §5。
