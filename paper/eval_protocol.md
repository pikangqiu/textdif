# UTEP — 统一可复现文本超分评测协议（论文 C4）

> 建立 2026-06-27。本文是"统一对比标准"这条贡献的设计规范；参考实现 = `/data/ywk/eval`（原样 GitHub 权重 + 冻结评测管线）。命名 **UTEP**(Unified Text-SR Evaluation Protocol) 为工作名，定稿可改。
> 单一事实来源：跑出的数进 `../doc/result_exp.md`。

## 0. 动机（这个贡献为什么成立）
现状文本超分对比"模糊复杂、无统一标准"，5 个轴没被任何论文同时锁死：
1. **OCR 引擎不统一**：CRNN/TransOCR/TESTR/PaddleOCR/ABCNet v2 各跑各 → 数不可搬（铁证：TeReDiff published E2E None **49.39** vs 官方权重复现 **33.53**，已逐字节溯源、IQA 三方对拍、HQ 校准准，缺口锁死在"论文权重的 TESTR E2E"本身）。
2. **任务形态混报**：端到端 spotting / 给定区域识别 / 参照式相似度被当一回事，实测不同能力。
3. **参照物不统一**：人工 GT / OCR-of-HR 伪参照 / 标注，天花板差很多。
4. **测试集与区域定义不统一**。
5. **几乎无人报 NFE，且不把 det 与 E2E 绑报** → E2E 单看被"低召回高精度假象"误导（我们 E1/2c/dino 即如此）。

## 0.5 前人评测协议实测扫描（2026-06-27，UTEP 的实证地基）
> 直接读 `sources/txt/*` 抽取，不是凭空设计。结论：**领域裂成两阵营 + 连"OCR-A"都不统一**。

| 论文 | 任务形态 | OCR 引擎 | 识别指标 | 参照物 | 测试集 | ΔvsLR | NFE列 |
|---|---|---|---|---|---|---|---|
| **TAIR** | 全图 spotting(det+E2E) | TESTR + ABCNet v2 | det F1 / E2E None/Full | GT标注+lexicon | SA-Text, **Real-Text 847** | 否 | 否 |
| Real-CE | 行级识别 | CRNN(中文) | ACC(词准)/NED | GT转写 | Real-CE test | 否 | 否 |
| DiffTSR | 裁好的文本行 | TransOCR | ACC/NED | GT转写 | CTR-TSR | 否 | 否 |
| TADiSR | GT框裁区域 | PP-OCR | OCR-A=**Levenshtein ratio** | **OCR-of-HR(伪参照)** | FTSR-TE, Real-CE-val | 否 | 否 |
| **TIGER** | GT框裁区域 + **区域IQA** | PP-OCR | OCR-A(Levenshtein)+**ΔOCR-A vs LR** | **标注转写** | Real-CE(重标188), UZ-ST | **是** | 否 |
| **TEXTS-Diff** | GT框裁区域 | **PP-OCRv5** | OCR-A=**整行精确匹配** | **GT标注** | **Real-Texts(=847)**, Real-CE | 否 | 否(一步却没报) |

**三条关键事实**：
1. **没有任何一篇在同一批图同时跑 spotting + 区域识别** → UTEP 的真贡献 = **跨阵营桥接**(L1+L2 同图)。
2. **ΔOCR-A 是 TIGER 的，不是我们的**（见 §2，必须 credit）。
3. **"OCR-A"三家口径都不同**：Levenshtein(TADiSR/TIGER) vs 精确匹配(TEXTS-Diff)；OCR-of-HR(TADiSR) vs 标注(TIGER/TEXTS-Diff) → 数不可比(即 0.882 vs 64.7 之谜根源)，正是 UTEP 的 motivation。
4. **白送证据**：TIGER Table 3 已证 SeeSR/OSEDiff/SupIR/DiffBIR/DiffTSR 在 Real-CE 上 ΔOCR-A 全负；TEXTS-Diff 在 **Real-Texts(=847)** 上用 PP-OCRv5 报 OCR-A，与我们 L2 最可对齐。

## 1. 三层任务解耦（核心设计，全报，禁止只挑一层）
> 2026-06-27 用户定：纳入 **L1 + L2 + IQA**；不单列 L3 参照式 OCR-A（统一走 L2 区域识别，需 GT 转写）。

- **L1 端到端 spotting**：`det F1` + `E2E None` + `E2E Full`。问"下游能否在 SR 图里**找到并读对**字"。引擎 = **TESTR**（官方权重/config，与 TAIR Table 3 同口径，向后兼容）。**det 与 E2E 必须同行出**——禁止只报 E2E。Full lexicon 用测试集 GT 重建词表、离线重评分既有 None 识别串得到。
- **L2 给定区域识别**：喂 **GT 框**，剥离检测混淆，只问"字形清不清楚"。引擎 = **PP-OCRv5**（与 TEXTS-Diff/TADiSR/TIGER 同代引擎，便于和它们 as-reported 对齐）。**同时报硬+软两种**：`EM/整行精确匹配`(=TEXTS-Diff 口径) + `CharAcc/CER`(字符级) + **`OCR-A=Levenshtein ratio`**(=TADiSR/TIGER 口径)。参照物 = **人工/curated 标注转写**(随 TIGER/TEXTS-Diff，**不用** TADiSR 的 OCR-of-HR 伪参照)。这层让我们 847 的 CharAcc 68.74 有同口径可比对象，且 OCR-A 列可与 TEXTS-Diff 在 Real-Texts(=847)、TADiSR/TIGER 在 Real-CE 的 published 对齐。
- **IQA 伴随层**：全图 `PSNR/SSIM/LPIPS/DISTS/FID` + **区域级 `PSNRcr/SSIMcr/LPIPScr/DISTScr`**（裁 GT 文本区域算，借 TIGER Table 3，比全图更反映文字保真）+ 可加 NIQE/MUSIQ/MANIQA。诚实暴露 **OCR–IQA 取舍**，并复现 TAIR/TIGER 的 IQA 列。

## 2. ΔOCR（**借自 TIGER，非我们原创——必须 credit**）
- **出处更正(2026-06-27 扫描前人后)**：`ΔOCR-A vs LR` 是 **TIGER(2510.21590) Table 3 已有的指标**，不是我们的发明。TIGER 已报 Real-CE 上 SeeSR −27.4% / OSEDiff −38.0% / SupIR −37.0% / DiffBIR −27.2% / DiffTSR −20.7%（**全负**）。
- **定义**：`ΔOCR = SR 的 OCR-A − bicubic-LR 的 OCR-A`（同口径，L2）。
- **我们怎么用**：作为**已被领域接受**的指标纳入（引用 TIGER），并证明**我们一步法 ΔOCR-A 仍为正**。它**不再当"协议招牌创新点"**，否则会被 TIGER scoop。集无关、清分 text-aware vs 通用扩散的价值仍在，但功劳归 TIGER。
- 待测：核心 4 baseline 在我们冻结管线下复现"通用扩散 ΔOCR<0"，我们一步法 >0。

## 3. 强制伴随列与诚实/防刷规则
- 每行强制：**NFE/steps**（我们=1，几乎无人报）、**det 与 E2E 同报**、**IQA**。
- **只采信自评**：所有方法在冻结管线下重跑；published 数仅作**标注的 as-reported 旁注行，绝不混入排名**（TeReDiff 教训）。
- **None+Full 都报**（Full 可被 snap-to-vocab 刷高）。
- **可复现回执**：放出 SR 输出 + 区域 mask + 引擎权重 hash + 评测脚本。

## 4. 防"自肥"三招（直接回应审稿人最可能的攻击）
1. **包住现有协议**：L1 的 TESTR 就是 TAIR 的协议 → 我们是统一/超集，不是另起炉灶帮自己。
2. **先复现 TAIR Table 3**：核心 4 baseline 在我们管线下的数 ≈ 他们 published → 协议中立可信，再谈我们的数。
3. **ΔOCR 奖励全领域共同目标**，非私货。

## 5. 测试集与区域（canonical）
- **Real-Text 847**（TAIR 公开测试集，字节一致）——主战场。区域 mask = `HQmask/`。
- **Real-CE val**（跨域，中文/真实退化）——需 **GT 转写**才能跑 L2；Real-CE 自带行框，转写经 TAIR 双-VLM 管道补（依赖 `ToDoExp/03` Part 0）。零样本为主，微调行作对照。
- SA-Text-test（**含训练污染声明**，仅 TAIR 同口径参考，不入主结论）。

## 6. 待讨论/未定
- 协议正式命名（UTEP? 是否换名）。
- IQA 具体取哪几项进主表（PSNR/SSIM/LPIPS/FID 为主，DISTS/NIQE/MUSIQ/MANIQA 进附录）。
- FID 的参照分布如何定（GT 集 vs 外部）。
- ΔE2E 是否与 ΔCharAcc 并列主表，还是 ΔCharAcc 单首列。
