# paper/ —— AAAI 论文工作区（入口）

> 第一入口。读完本文即知论文的写作主线、文献定位、实验素材、参考文献在哪。
> 项目总入口见 `../AGENT.md`；实验单一事实来源 `../doc/result_exp.md`。建立 2026-06-22。

## 这是什么

VOSR 文本超分**一步蒸馏**论文（投 AAAI）的写作工作区：思路、文献、结果素材、参考 PDF 集中于此。
**一句话卖点**：NFE=1 蒸馏 + "蒸馏均衡器"发现（唯 student 侧细粒度 OCR 监督穿透）+ 双空间 OCR 监督，在 TAIR + Real-CE 上可读性 SOTA 且快 ~50×。

## 文件索引

| 文件 | 用途 | 何时读 |
|---|---|---|
| **`writing_strategy.md`** | ⭐ 写作主线：基调/intro 漏斗/三贡献/章节骨架/评测口径/**待决策** | 写任何章节前先读 |
| **`related_work_scan.md`** | 文献地图 + 3 篇参考论文结构借鉴 + novelty 边界 + Real-CE 口径雷 | 写 Related Work / 定对标前 |
| **`experiments_for_paper.md`** | headline 数字 + 结果→表/图映射 + 待补实验状态 | 排表格 / 写实验节 |
| **`story_evidence_audit.md`** | ⭐ 证据-风险审视：每根支柱铁证/风险 + 风险总表 + 可发表判断 | 动笔前 / 排实验前 |
| **`ToDoExp/`** | ⭐ **待执行实验任务单**（给执行 Agent）：每个 `NN_*.md` 一个可执行实验包 | 派活给执行 Agent 时 |
| `sources/` | 参考文献 PDF（不入 git）+ `README.md` 索引 | 查原文 |

## 当前阶段

**写作前的思路梳理**。已完成：项目/实验全盘梳理、评测口径三层定调、文献扫描+对标定位、3 篇参考结构借鉴归档。
**待用户拍板**（见 `writing_strategy.md` §6）：① 重心/标题 ② C2 强度 ③ 贡献编排。拍板后即可起草 abstract + intro 第一段。

## 硬约束（继承自 `../AGENT.md`）

- 实验级动机/配置/结果入 `../doc/result_exp.md`（单一事实来源），本工作区不重复细节，只做论文向裁剪 + 指针。
- Real-CE 对标**不可跨论文搬数**，一律自己同协议（valid_list 260 + 官方 CRNN ACC/NED）重测。
- `sources/*.pdf` 已加入根 `.gitignore`，不提交。
