# ToDoExp —— 待执行实验计划（给执行 Agent）

> 本目录是**给执行 Agent 的任务单**。每个 `NN_*.md` 是一个可独立执行的实验包：目标 / 协议 / 逐步命令 / 交付物 / 记录要求。
> 上层背景见 `../AGENT.md`、`../writing_strategy.md`、`../related_work_scan.md`、`../experiments_for_paper.md`。
> 建立 2026-06-22。

## 执行 Agent 必读铁律（继承根 `../../AGENT.md` §3）

1. **绝不杀别人进程**；缺显存就等（启动器带重试循环）。长任务用 `screen`，不用 `nohup`。
2. **只写各机指定目录**：本地 `/data/ywk/`、199 `/data2/wyw/ywk/`、226.31 `/data07/dt_data/ywk/`。
3. **不改远端 conda 环境**；远端无外网（HF 用 `HF_HUB_OFFLINE=1`）。基线各建**独立 conda env**，互不污染。
4. **先记录后执行**：每次推理/评测在 `../../AGENT.md` §5 操作记录表追加一行；实验级结果入 `../../doc/result_exp.md`（单一事实来源）+ 回填本目录对应任务的"结果"段。
5. **远程操作助手**：199 `/data/ywk/claude_tools/{rexec,rput,rget}.py`；226.31 `{rexec_dt,rput_dt,rget_dt}.py`。

## 任务清单（2026-06-27 重排：05 升 P0 中心 + C4 降级定位后）

| 文件 | 任务 | 解决什么 | 依赖 | 优先级 |
|---|---|---|---|---|
| `05_unified_eval_framework.md` | 建 `/data/ywk/eval`、核心4 baseline(SeeSR/OSEDiff/DiffBIR/FaithDiff)同管线、复现 TAIR Table3、出公平主表 + ΔOCR；**Part E 机理探针** | Q1 公平对比 + Q2 Real-CE 翻盘 + 强化 C2(基准贡献 C4) | Real-CE 行依赖 03 | **P0（中心）** |
| `01_c3_ranking_stabilize.md` | 稳住 C3：补 2d/2a+2d 噪声带 + 重写为"无冲突叠加"、锚 det | 🔴 C3 精排序踩自定噪声线 | 无 | P0 |
| `03_realce_finetune_and_consistency.md` | **Part 0 重点**：TAIR 双-VLM 管道给 Real-CE 补 GT 转写 → 喂 05 跨域表 L2（微调救观感已证 null） | Real-CE 跨域表识别行 | 本机管道已 clone | P1 |
| `02_c1_nfe_control_and_scaling.md` | NFE 漂移对照坐实 C1 + 1.4B scaling 决策 | 🟡 C1 机制未隔离/1.4B 坏 | 无 | P1 |
| `04_tair_alignment_and_terediff.md` | **大部分已被 05 Part C 吸收**；仅留可选 ABCNet v2 / 联系作者残项 | 🟡 ABCNet 缺口 | 与 05 重叠 | P2 |

> 推荐顺序：05 Part A–C(847 公平主表,立即可做) → 05 Part E 机理探针(并行,吃存量 ckpt) → 03(Real-CE 转写) → 05 Part D 跨域表 → 01(可穿插) → 02。
> C4 定位：**基准/测量贡献,非"发明新协议"**（ΔOCR-A 属 TIGER，详 `../eval_protocol.md` §0.5/§2）。
> 旧计划 `01_realce_baseline_reeval.md`(跑全部 9 基线)已废弃删除——改为"验证-引用"轻量策略 + 风险驱动整改。
> 背景：先读 `../story_evidence_audit.md`(为何做这些) → 再读 `../writing_strategy.md`(story 怎么用这些)。

## 完成定义（DoD）

一个任务完成 = ① 所有命令跑通无错；② 结果数字回填该任务 md 的"结果"段 + 写入 `../../doc/result_exp.md`；③ `../../AGENT.md` §5 留操作流水；④ 产物路径明确可追溯。
