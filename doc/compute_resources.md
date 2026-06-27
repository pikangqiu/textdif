# 计算资源说明（compute_resources.md）

> 本文是本项目**全部计算资源**（服务器、GPU、数据与代码位置、登录方式、当前占用）的单一事实来源，供任意 AI / 协作者同步上下文。
> **所有服务器均为多人共享**，下方「铁律」是硬约束，任何自动化操作前必须遵守。
> 最后更新：2026-06-24（加 §9 评测/推理命令+环境+代码检查；刷新 §5 占用；清理 doc 旧文档）。改动资源或迁移代码后请同步更新本文。

---

## 0. 通用铁律（所有机器、所有 AI 必须遵守）

1. **绝不杀任何别人的进程。** 三台机都与他人共用 GPU/磁盘。只能 `kill` / `screen -X quit` **自己启动**的进程；需要显存时**等**，不抢、不杀。
2. **数据/产物只写各机指定目录**（见下表），其他盘视为满/只读。所有本地文件操作集中在 `/data/ywk/` 下，便于日后清理。
3. **不修改远端 conda 环境**（不 pip install / conda install / 改依赖）。环境是别人也在用的。
4. **长任务一律用 `screen`，不用 `nohup`**；启动器自带 6 次重试循环。
5. **远端无外网**（199 / 226.31）：HuggingFace 需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 走本地缓存；`torch.hub`（dinov2）需用本地缓存仓 `source='local'`，勿依赖 github 拉取。
6. 频繁重启后清理残留 rank：`pkill -9 -f <train脚本>` 并 bump `--master_port`，否则 rank0 卡在占用端口。

---

## 1. 三台服务器总览

| # | 机器 | 登录 | GPU | 代码仓 | 数据/产物目录 | 远程助手 |
|---|---|---|---|---|---|---|
| **本地** | 本机（工作目录 `/data/ywk/VOSR`） | 直接 | 4× ~24GB（共享 ljc/dmj/hty） | `/data/ywk/VOSR`（分支 `text-distill-ablation`，**主开发仓**） | `/data/ywk/`（所有文件操作集中此处） | — |
| **199** | `172.31.233.199` (node199) | `yawen` / `<见 doc/SERVERS.private.md>`<br>`source /data/anaconda/bin/activate && conda activate vosr` | 5× **A800 80GB**，跨 2 NUMA | `/data2/wyw/ywk/VOSR_textsr`（文本 SR 栈；同级 `/data2/wyw/ywk/VOSR` 是**不可动**的旧仓） | `/data2/wyw/ywk/`（数据 `datastes/`，产物 `VOSR_textsr/`） | `/data/ywk/claude_tools/{rexec,rput,rget}.py` |
| **226.31** | `172.31.226.31` (inspure) | `dt` / `<见 doc/SERVERS.private.md>`<br>`/data07/dt_data/ywk/conda_envs/vosr/bin/python` | 5× **RTX A6000 49GB** | `/data07/dt_data/ywk/VOSR`（base `main`，**无 OCR-REPA 栈**） | `/data07/dt_data/ywk/`（**其他盘全满**，只能用这里） | `/data/ywk/claude_tools/{rexec_dt,rput_dt,rget_dt}.py`（`python rexec_dt.py "<cmd>" 0 [timeout]`，第三参可调超时） |

---

## 2. 本地机（主开发仓）

- **路径**：`/data/ywk/VOSR`，git 分支 `text-distill-ablation`。这是**论文实验的主代码仓**与文档仓（`doc/`）。
- **GPU**：4× ~24GB，与 ljc/dmj/hty 共享，经常被占满。E2 之类 4 卡作业常需排队等卡。
- **角色**：写代码、组织实验、维护 `doc/result_exp.md`、向 199/226.31 下发任务。
- **dinov2 离线缓存**：`preset/ckpts/torch_cache/`（`facebookresearch_dinov2_main` 仓 + `checkpoints/dinov2_vit{b,l}14_pretrain.pth`），由 `train_vosr_distill.py:35-37` 的 `torch.hub.set_dir('preset/ckpts/torch_cache')` 指定。

## 3. 服务器 199（172.31.233.199，主力训练机）

- **GPU 拓扑**：5×80GB 跨 2 NUMA。NUMA0={GPU0,GPU1}；NUMA1={GPU2,GPU3,GPU4}，**GPU2↔GPU4=NVLink(NV8)**。GPU0 常被他人占。跨 NUMA（GPU1↔{2,3,4}）走 SYS（最慢）→ **任何含 GPU1 的多卡组合都引入一条慢 all-reduce**，固定有效 batch 时加卡≠线性加速。详见 `doc/node199_拓扑与配置原理.md`。
- **代码仓**：`/data2/wyw/ywk/VOSR_textsr` —— 文本 SR 一步蒸馏栈，含 `models/ocr_repa.py`、`train_vosr_distill.py`、`text_distill_ablation/` 配置、`PaddleOCR2Pytorch/`（OCR-REPA recognizer，GT-box 模式无需 detector）、`preset/ckpts/torch_cache`（dinov2 + pyiqa 离线缓存）。**注意**：同级 `/data2/wyw/ywk/VOSR` 是不可动旧仓。
- **数据**：`/data2/wyw/ywk/datastes/`
  - 训练：`SA/images`（119,495 张 `sa_*_crop_*.jpg`）+ `SA/train.txt`；GT-box 索引 `VOSR_textsr/text_boxes_index.pkl`（10MB，按 basename）。配置里 `train_dataset_config: configs/train_txt/sa_remote_dataset.txt`。
  - 0.5B 文本 teacher：`VOSR_textsr/preset/checkpoint-00040000/clean_weights/ema_model.safetensors`。
  - 测试/评测：Real-Text 847 在 `/data2/wyw/ywk/datastes/real_test/{HQ,LQ}`；协议① PP-OCR+pyiqa 评测脚本在 `/data2/wyw/ywk/OSEDiff/metric_*.py`（osediff 环境）。
  - **Real-CE**（第二 benchmark，含 train）：`/data2/wyw/ywk/datastes/Real-CE/{train,val}/{13mm,26mm,52mm,det_annos,trans_annos_52mm,...}`。微调用 52mm HR（645 张，~2572×2492）；GT 框由 `trans_annos_52mm`（四点多边形+文本，52mm 原生坐标）经 `scripts/build_realce_boxes_index.py` 转成 `preset/realce_text_boxes_index.pkl`（608 图/9033 框，schema=`{stem:{boxes:[[x1,y1,x2,y2]],texts:[...]}}`，与 SA 的 `text_boxes_index.pkl` 同格式）。
- **conda 环境位置**：`/home/yawen/.conda/envs/vosr/bin/python`（python 3.10）；激活 `source /data/anaconda/bin/activate && conda activate vosr`。**缺 `shapely`** → PaddleOCR 在线 DBNet 检测会 `No module named 'shapely'` 崩 → **199 上只能跑 GT-box OCR-REPA（用框索引），跑不了在线检测 2a**；按"不改环境"铁律，需要框时一律走 GT-box。
- **启动器范式**（`run_gt_*.sh` / `run_e2_199.sh`）：`source /data/anaconda/bin/activate; conda activate vosr` + `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_COMPILE_DISABLE=1 NCCL_P2P_LEVEL=NVL HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + `torchrun --nproc_per_node=N --master_port=PORT train_vosr_distill.py --config ...`，6 次重试。
- **Resume 陷阱**：保持 `resume_ckpt: ~`（自动发现原目录最新 ckpt 续训）；设成显式路径会给 exp 目录加 `_resume` 后缀导致 ckpt 分裂。扩 world size 要复制 `random_states_<rank>.pkl`。
- **验证关闭**：config `inference_steps: 1` = 永不验证（代码 `global_step % inference_steps == 1` 的哨兵；勿设成 >1，否则 step1 触发验证撞 `lpips_avg` 未赋值 bug）。

## 4. 服务器 226.31（172.31.226.31，1.4B 大模型机）

- **GPU**：5× RTX A6000 49GB。GPU0 常被 `xsh` 占。
- **代码仓**：`/data07/dt_data/ywk/VOSR`（base `main` 分支，**未含 OCR-REPA / OCR-cond 栈**；如要跑文本 OCR 实验需移植，详见 [[server-226-31-and-1p4b]] 记忆）。
- **数据**：只能用 `/data07/dt_data/ywk/`（其他盘已满）。SA-Text parquet 在 `datasets/SA-Text`；训练产物 `VOSR_runs/`；conda 在 `conda_envs/vosr`。
- **1.4B 架构**：d1536/b36/h24 + **Qwen-Image f8c16 VAE**（latent 16×64×64），dinov2l，`distill_type=simple`。

---

## 5. 当前占用快照（2026-06-24）

| 机器 | 在跑 | 备注 |
|---|---|---|
| **226.31** | **1.4B teacher（ms_from_pretrained）847 评测推理**（screen `teacher40k`，GPU1，`inference/teacher40k_847.log`） | 25 步 cfg0.5/wc0.1，ETA ~4h；自动链 `harvest_teacher40k.sh` 推理完→拉 199→协议① CharAcc，验证 from-pretrained 是否修好识别欠拟合（旧从头训 36.82）。 |
| **199** | 空（gtsup 收割已完成） | GPU2 我方已释放；GPU0/1/3/4 多为他人作业，勿动。 |
| **本地** | 空（Exp B + gtsup 收割均完成） | GPU0-3 空闲。 |

> 已完结（详 `result_exp.md`）：Experiment B 结构增强×2a+2d（**结构增强未叠加**）；Real-CE 论文式监督微调 gtsup（**我们 Real-CE 微调最优 rec_acc 0.2991/PSNR 18.0，仍 < RRDB**）；1.4B teacher 训练 40000 干净收敛。

## 6. E2 在 199 的迁移要点（2026-06-21，可复用模板）

把本地分支的实验搬到 199 的最小步骤：
1. **代码**：`models/lightningdit.py` 199 与本地**完全一致**（已原生支持 OCR-cond，无需动）；`models/ocr_repa.py` 与 `train_vosr_distill.py` 本地是超集 → 直接 rput 覆盖（先 `.bak_pre_e2` 备份）。`build_ocr_repa_encoder` 定义在 `train_vosr_distill.py` 内（非 ocr_repa）。
2. **配置**：基于本地 E2 config，仅换 6 处 199 路径键（`train_dataset_config`→`sa_remote_dataset.txt`、`test_*_dir`→`SA/images`、`teacher/pretrained_ckpt`→`preset/checkpoint-00040000/...`、`ocr_boxes_index_path`→`VOSR_textsr/text_boxes_index.pkl`、`inference_steps`→1），recipe（dim1024/sd2/w=0.5/cond/shortcut/eff-16）保持与本地基线一致 → 干净单变量。
3. **数据对齐**：199 的 `SA/train.txt` 与本地 `text_hr_512_images.txt` 是**同一图集**（同序、119495 张 `sa_*_crop`），故 E2 训练数据与本地 2a-GT 基线完全一致。
4. **离线坑**：trocr-base-printed 已缓存 → `HF_HUB_OFFLINE=1`；dinov2 缓存在 `preset/ckpts/torch_cache` 但 `torch.hub.load` 仍 ping github → 已改 `source='local'` 用本地仓。
5. **收割**（E2 到 20000 后）：推理 NFE=1 → 协议① PP-OCR+pyiqa（`/data2/wyw/ywk/OSEDiff/metric_*.py`）+ 协议② TESTR spotting → 并入 `doc/result_exp.md`（§2 + §3.1/§3.2，对比 2a-GT 单变量）。

---

## 7. 各服务器环境位置与 workspace 速查

| 机器 | workspace（cwd） | python 解释器 | 激活方式 | 关键缺失/特性 |
|---|---|---|---|---|
| 本地 | `/data/ywk/VOSR` | `/home/ywk/anaconda3/envs/vosr/bin/python` | `conda activate vosr` | 有网；dinov2/PaddleOCR/shapely 齐全（主开发） |
| 199 | `/data2/wyw/ywk/VOSR_textsr` | `/home/yawen/.conda/envs/vosr/bin/python` | `source /data/anaconda/bin/activate && conda activate vosr` | **无外网**；**缺 shapely**（→只能 GT-box）；dinov2/pyiqa 离线缓存在 `preset/ckpts/torch_cache` |
| 226.31 | `/data07/dt_data/ywk/VOSR` | `/data07/dt_data/ywk/conda_envs/vosr/bin/python` | 直接用绝对路径 python | **无外网**；GPU0 常被 xsh 占；只能写 `/data07` |

- **远程操作助手**（本地机执行）：199 用 `python /data/ywk/claude_tools/{rexec,rput,rget}.py`；226.31 用 `{rexec_dt,rput_dt,rget_dt}.py`（第三参可调超时）。rexec 是**非登录 shell**，PATH 里没有 conda → 远程跑 python 要用**绝对路径解释器**（如上表），或在脚本里 `source ... && conda activate`。

## 8. 代码差异（建云端仓库前必读）

三台机的代码**同源但已分叉**，迁移/合并时必须 diff + 备份，切勿盲目覆盖：

| 仓 | 分支/基线 | 含 OCR-REPA | 含 OCR-cond(E1/E2) 训练接线 | detmap/real_ctc(2c/2d) | dataloader | 其他 |
|---|---|---|---|---|---|---|
| **本地** `/data/ywk/VOSR` | `text-distill-ablation`（**超集**） | ✅ | ✅ | ✅ | `realsr_dataset.py`（`TxtPairDataset`+GT-box `RandomCropWithParams`） | dinov2 已 patch `source='local'`；`build_ocr_repa_encoder` 在 `train_vosr_distill.py` 内 |
| **199** `VOSR_textsr` | 文本 SR 栈 | ✅（**仅 GT-box 可用**，无 shapely 不能在线检测） | ✅（2026-06-21 为 E2 从本地移植） | ✅（随 train 文件一并移植） | 同本地（已对齐） | `models/lightningdit.py` 与本地完全一致 |
| **226.31** `VOSR` | base `main`（**最旧**） | ❌ 无 | ❌ 无 | ❌ 无 | `SATextDataset`（**不同**，无 GT-box） | 有 `prune_checkpoints`；`train_vosr.py` 走多步 teacher；1.4B/Qwen-VAE 专用 |

**迁移要点**：
- 本地→199 已打通（见 §6）：`lightningdit.py` 一致免动；`ocr_repa.py`+`train_vosr_distill.py` 本地超集直接覆盖（备份 `.bak_pre_e2`）；config 仅换路径键；离线坑（HF offline + dinov2 `source='local'`）。
- 本地→226.31 **未打通**：226.31 是 base main，要跑 OCR 实验需把 OCR hook 移植进其 `train_vosr_distill.py`（以其为基）+ 给 `SATextDataset` 加 GT-box + 传 PaddleOCR 权重。详见记忆 [[server-226-31-and-1p4b]]。
- **dataloader 是最大分叉点**：本地/199 用 `realsr_dataset.TxtPairDataset`（HR 列表 + 在线 RealESRGAN 退化合成 LR + GT-box 随机裁剪映射）；226.31 用 `SATextDataset`（SA-Text parquet）。Real-CE 也走前者：52mm HR 列表 + 合成退化（非真实 13mm 配对）。
- **建云端仓库建议**：以本地 `text-distill-ablation`（超集）为主干；把三台机的「路径差异」抽成 config/env 变量（数据根、ckpt 根、解释器、离线开关），避免硬编码绝对路径；离线补丁（HF offline、dinov2 local、GT-box-only 回退）保留为默认。

---

## 9. 评测 / 推理 / 测试管线（命令 + 环境 + 代码检查）★

> **一句话铁律：`metric_*.py`（协议① PP-OCR+pyiqa）跑在 `osediff` 环境，不是 `vosr`！** 在 screen 里务必 `source /data/anaconda/bin/activate && conda activate osediff`，或用绝对路径解释器；用错环境会 `No module named 'paddleocr'` 立刻崩。

### 9.1 各脚本 ↔ 环境 ↔ 依赖（用错即崩，先查这张表）

| 脚本 | 作用 | **环境** | 关键依赖 | 机器 |
|---|---|---|---|---|
| `inference_vosr_onestep.py` | 一步学生推理（NFE=1） | **vosr** | torch/dinov2/SD2-VAE | 本地 / 199 |
| `inference_vosr.py` | 多步 teacher 推理（如 25 步） | **vosr** | 同上 + Qwen-VAE(1.4B) | 226 / 199 |
| `OSEDiff/metric_*.py` | **协议①** PP-OCR 识别 + pyiqa 画质 | **⚠️ osediff（非 vosr）** | **paddleocr**、pyiqa(LPIPS/MUSIQ/MANIQA) | **199** |
| `scripts/eval_realce.py` | **Real-CE** 基准（RGB+mask PSNR/SSIM+中英 CRNN） | **vosr** | **cv2**、editdistance、pyiqa、CRNN | 本地 |
| `realtext_eval/eval_spotting.py` | **协议②** TESTR 文本 spotting（det+E2E None） | **tair** | TESTR/detectron2、`PYTHONPATH=/data/ywk/TESTR` | 本地 |
| `realtext_eval/rescore_lexicon.py` | 协议② Full lexicon 重打分（离线，无 GPU） | **tair** | — | 本地 |

环境解释器：本地 `/home/ywk/anaconda3/envs/{vosr,tair}/bin/python`（**osediff 仅在 199**）；199 `conda activate {vosr,osediff}`；226 用绝对路径 `conda_envs/vosr/bin/python`。

### 9.2 标准收割链（一个 ckpt → 双协议 + Real-CE）

**(a) 一步学生推理**（vosr）。Real-Text 847：`-u 4 --infer_steps 1`；Real-CE（13mm→52mm 已配准）：`-u 1 --tile_size 512 --tile_overlap 64`。输出子目录恒为 `sd2_steps1_seed42_shortcut`。
```bash
python inference_vosr_onestep.py -c <CKPT_DIR> -i <LQ_dir> -o <OUT> \
  -u 4 --infer_steps 1 --align_method nofix --force_rerun
```

**(b) 协议① PP-OCR + pyiqa（osediff @ 199）** —— 从模板派生（换 SR 目录串 + 输出名）：
```bash
source /data/anaconda/bin/activate && conda activate osediff
cd /data2/wyw/ywk/OSEDiff
sed -e 's/realctc_20000/<NAME>/g' -e 's/evaluation_results_detmap.txt/evaluation_results_<NAME>.txt/g' \
    metric_realctc.py > metric_<NAME>.py        # SR 取 real_test/<NAME>/sd2_steps1_seed42_shortcut
# 或 SR_PATH 直改版：sed 's#SR_PATH = .*#SR_PATH = "<abs_SR>/sd2_steps1_seed42_shortcut"#' metric_2a_online_ft.py > metric_X.py
CUDA_VISIBLE_DEVICES=N python metric_<NAME>.py   # 出 PSNR/SSIM/LPIPS/MUSIQ/MANIQA/OCR-CER/OCR-CharAcc
```

**(c) 协议② TESTR（tair @ 本地）**：
```bash
cd /data/ywk/TAIR/realtext_eval
PYTHONPATH=/data/ywk/TESTR CUDA_VISIBLE_DEVICES=N /home/ywk/anaconda3/envs/tair/bin/python eval_spotting.py \
  --img_dir <SR>/sd2_steps1_seed42_shortcut --data_dir . \
  --config /data/ywk/TESTR/configs/TESTR/TotalText/TESTR_R_50_Polygon.yaml \
  --ckpt /data/ywk/TAIR-main/weights/totaltext_testr_R_50_polygon.pth --out_dir eval_out/<NAME>
/home/ywk/anaconda3/envs/tair/bin/python rescore_lexicon.py --eval_dir eval_out/<NAME> --lexicon full
# DETECTION_ONLY_RESULTS=det F1；E2E_RESULTS（None=几何/Full=lexicon 重打分）
```

**(d) Real-CE 基准（vosr @ 本地）**：
```bash
/home/ywk/anaconda3/envs/vosr/bin/python scripts/eval_realce.py \
  --sr_dir <SR>/sd2_steps1_seed42_shortcut --out_json <out>.json   # psnr/ssim/mask_*/lpips/rec_acc/rec_ned
```

**(e) 多步 teacher 推理（1.4B，vosr @ 226）**：
```bash
/data07/dt_data/ywk/conda_envs/vosr/bin/python inference_vosr.py -c <TCKPT> -i <LQ> -o <OUT> \
  -u 4 --infer_steps 25 --cfg_scale 0.5 --weak_cond_strength_aelq 0.1 --align_method nofix --force_rerun
# 输出子目录 qwen_steps25_cfg0.5_wc0.1；评测仍走 (b)/(c)，需拉到 199/本地
```

### 9.3 高频坑（已踩，勿再犯）
1. **screen 内 `conda activate` 静默回退 base**：`screen -dmS x bash -lc '... conda activate vosr; python ...'` 有时 activate 不生效 → `python`=base（缺 cv2/paddleocr）→ metric 崩。**根治：用绝对路径解释器**（`envs/vosr/bin/python`、`envs/tair/bin/python`），或在 osediff 用 `source /data/anaconda/bin/activate && conda activate osediff` 并验证 `which python`。
2. **`metric_*.py` 用错环境**：默认想当然 vosr → 必崩（paddleocr 在 osediff）。见 §9.1。
3. **rexec 跑长 until 轮询**会撞 paramiko 120s socket timeout；评测等待用本地后台 `run_in_background` 轮询远端 marker，别在单条 rexec 里 sleep 很久。
4. **跨机搬产物无 226↔199 直连**：226 产物 → 本地中转 → 199；用 `rget_dt` 拉到本地再 `rput` 上 199。

---

相关文档：[[AGENT.md]]（项目总览与文档索引）、`doc/node199_拓扑与配置原理.md`（199 多卡配置原理）、`doc/result_exp.md`（实验总记录）、`doc/realce_benchmark.md`（Real-CE 评测流程）。
