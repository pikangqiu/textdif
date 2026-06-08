---
name: node199-gpu-topology
description: "远端训练机 172.31.233.199(node199)的 GPU 拓扑、NVLink 布局,以及 VOSR 蒸馏多卡配置的设计原理与经验教训"
metadata: 
  node_type: memory
  type: project
  originSessionId: 67f8b725-e174-45ba-8598-3306eae189c2
---

第二台训练服务器 **172.31.233.199(node199)**,用户 `yawen` / 密码 `yawen123456`,仓库在 `/data2/wyw/ywk/VOSR_textsr`(与不可动的 `/data2/wyw/ywk/VOSR` 是同级目录),数据在 `/data2/wyw/ywk/datastes/SA/`。Conda 激活:`source /data/anaconda/bin/activate && conda activate vosr`(**不要修改这个环境**)。

**GPU 拓扑(`nvidia-smi topo -m`)**:5×80GB GPU,跨 2 个 NUMA 节点。
- NUMA0:GPU0、GPU1(互联 NODE)。GPU0 通常被别人占着。
- NUMA1:GPU2、GPU3、GPU4。**GPU2↔GPU4 = NVLink(NV8)**;GPU2↔GPU3 = PHB;GPU3↔GPU4 = NODE。
- 跨 NUMA(GPU1 ↔ {2,3,4})= SYS(QPI,最慢)。

**经验 —— 在这台机器上跑 VOSR 文字 SR 蒸馏(有效 batch 16)的多卡配置:** 任何 4 卡组合都必须包含 GPU1,这就强制引入一条慢的跨 NUMA all-reduce。配 `NCCL_P2P_DISABLE=1`(为稳定保留)时,all-reduce 走主机内存。
- 4 卡,1 图/卡,accum4(完全等效有效 batch)= **6.33 s/it,GPU 利用率仅 48-59%** —— 比 2 卡还慢。
- 2 卡 {1,4},2 图/卡,accum4 = 4.89 s/it。
- 4 卡 {1,2,3,4},2 图/卡(train_batch_size=8),accum2 = ~4.36 s/it,~37.5GB/卡,全 100%(「方案 A」)。
- **4 卡 {1,2,3,4},4 图/卡(train_batch_size=16),accum1 = ~4.17 s/it,~57GB/卡,全 100% —— 当时选定的最终配置。** 比 accum2 略快(单次更大的 fwd/bwd → 更少 kernel/Python 开销;OCR 解码翻倍没成为瓶颈),且用上了原本闲置的显存。设了 `NCCL_P2P_LEVEL=NVL`,但没明显胜过 `NCCL_P2P_DISABLE`。
- **重要耦合:** `on_sync_only` 的 OCR 在每个优化步只算一次,等效权重 ∝ `ocr_repa_weight / accum`。从 accum2→1 时为保持 OCR 贡献与方案 A 一致,把 `ocr_repa_weight` 从 0.5 减半到 0.25。(最初 B 基线的设计意图是 accum4 那档 = 0.125。)
- **结论:** 固定有效 batch 时,DDP 每个优化步只做 1 次 all-reduce(no_sync),与 accum 无关,所以改 accum 几乎不影响通信。增大每卡 batch 在这里有小幅收益、也用上了空闲显存;真正的天花板是经 GPU1 的跨 NUMA 通信,所以固定有效 batch 下加卡不会带来线性加速。

**OOM 抢占风险(2026-06-08):** 4 卡 4图/卡 配置(~57GB/卡)很脆弱 —— GPU1 是共享的,别人随时会抢。一旦别人的进程把 GPU1 填到 ~76GB,rank1 就 OOM,而那个 6 次重试循环每次都撞同一堵墙(第 1 次是 dinov2 的 github 抖动,第 2-6 次全是 GPU1 上的 `torch.OutOfMemoryError`)。**更稳的退路 = 2 卡 {1,4},2图/卡(~37GB/卡)**,留足余量。已回退为:`CUDA_VISIBLE_DEVICES=1,4`、`--nproc_per_node=2`、`train_batch_size=4`、`accum4`、`ocr_repa_weight=0.5`(用户选了字面基线值 → 等效 OCR 0.125,相对之前的 0.25 在训练中途形成一个 kink),有效 batch 仍是 16 → 同一个 `bs016` 目录,resume 可续训。远端配置备份:`*.yml.accum1_4card.bak`(4 卡版)、`*.yml.accum2.bak`(方案 A)。启动器 `run_gt_remote.sh` 现为 2 卡版(master_port 改为 29571)。world size 从 4→2 缩小只需要 random_states_0/1.pkl(本来就有),无需准备。

**关于「先启动不等于占住显存」(2026-06-08 用户问):** CUDA 没有按启动顺序的显存预留/优先级。显存是进程运行中按需 `cudaMalloc` 逐步分配的,57GB 是在几十秒加载 DiT→优化器→DINOv2→VAE→OCR 的过程中慢慢涨上去的。这次 OOM 的因果链:dinov2 去 github 校验时网络抖动 → 我方进程崩溃、释放显存 → 8s+ 重试间隙别人占满 GPU1 → 我方重启时无位可用 → rank1 OOM。dinov2 是「触发器」,不是耗显存的元凶。

**进度条 loss 显示的伪冻结(`on_sync_only` + 梯度累积):** postfix 里的 `loss` 是当前那个 micro-step 的值(`train_vosr_distill.py:1589`),而 OCR 只在每个优化步的最后一个(同步)micro-step 才加进去。第 1587 行 `if global_step % 50 == 0` 在 micro-step 循环内、`global_step` 只在同步步 +1,所以 50 倍数那一刻命中后,紧接着下一个优化步的前几个**非同步** micro-step 会再次 set_postfix,把含 OCR 的值覆盖成 v_loss-only。
- accum1(4 卡):没有非同步 micro-step,postfix 永远停在含 OCR 的值(显示冻结成如 0.0961)。
- accum4(2 卡):非同步步覆盖后,非 50 步显示 `loss == v_loss`(不含 OCR)。
- **两者都不是 bug。** 真实含 OCR 的正确值在每个同步步(每 50)与 `accelerator.log`(每 `log_loss_steps`,默认 250)中都对得上:`loss = v_loss + ocr_repa_weight × ocr_repa_loss`(逐行校验精确)。

**Resume 陷阱:** `train_vosr_distill.py` 里 `resume = '_resume' if args.resume_ckpt is not None` → 把 `resume_ckpt` 设成显式路径会给 exp/output 目录加 `_resume` 后缀,导致新旧 checkpoint 分裂、断点恢复失效。**正确做法:保持 `resume_ckpt: ~`** → 自动发现原目录里最新的 checkpoint 并继续存到那里。accelerate 的 `load_state` 需要每个 rank 的 `random_states_<rank>.pkl`;扩大 world size 时要复制已有的(cp random_states_0→2、1→3)。

**启动器 `run_gt_remote.sh`(当前 = 2 卡 {1,4}):** CUDA_VISIBLE_DEVICES=1,4、--nproc_per_node=2、master_port=29571、screen 名 `train_textsr_gt`、日志 `train_gt.log`。重启 = `cd /data2/wyw/ywk/VOSR_textsr && screen -dmS train_textsr_gt bash run_gt_remote.sh`。频繁 kill/重启后,要强制清理残留 rank(`pkill -9 -f train_vosr_distill.py`)并 bump `--master_port`,否则 rank0 会卡在被占用的端口上。如存在 [[text-distill-experiments]] 可参见。
