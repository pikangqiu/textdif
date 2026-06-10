# OCR 语义条件注入(OCR-cond)实验计划与执行手册

> 日期:2026-06-10。本文档覆盖:注入方案设计原理、已完成的代码改动、以及**之后所有实验的顺序、配置与启动命令**。
> 相关文档:`text_distill_2a_breakthrough_analysis.md`(结果分析)、`text_ocr_local_repa_experiment.md`(2a/2b 定义)。

---

## 1. 动机与定位

实验表已确立两个事实:

1. **蒸馏均衡器效应**:teacher 侧的改进在一步蒸馏后被洗掉(结构增强 teacher +1.9 → distill 仅 +0.24);所有"只改 teacher / 只改目标构造"的方法 OCR 全部聚在 0.66 天花板。
2. **唯一突破来自 student 侧监督**:2a 局部 OCR-REPA 直接监督 student 解码图,拿到 67.28%。

OCR 条件注入是第三类路径:**输入侧先验**。与 teacher 侧改进不同,条件在**推理时仍然存在**,不会被均衡器洗掉;与 2a(输出侧、train-only)互补,构成论文叙事:

> **"prior in, supervision out"**:输入端注入文本识别先验(OCR-cond),输出端施加区域局部文本监督(2a),两端夹击一步蒸馏的文字结构损失。

与文献的差异:TPGSR/TATT 在像素域 CNN 超分中注入文本先验;本方法注入到 **latent 一步流匹配蒸馏的 DiT student**,且与 train-only 局部损失组合。

## 2. 设计 v1(已实现)

完全复用 DINOv2 的条件通路模式:

```
LQ 图像 ──► 冻结 TrOCR encoder(microsoft/trocr-base-printed)──► tokens (B,577,768)
                                                                      │
DINOv2-b layer8 tokens (B,1024,768) ──► LayerNorm ──► mlp_ca ──► (B,1024,1024) ─┐
                                                                                ├─ token 维拼接 ──► 每个 block: x + CrossAttn(x, z)
OCR tokens ──► layer_norm_ocr ──► mlp_ca_ocr ──► × gate ──► (B,577,1024) ───────┘
```

关键设计点:

| 设计 | 选择 | 理由 |
|---|---|---|
| 编码器 | TrOCR encoder(非解码字符) | 软先验:LQ 上识别不可靠,encoder 特征错了只是弱化,解码出错字会引导生成错字 |
| 注入位置 | 与 DINO tokens 拼接进同一 CA | 零新增 block 结构;CA 不要求 token 空间对齐 |
| 门控 | 可学习标量 `ocr_cond_gate`,初始 0 | 训练起点与原模型功能等价(见 §4 说明);gate 曲线本身是"网络是否需要该先验"的直接证据 |
| 注入对象 | **仅 student**,teacher 不变 | teacher 收 `z` list 时只取 `z[0]`,自动忽略 OCR 流;蒸馏目标完全不变 |
| CFG/弱条件 | 与 DINO 同步置零 | `_zero_like(z)` 对 list 逐项处理,无需改 vosr.py |
| 推理 | 推理时同样从 LQ 提取并注入 | 这是它与 2a 的本质区别:先验保留到部署 |

成本:TrOCR encoder ~86M 参数(冻结),每步一次 384² ViT 前向;新增可训练参数仅投影 MLP(~5M)+ gate。

### v2(规划,暂不实现)

SwinSR 先粗 SR → 对粗 SR 提 OCR 特征 → 注入。解决"LQ 上 OCR 特征质量差"的根本问题,代价是推理多一级轻量 SR。**只有 v1 证明 gate 学到非零值但收益受限于 LQ 特征质量时才启动 v2。**

## 3. 代码改动清单(已完成,2026-06-10)

| 文件 | 改动 |
|---|---|
| `models/lightningdit.py` | `__init__` 增加 `ocr_cond_dim/ocr_cond_gate_init`;新增 `layer_norm_ocr + mlp_ca_ocr + ocr_cond_gate`;新增 `_project_cond_tokens()` 统一投影,`forward` 与 `forward_flexible` 共用 |
| `train_vosr_distill.py` | `build_ocr_repa_encoder` 支持 `model_name` 覆盖;`use_ocr_cond` 时构建冻结 TrOCR 并把 tokens 追加为 `z[1]`(主循环 + 验证采样两处);student 构建传入 `ocr_cond_dim`;postfix/tensorboard 记录 `ocr_cond_gate` |
| `inference_vosr_onestep.py` | 新增 `load_ocr_cond_encoder`/`extract_ocr_cond_tokens`;`get_venc_features` 追加 OCR tokens(平铺模式逐 tile 提取);模型构建传 `ocr_cond_dim`;`args.json` 里有 `use_ocr_cond` 时自动启用 |
| 新配置 | `VOSR_0.5B_text_guided_target_no_rc_ocr_cond.yml`(注入 vs B 单变量)、`..._ocr_cond_local_gt.yml`(注入+2a 组合) |
| `scripts/train_text_repa_ablation.sh` | 新增 `ocr_cond` / `ocr_cond_local_gt` 入口(此前已加 `ocr_ctc_gt`) |

## 4. 验证状态(CPU 冒烟,已通过)

- forward / forward_flexible 形状正确;OCR tokens 实际改变输出(gate≠0 时)。
- gate=0 时 OCR tokens **不携带任何信息**(577 个 token 全坍缩为共享 K/V bias 向量),只引入极小常量扰动(随机模型 max diff ~5e-3);v_loss 会在最初几步内吸收,蒸馏起点功能上等价于无注入模型。
- 梯度可流到 gate(gate 能离开 0)。
- 旧 ckpt → 新模型 `strict=False` 加载:缺失 key 恰为 7 个 OCR 模块参数,无多余 key。
- ⚠️ 尚未做 GPU 训练冒烟(4 卡被 2b 占用),正式启动前先按 §6 冒烟。

## 5. 实验队列(按执行顺序)

> 资源约定:本机 4×24GB,训练需 4 卡全占(每卡 batch 1),~5.6-6.2 s/it,20000 步 ≈ 31-34h。
> 每个实验完成后:推理 18000/19000/20000 三个 ckpt 各 847 张,报 mean±std。

### E0(进行中)2b-GT:局部 CTC 字符监督
- 状态:已启动 2026-06-10 21:13,screen `train_ocr_ctc_gt`,ETA ~6/12 早上。
- 问题:同框同位置下,字符级监督 vs 特征对齐(2a 67.28)谁强。
- 完成后推理:

```bash
EXP=exp_vosr_text_distill_ablation/ldit_distill_bs016_sd2f8c4_size512_ps2_psr1_d1024_b28_h16_uw0.0_cfgs0.5-r0.0-wc0.05-0.25_ts0e1_edr3_tduni_typetxt_distshortcut_text_ablation_guided_target_no_rc_ocr_ctc_gt
for STEP in 00018000 00019000 00020000; do
  CUDA_VISIBLE_DEVICES=0 python inference_vosr_onestep.py \
    -c ${EXP}/checkpoints/checkpoint-${STEP} \
    -i /data/ywk/datasets/real_test/LQ \
    -o preset/results/text_distill_ablation/ocr_ctc_gt_step${STEP} \
    -u 4 --infer_steps 1 --align_method nofix --force_rerun
done
```

### E1(下一个,已提前)OCR-cond v1:注入 vs B
- 单变量:B + `use_ocr_cond`。判断注入本身值多少。
- 启动(2b 结束、GPU 释放后):

```bash
cd /data/ywk/VOSR
screen -dmS train_ocr_cond bash -c 'NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_cond > logs/train_ocr_cond.log 2>&1'
```

- **判读标准**:
  - `ocr_cond_gate` 曲线离开 0 并稳定到非零 → 网络确实在用先验(论文图素材);
  - OCR vs B(65.99):+0.5 以上才算先验有效;
  - 若 gate≈0 不动 → LQ 上的 TrOCR 特征无增量信息,直接跳到 v2(SwinSR 两级)或放弃注入线。

### E2 OCR-cond + 2a 组合(论文主配置候选)
- 前置:E1 的 gate 非零且 OCR ≥ B。
- 启动:

```bash
screen -dmS train_ocr_cond_2a bash -c 'NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_cond_local_gt > logs/train_ocr_cond_local_gt.log 2>&1'
```

- 判读:vs 2a(67.28)的增量 = 注入在局部监督之上的边际价值;>67.8 即可写"两端互补"。

### E3 2c:真 GT 文字 CTC(代码待写,半天工作量)
- `text_boxes_index.pkl` 中已有 `texts` 字段(真转录)。把 2b 的 HR 伪标签换成 GT 文字:仅需在 `OCRRepaSystem` 加"字符串→CRNN 字典索引"映射并在 `compute_ctc_loss` 接收外部 targets。
- 价值:去掉伪标签噪声上限;若 2b 已超 2a,2c 大概率再涨。
- 可与 E1/E2 训练**并行准备代码**,排队跑。

### E4 2a 权重扫描(0.25 / 1.0)
- 改 `ocr_local_gt.yml` 的 `ocr_repa_weight`,确认 0.5 是否最优、是否复现"权重↑→拿 PSNR 换感知"模式。
- 两个值可分别在本机与 199 并行(199 配置已就绪,只改 weight)。

### E5 2a + 弱全局 REPAocr(0.25)
- 表中 REPAocr 是 PSNR/SSIM 冠军、2a 是 OCR/LPIPS 冠军;在 `ocr_local_gt.yml` 基础上加回 `repa_type: ocr, repa_weight: 0.25, repa_layer: 13`。
- 判读:能否同时 OCR≥67 且 PSNR≥25.5。

### E6(条件触发)OCR-cond v2:SwinSR 两级注入
- 触发条件见 E1。需要:接入现成 SwinIR/SwinSR 轻量模型 → 粗 SR → TrOCR 提特征,训练与推理同路径。

## 6. 正式启动前的 GPU 冒烟(每个新实验必做)

```bash
# 单卡 batch=1 冒烟(全局 train_batch_size 改 1,否则单卡吃 4 张图必 OOM——2b 冒烟的教训)
sed -e "s/^train_batch_size: 4/train_batch_size: 1/" \
    -e "s/^suffix: .*/suffix: '_smoke_ocr_cond'/" \
    configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_ocr_cond.yml > /tmp/ocr_cond_smoke.yml
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29588 \
  train_vosr_distill.py --config /tmp/ocr_cond_smoke.yml > logs/smoke_ocr_cond.log 2>&1 &
# 看第 50 步 postfix:应出现 ocr_cond_gate=...(初始 0.0,前几百步内应开始漂移)
# 通过后:pkill 对应进程,删除 exp_vosr_text_distill_ablation/*smoke_ocr_cond 目录,再启正式 4 卡
```

## 7. 监控与风险

- **gate 曲线是核心观测量**(tensorboard `ocr_cond_gate`):离开 0 的速度、稳定值的符号与大小,直接进论文分析。
- 显存:TrOCR encoder bf16 ~0.2GB + 激活,注入版应与 2a 持平(~21GB/卡);组合版(E2)最重,冒烟时盯第一个 sync 步。
- 风险:LQ 特征无信息 → gate 不动(E1 判读已覆盖);TrOCR processor 每步 CPU 预处理拖慢 s/it(若 >7s/it,改为 GPU 内 resize+normalize 旁路 processor);辅助 FD 队列路径未接 OCR tokens(当前实验不启用 FD,无影响)。
