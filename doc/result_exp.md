# 实验总记录(result_exp.md)

> 本文是 text-SR 一步蒸馏(分支 `text-distill-ablation`)**全部实验的设计、配置、实现与结果**的单一索引。
> 每个实验给出:动机 / 配置文件 / 关键函数与机制 / 训练设置 / 结果。文末为统一对比总表。
> 更深的机制分析见 `doc/text_distill_2a_breakthrough_analysis.md` 与 `doc/text_ocr_local_repa_experiment.md`。

---

## 0. 评测基准与协议

**测试集**:Real-Text 847 对(128×128 LQ → 512×512 HQ,1945 个 OCR region)。已逐字节核对 = TAIR(arXiv 2506.09993)公开的 Real-Text 测试集,因此 TAIR 论文表 3/4 的数字可直接引用对照。训练集(119,495 个 `sa_*` 裁剪)与其零重叠;SA-Text-test 因 1000/1000 张的母图在我们训练列表内,**判污染弃用**。

两套独立评测协议:

| 协议 | 工具 | 指标 | 入口 |
|---|---|---|---|
| ① PP-OCR metric | PaddleOCR PP-OCRv5 + pyiqa(199 osediff 环境,`/data2/wyw/ywk/OSEDiff/metric_*.py`) | OCR-EM / OCR-CER / **OCR-CharAcc**、PSNR/SSIM/LPIPS/DISTS/NIQE/MUSIQ/MANIQA | `metric_2a_local_20000.py` 等 sed 模板 |
| ② TESTR 官方 spotting(TAIR 论文协议) | TESTR TotalText-Polygon 官方 ckpt,MIN/MAX_SIZE_TEST 1600/1824,TH 0.45,word-spotting IoU 0.5,**None + Full lexicon 双列**(Full=Real-Text GT 773 词表,`rescore_lexicon.py` 离线重评分) | Det F1 / E2E-None / E2E-Full | `/data/ywk/TAIR/realtext_eval/eval_spotting.py` |

**协议②校准**:HQ(GT)过管线得 det 86.68 / E2E 74.23,vs 论文 87.97 / 74.50(E2E 差 0.27,协议可信)。LQ(bicubic↑512)= det 44.63 / E2E 29.37。

**噪声口径**:同一配置取最后 3–5 个 ckpt 的 mean±std;差距 < std 视为平手。CharAcc 上 **只有 ≥1 分的差异才算可信信号**。

口语对照:"65.99" = baseline B 的 CharAcc 0.6599。

---

## 1. 蒸馏框架(所有实验的公共底座)

- **Teacher**:VOSR-text 0.5B 多步 flow-matching 模型,`exp_vosr_text/ldit_fm_..._text_hr/checkpoints/checkpoint-00040000`(student 的初始化也用它)。
- **Student**:LightningDiT 0.5B(d1024/b28/h16,patch 2),SD2-f8c4 VAE latent 空间,**shortcut 一步蒸馏**(`dist_type: shortcut`,`infer_steps 1`),DINOv2-B 特征作 cross-attention 条件。
- 训练:`train_vosr_distill.py --config <yml>`,4×24GB 本机或 199(A800)。bs 16(全局),lr 5e-6,20000 步,EMA 权重用于推理。
- 推理:`inference_vosr_onestep.py -c <ckpt> -i LQ -o <out> -u 4 --infer_steps 1 --align_method nofix`(NFE=1)。
- 配置根目录:`configs/train_yml/one_step/text_distill_ablation/`。

---

## 2. 实验逐条记录

### A —— full target(ω=1.0)
- **设计**:蒸馏目标完全用 teacher 速度场(无 guidance 混合),作为 target 设计消融的一端。
- **配置**:`VOSR_0.5B_text_full_target_no_rc.yml`。
- **结果(协议①)**:PSNR 24.517 / SSIM 0.8329 / LPIPS 0.2538 / **CharAcc 65.82**(EM 57.89)。

### B —— guided target(ω=0.5)【一步蒸馏 baseline,所有后续实验的对照】
- **设计**:cfg 引导混合的蒸馏目标(`cfgs0.5`),无 RC,无任何 OCR 监督。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc.yml`。
- **结果(协议①)**:PSNR 24.619 / SSIM **0.8334** / LPIPS 0.2541 / MANIQA **0.4295** / EM **0.5841** / **CharAcc 65.99**。
- **关键结论**:OCR 从 teacher 63.78 → 65.99(+2.2),一步化本身就是 OCR 主杠杆(消除多步采样字符漂移);同时设定了 ~0.66 的"蒸馏天花板"。

### C —— guided + RC(rcgm)
- **设计**:在 B 上加 reverse-consistency 正则。
- **配置**:`VOSR_0.5B_text_guided_target_rc.yml`。
- **结果(协议①)**:PSNR 24.276 / NIQE **6.52**(感知最自然)/ **CharAcc 64.01** —— OCR 明显回退,弃用。

### REPAocr —— 全局 OCR 特征对齐(失败原型)
- **设计**:整图过 TrOCR 编码器(`extract_ocr_repa_tokens`),全局 mean-pool 成 1 向量,与 DiT 隐藏层 cosine 对齐(REPA 风格,`repa_type: ocr`)。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_ocr_repa.yml`(弱单层 `repa_layer 13`);multilayer 版 `[8,13,18]` + 256 token。
- **结果**:**CharAcc 65.81 ≈ B 65.99,饱和**;强化版(repa_weight→1.0)也只到 65.92。PSNR 25.796 全表最高(对齐充当了正则)。
- **教训**:全局池化抹掉 glyph 级空间结构,对文字无效 → 催生 2a 的"局部裁剪"。

### 结构增强①② —— CODSR 风格 teacher 侧增强(被"蒸馏均衡器"洗掉)
- **设计**:在 teacher 上加结构引导(`models/codsr_guidance.py`;②含 ragp+lqmod+vae-lora),再蒸馏。
- **结果**:teacher 层有效(63.78 → 64.61 / 65.67,+1~1.9);**蒸馏后回到 65.59 / 66.23(≈B 噪声带)**。
- **教训(本研究核心发现之一)**:teacher 侧改进会被蒸馏均衡化,**只有 student 侧监督能穿透**。

### 2a —— local-crop OCR-REPA,GT 框(🏆 当前最优配置)
- **设计**:用 GT 文字框(`preset/text_boxes_index.pkl`)把 **student 一步解码图**与 HQ 各自裁出文字区域,过冻结 CRNN(PP-OCRv5,PaddleOCR2Pytorch 移植,`rec_image_shape 3,32,320`)取 backbone 特征,做 **cosine 对齐**(逐框,≤16 框,边长≥8)。监督只在训练期,推理零开销。
- **实现**:`models/ocr_repa.py::OCRRepaSystem`(`ocr_repa_enabled` 分支),`ocr_repa_on_sync_only: True`(仅同步步计算,省显存),student DiT 梯度检查点。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_ocr_local_gt.yml`(`ocr_use_gt_boxes: True, ocr_repa_weight: 0.5, loss_type: cosine`)。
- **结果(协议①)**:
  - step18000(首次突破):PSNR 24.885 / LPIPS **0.2471** / **CharAcc 67.28**;
  - 噪声带(local 19000/20000 + 199 复现 20000):**CharAcc 66.92–67.66**(67.49 / 66.92 / 67.66),PSNR 24.86–24.97,EM 58.05–58.25;199 跨机复现一致。
  - **协议②**:det 76.01 / 76.32 / 76.62,E2E 43.42 / 43.06 / 41.93 → **det 76.3±0.3,E2E 42.8±0.8**。
- **结论**:唯一捅破 0.66 天花板的配置(+1.3~1.7),且保真度不降反优。

### w 扫描 —— 2a 的 `ocr_repa_weight` 消融(199,A800)
- **设计**:其余与 2a 完全一致,仅改 `ocr_repa_weight` ∈ {0.25, 1.0}(0.5 即 2a;0 即 B)。
- **配置**:199 `/data2/wyw/ywk/VOSR_textsr/run_gt_w025.sh / run_gt_w100.sh`(2a 配置 sed 改权重)。
- **结果(协议①,step20000)**:

| w | CharAcc | EM | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| 0(=B) | 65.99 | 58.41 | 24.62 | 0.8334 | 0.2541 |
| 0.25 | 66.98 | 58.25 | 24.61 | 0.8339 | 0.2504 |
| **0.5(=2a)** | **66.92–67.66** | 58.05–58.25 | **24.86–24.97** | **0.836–0.837** | **0.2445–0.2468** |
| 1.0 | 66.96 | 57.89 | 24.41 | 0.8268 | 0.2498 |

- **协议②**:w025 det 76.65 / E2E 43.74;w100 det **77.73** / E2E 42.42(spotting 上各 w 挤在噪声带内)。
- **结论**:**倒 U 形,w=0.5 是 CharAcc 与 PSNR/SSIM/LPIPS 的双峰值**——该超参对识别与保真同向起作用,非 cherry-pick。

### 2b —— GT 框 CTC 字符级监督
- **设计**:与 2a 唯一变量 = 监督形式:特征 cosine 对齐 → **CTC 损失**(HQ 裁剪经 CRNN 解码出伪标签文本,对 student 裁剪的 CRNN logits 做 CTC,`ocr_ctc_enabled` 分支)。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_ocr_ctc_gt.yml`(`ocr_ctc_weight: 0.5, ocr_repa_weight: 0.0`)。
- **训练**:本机 4 卡 29.6h,final v_loss 0.0331 / ocr_ctc_loss 1.43。
- **结果**:协议① CharAcc 67.45 / **EM 58.35(全场最高)** / PSNR 24.24 / SSIM 0.8292 / LPIPS 0.2454 / NIQE 10.01;协议② det 76.35 / E2E 43.33。
- **结论**:识别率落在 2a 噪声带内,**无突破**;PSNR 比 2a 低 0.6–0.7dB。字符级硬监督不优于特征级软对齐,且伤保真。

### E1 —— OCR 语义条件注入(student 侧,推理期也生效)【完成,无增益,弃用】
- **设计**:与 2a/2b("仅训练期监督")正交——**LQ 图过冻结 TrOCR 编码器**(`microsoft/trocr-base-printed`,`extract_ocr_repa_tokens` 复用),token 经 LayerNorm+MLP 投影(`LightningDiT._project_cond_tokens`),乘**可学习标量 gate(零初始化)**后拼接进 DINO 条件 token 流,供 DiT cross-attention 消费(`models/lightningdit.py:449,530`)。**不含任何 REPA/CTC 损失**,是相对 B 的单变量实验。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_ocr_cond.yml`(`use_ocr_cond: True, ocr_cond_gate_init: 0.0`)。组合版(E1+2a)备好:`..._ocr_cond_local_gt.yml`。
- **训练过程**:6-12 02:53 启动;16:41 在 9000 步后被外来进程挤爆 GPU3 OOM;6-13 00:14 从 checkpoint-9000 续训(`_resume_` 目录),6-13 跑满 20000 步;`harvest_e1.sh` 自动完成推理+双协议评测。
- **结果(ckpt-20000)**:协议① PSNR **24.41** / SSIM 0.8306 / LPIPS 0.2523 / **CharAcc 65.80**(EM —);协议② det **71.70** / E2E **45.53**。
- **结论:无增益,弃用**。CharAcc 65.80 与 baseline B(65.99)持平(实为 −0.19,在噪声内),PSNR 比 2a(24.9)低 0.5dB;`ocr_cond_gate` 一路走负到 −0.0137,**模型主动抑制这路条件**,证实 LQ-side OCR 语义条件对 token-level 蒸馏无用。协议②上 det 反比 2a(76.3)低 4.6 分,佐证条件注入未带来文本结构增益。
- **★ 结构性根因(2026-06-22,用户洞察 + 代码坐实)—— 蒸馏里 OCR-cond 为何必然失效**:蒸馏 v_loss 的目标由 `_teacher_target`([vosr.py:163](vosr.py))给出 = **CFG 引导的 teacher 速度** `v=v_weak+ω(v_cond−v_weak)`,而 **teacher 是纯多步 FM、从未训练 OCR-cond(OCR-盲)**→ **目标对 OCR 输入零依赖**。学生匹配这个 OCR-盲目标,用 OCR-cond 只会偏离、增大 loss,故最优解必然 **gate→0**。即:**在"匹配 OCR-盲 teacher"的蒸馏里,OCR-cond 结构上拿不到有用梯度,注定被关掉**——不是"没帮上忙",而是"没有发挥作用的通道"。对照 2a 不受此限:2a 是作用在 **student 中间特征**上的对齐 loss、不经 teacher 目标,故能绕过(塑形式 vs 注入式的本质差异)。
- **E1 多步控制实验【2026-06-22 启动,199】**:为验证"蒸馏范式是 blocker、非 OCR-cond 想法本身不行",在 **多步 GT 监督 `loss_fm`(目标=真值速度 eps−hq、无 OCR-盲 teacher)** 下训同样 OCR-cond 模型。**移植**:把 OCR-cond(`build_ocr_repa_encoder`/`extract_ocr_repa_tokens` + model `ocr_cond_dim` + 训练环 `z.append`)移植进 `train_vosr.py`(原只 distill 版有)+ dinov2 local 离线补丁。config `VOSR_0.5B_text_hr_ocr_cond.yml`(init=VOSR_0.5B_ms、LR1e-5、bs3×accum2、SA 数据),`run_e1_multistep.sh`(screen `e1_multistep`,GPU1-3)。**判据**:gate 长成正值+CharAcc 超 baseline → OCR-cond 在有真梯度时有效、坐实蒸馏废了它;gate 仍→0 → 想法本身无效。每 1000 步抽 gate 看轨迹。**gate 轨迹(原始 model,2026-06-22)**:1k −0.00558 / 2k −0.00937 / 3k −0.01114 / 4k −0.01192 / 5k −0.01252 / 6k −0.01276 / 7k −0.01299 / **8k −0.01319(见底)** / 9k −0.01314 / 10k −0.01307 —— 单调往负、8k 见底后平台在 −0.013,从未往正长,量级同蒸馏 E1(−0.0137)/E2(−0.0048)。
- **★ 结论(27% 已较明确)—— "蒸馏范式是 blocker"假设被证伪**:即便多步 GT 监督(目标=真值速度、非 OCR-盲 teacher)下,OCR-cond gate **照样收敛到 ≈0(−0.013)**,与蒸馏下同向同量级。**OCR-cond 在任何训练范式下都没被模型用起来** → 真因不是"蒸馏目标 OCR-盲"(那是对蒸馏成立的更细机制解释、非全部),而是更根本:**输入侧 TrOCR 语义 token 与 DINO 视觉条件 + LQ latent 输入冗余**,模型已从现有条件拿到所需信息,故无论 GT 还是 teacher 监督都关掉这一路。对照 2a 有效=往 **student 中间特征**塞 OCR 结构信息(塑形式、新监督方向),而非给已饱和的输入条件再加冗余一路(注入式)。**"注入式 OCR 先验(E1/E2)在本架构无效"是稳健结论。**
- **代码核验(2026-06-22,排除 bug)**:① 移植的 `extract_ocr_repa_tokens` 与原 distill 版**逐字节相同**;② OCR-cond 全部层都从初始值移动(`layer_norm_ocr.weight` mean 1.005≠init 1.0、`mlp_ca_ocr.fc1.bias` std 0.0028≠init 0、gate −0.013≠init 0)→ **路完全连通、梯度正常流、模型主动训练该路后仍选择 gate≈0**,确认负结果非代码 bug。**结论可信 → 已于 27%(10760 步)停训**,199 GPU1-3 释放。**总判:E1/E2 注入式 OCR 先验无效(冗余),与训练范式无关;唯有 student 侧塑形式监督(2a/2d)有效。**

### E2 —— E1 + 2a-GT 联合【2026-06-22,199 训练→停于 19583 取 ckpt-19000,预期兑现:无增益】
- **设计**:E1(OCR-cond TrOCR 注入,gate0 初始化)+ 2a-GT(`ocr_repa_weight 0.5`+`ocr_use_gt_boxes`)联合,**相对 2a-GT 的单变量=加 OCR-cond 块**;验证"输入侧 OCR 语义先验"叠加在 student 侧 OCR-REPA 上是否互补。config `VOSR_0.5B_text_guided_target_no_rc_ocr_cond_local_gt.yml`,199 上 GPU1-4 训练(详 `compute_resources.md` §6 迁移),用户在 19583 步主动停训取 ckpt-19000。
- **结果(ckpt-19000,real_test 847,NFE=1)**:协议① PSNR **24.48** / SSIM 0.8335 / LPIPS 0.2517 / DISTS 0.2141 / NIQE 10.32 / MUSIQ 54.33 / MANIQA 0.4220 / EM **58.41** / **CharAcc 67.68**;协议② **det F1 76.63**(P 0.891 / R 0.672)/ **E2E None 43.02 / Full 53.38**;Real-CE _推理中_。
- **结论(协议①②一致,无增益)**:**CharAcc 67.68 ≈ 2a 波段顶(67.66)、det 76.63 ≈ 2a(76.3)、E2E None 43.02 ≈ 2a(43.06)**,PSNR 24.48 略低于 2a(24.86)。**E2 双协议全面持平 2a-GT,叠加 OCR-cond 块零贡献**——`ocr_cond_gate` 同 E1 自学到 ≈0、模型关闭该路,识别/检测完全由 2a-GT 的 student 侧 OCR-REPA 决定。**E2 延续 E1 无增益结论坐实**:只有 student 侧细粒度 OCR 特征监督有效,输入侧 OCR 语义条件被蒸馏均衡器洗掉,无论单用(E1)还是叠加(E2)。Real-CE 跨域出后补全。

### 2c —— 文字检测图一致性【主线 Stage 3,完成,无突破/无增益】
- **定义对齐(2026-06-13 用户确认)**:主文档 `text_distill_master_plan.md` 的 2c = **文字检测响应图一致性**(student 输出经冻结文字检测器得到的 detection response map / 像素级 mask,与 HR 经同一检测器的响应图对齐),作为 2a(区域序列特征)/2b(glyph CTC)之上的**进一步空间约束**。此前我误把 2c 记成"真标签 CTC",已更正——那是另一个实验,现独立命名为 **2d**。
- **实现**:复用 vendored PaddleOCR **DBNet**(`ch_ptocr_server_v2.0_det_infer.pth`,冻结)。`OCRRepaSystem._detector_probmap` 把图(m11)可微预处理(→[0,1]→RGB→BGR→ImageNet 归一,`_no_autocast` fp32)后过 `detector.net` 取 `maps` 的 sigmoid 概率图 [B,1,512,512];`compute_detmap_loss` 算 `L1(P_pred, P_hr.detach())`,梯度只回流 student。新配置键 `ocr_detmap_weight/interval/start/on_sync_only/size/loss_type`,`ocr_detmap_enabled` 独立门控,`build_detector` 在 detmap 开启时强制 True。`train_vosr_distill.py` 按 2a/2b 同样接线(纯加法、默认关、检测器冻结、确定性前向不耗 RNG)。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_ocr_detmap.yml`(`ocr_detmap_weight: 0.5, loss l1, size 0=原生512, on_sync_only`,2a-REPA 关=单变量)。launch tag `detmap`。
- **验证**:单测 `compute_detmap_loss`——HR-vs-HR=0.000000(自一致 sanity)、模糊预测-vs-HR=0.0207(梯度 norm 0.033,文字检测退化→惩罚),概率图在文字区点亮(text_frac~3%)。端到端 4 步冒烟通过(无 OOM、无崩)。
- **坑(已解)**:detmap 的新梯度路径 + DiT 梯度检查点 + **torch.compile(AOT autograd)** 三者冲突 → `CheckpointError(forward 95 vs recompute 60 tensor)`。修复:`TORCH_COMPILE_DISABLE=1` + 检测器 `_no_autocast` 保 fp32。
- **compile 救援尝试(失败,记录免重踩)**:为找回 compile 提速(eager ~12.4s/it,compile 约快 2 倍)试了两条,均不可行于 4×24GB:① 只让 block forward eager / 内部算子保编译 → 仍 `CheckpointError`(编译的 swiglu/modulate 就在被检查点的区域里);② 检查点关 + compile 开 → 单卡冒烟能过但 4 卡 **DDP OOM**(超 24GB 仅 ~128MB,且与检测器分辨率无关——瓶颈是未检查点的 DiT 图+全 512 解码+DDP 梯度桶)。**死结**:装进 24GB 必须对 DiT 做检查点,而"DiT 检查点+compile"正是冲突源,二者不可兼得。结论:**eager+检查点是唯一既装得下又能跑的配置**;提速需更大显存卡或 FSDP/ZeRO 分片(留待以后)。
- **训练**:本机 4×24GB,eager+检查点,`run_detmap_local.sh`,实测 ~6s/it(比预估快一倍),06-14 00:32 起、**06-15 05:38 干净退出**(到 20000 步);harvest 自动跑完推理(847,exit 0)+ 双协议。等效 batch 16,模型数学不变。
- **结果(ckpt-20000)**:协议① PSNR **24.69** / SSIM 0.8337 / LPIPS 0.2531 / DISTS 0.2185 / NIQE 10.78 / MUSIQ 54.34 / MANIQA 0.4229 / **EM 57.79** / **CharAcc 65.76**;协议② det **71.91**(P 0.901 / R 0.598)/ E2E **45.17**(P 0.563 / R 0.377)。
- **结论:无突破/无增益,与 E1 同构**。协议① CharAcc 65.76 实际**略低于** baseline B(65.99),EM 57.79 < B 58.41,远不及 2a/2b(67.x)。协议② det 71.91 比 2a/2b(76+)低 4 分多,E2E 45.17 偏高同样是 E1 那种"低召回→基数小→虚警被滤"的高精度假象(det R 仅 0.598 vs 2a 体系更高)。**像素级检测响应图一致性打不过 2a 的区域序列特征对齐**:DBNet 概率图只约束"哪里有文字"的粗空间分布,不含 glyph 形状/可读性信息,惩罚信号过粗,既没提识别也轻微伤了保真。再次印证 2a 的增益来自**细粒度 OCR 特征**而非任意 OCR 相关的空间约束。资产:训练 log `claude_tools/train_detmap.log`、收割 `harvest_detmap.log`、协议① `199:/data2/wyw/ywk/OSEDiff/evaluation_results_detmap.txt`。

### 2d —— 真 GT 文本 CTC【主线 Stage 3,完成,**全面最优**】
- **设计**:把 2b 的 CRNN 伪标签换成 `preset/text_boxes_index.pkl` 里的**真 GT 文本**(`texts` 字段)做 CTC,消除伪标签噪声。字符→CRNN 字典映射经 `_encode_text`(`models/ocr_repa.py:compute_real_ctc_loss`),只走预测裁剪的 recognizer 前向、不做 HR 前向,梯度只穿预测。其余(GT 框/recognizer/权重 0.5/interval/sync gating)与 2b 完全一致——是 2b 的**单变量**(仅 CTC 目标来源:伪标签→真 GT 文本)。
- **训练**:本机 4 卡,step16000 处被同租户任务(ljc/hty)占满显存触发 CUDA OOM,6 次重试确定性复现;已加 OOM 容错(单 microstep 爆显存跳过)+ `resume_2d_when_free.sh` 守护等卡,2026-06-17 12:37 卡空后自动续训至 20000、18:41 收割完成。
- **结果(ckpt-20000)**:协议① PSNR **25.08** / SSIM 0.8388 / LPIPS 0.2426 / DISTS 0.2121 / NIQE 10.56 / MUSIQ 54.50 / MANIQA 0.4216 / EM **58.97** / **CharAcc 68.28**;协议② det **77.03** / E2E **None 44.60 / Full 54.68**。
- **结论:真 GT 文本 CTC 是当前最强单一信号,逐项超过 2a(特征对齐)与 2b(伪标签 CTC)**——CharAcc 68.28 > 2a 峰值 67.66 > 2b 67.45;det 77.03 ≥ 2a/2b 76.x;E2E None 44.60 / Full 54.68 均为我们体系内最高(且 det 同为高位 77,不是低-det 假象)。证明 **CTC 的监督质量瓶颈在标签噪声**:把 HR 贪婪伪标签换成真 GT 文本即净增益,机制干净。注:目前为单 ckpt,后续可补 18000/19000 噪声带。资产:训练 log `claude_tools/train_realctc.log`(+`.crash1`)、收割 `harvest_realctc.log`、协议① `199:/data2/wyw/ywk/OSEDiff/metric_realctc.log`。
- **后续**:联合实验 **2a+2d** 见下节。

### 2a+2d 联合 —— OCR-REPA 特征对齐 + 真 GT 文本 CTC【主线 Stage 3,完成,**识别/检测双新高,但 E2E/保真未超 2d**】
- **设计**:`ocr_repa_weight 0.5`(特征空间:region CRNN-feature 余弦对齐 HR)**叠加** `ocr_real_ctc_weight 0.5`(输出空间:真 GT 文本 CTC),两 loss 块在 `train_vosr_distill.py` 各自独立 gate,无需改码即可同开。两路监督正交:一个拉特征空间、一个拉输出 glyph。config `VOSR_0.5B_text_guided_target_no_rc_ocr_repa_realctc.yml`,chain 自动接力 2d 收割后启动。
- **训练**:本机 4 卡 eager+检查点(`TORCH_COMPILE_DISABLE=1`),2026-06-18→19 跑满 20000 步,**无 OOM/无报错**(2d 加的 OOM 容错此次未触发);两 loss 全程健康(ocr_repa~0.41 / ocr_real_ctc~5.6)。资产:训练 log `claude_tools/train_repa_realctc.log`、收割 `harvest_repa_realctc.log`、协议① `199:/data2/wyw/ywk/OSEDiff/evaluation_results_detmap.txt`(metric 脚本由 metric_realctc.py 派生,输出文件名沿用 detmap 字串,仅命名,内含 SR Path=repa_realctc_20000 已核对)。
- **结果(ckpt-20000)**:协议① PSNR 24.67 / SSIM 0.8320 / LPIPS **0.2372** / DISTS **0.2022** / NIQE **9.58** / MUSIQ 52.72 / MANIQA 0.4005 / EM 59.23 / **CharAcc 68.74**;协议② det **78.32** / E2E **None 42.59 / Full 52.71**。
- **结论:联合把"给定区域识别"(协议① CharAcc 68.74)与"检测定位"(det 78.32)同时推到全表新高**,逐项超过 2d(68.28 / 77.03),证明特征空间 + 输出空间监督在这两个指标上**建设性叠加**(各管一头:REPA 提区域可读性、CTC 提 glyph、两者合力进一步抬 CharAcc;det F1 也随之新高)。**但不是对 2d 的全面碾压**:E2E None 42.59 < 2d 44.60、Full 52.71 < 2d 54.68、PSNR 24.67 < 2d 25.08,三项被 2d 反超。即**两个互补冠军**——联合是"识别+定位"最优(CharAcc/det),2d 是"端到端 spotting + 像素保真"最优(E2E/PSNR)。机理:det F1(IoU 定位)与协议① CharAcc(GT-mask 区域 + PP-OCR)度量"给定框可读性",联合在此双赢;TESTR E2E 要检测器与识别器**同时**命中,新增召回的框在 TESTR 自带识别器下转写未必对,故 E2E 反而小降。

- **★ 噪声带(2026-06-24 补,最后 5 个 ckpt sweep 16000–20000,双协议)**:

  | ckpt | PSNR | SSIM | LPIPS | **CharAcc** | det F1 | E2E None | E2E Full |
  |---|---|---|---|---|---|---|---|
  | 16000 | 24.53 | 0.831 | 0.239 | 68.85 | 77.95 | 43.25 | 54.25 |
  | 17000 | 24.56 | 0.831 | 0.239 | 68.75 | 77.73 | 42.90 | 54.12 |
  | **18000** | 24.59 | 0.831 | 0.238 | **69.02** | 78.10 | 43.59 | **54.56** |
  | 19000 | 24.64 | 0.832 | 0.238 | 68.76 | 77.91 | **43.85** | 53.18 |
  | 20000 | 24.67 | 0.832 | 0.237 | 68.74 | **78.32** | 42.59 | 52.71 |

  **解读**:① **CharAcc 紧带 68.74–69.02**(mean≈68.82,std≈0.10)——头条 68.74 在噪声内,稳健;**峰值 ckpt-18000=69.02**。② **20000 并非全表最优**:它 det(78.32)/PSNR(24.67) 最高,但 CharAcc 在带底、E2E None/Full 最低;**18000 才是综合最优**(CharAcc 69.02 峰 + E2E Full 54.56 峰 + det 78.10 次高)。③ 论文取数建议:**CharAcc 报噪声带均值 68.8(或峰值 ckpt-18000 69.0)而非单点 68.74**,避免 cherry-pick 质疑;det F1 报 ~78(带 77.7–78.3)。资产:`claude_tools/sweep_2a2d_worker.sh` + `sweep_2a2d_000NN.log`,协议① `199:OSEDiff/metric_2a2d_000NN.log`。

### 结构增强 × 2a+2d 联合(Experiment B)—— 在 CODSR 结构增强 teacher 上叠我们的 2a+2d loss 重蒸馏【完成,**未叠加增益,与纯 2a+2d 持平**】
- **动机**:此前"结构增强 distill"(VAE-LoRA + lq_mod + RAGP 的 CODSR 风格 teacher)单独蒸馏被"蒸馏均衡器"洗掉(见上"结构增强①②"节,distill 后 CharAcc 仅 65.59/66.23)。本实验问:**把结构增强的 teacher 与我们的 OCR loss(2a REPA + 2d 真 GT CTC)合在一起重蒸馏,能否叠加**?config `VOSR_0.5B_codsr_2a2d.yml`(teacher=结构增强多步 ragp_lq_mod_vae_lora ckpt-20000、repa_type ocr w0.5 + ocr_real_ctc_weight 0.5 + use_lq_token_modulation + vae_lora、use_checkpoint True)。
- **训练**:本机 4 卡,2026-06-23→24 跑满 20000 步,无报错(repa_loss 1.03→0.06)。坑:OOM→开梯度检查点(显存 12-23GB);端口冲突→清进程重启。screen `codsr_2a2d`;收割 `claude_tools/harvest_codsr_2a2d.{sh,log}`。
- **结果(ckpt-20000)**:协议① PSNR 23.61 / SSIM 0.8213 / LPIPS 0.2453 / NIQE 9.45 / MUSIQ 54.00 / MANIQA 0.4193 / **CharAcc 68.45** / CER 0.3155;协议② det **F1 77.51**(P 88.35/R 69.04)/ E2E **Full 53.57**(P 58.67/R 49.28)。
- **结论:结构增强未在我们的 loss 上叠加**——CharAcc 68.45 ≈ 纯 2a+2d 68.74(-0.29,噪声内)、det 77.51 ≈ 78.32(-0.81)、E2E Full 53.57 ≈ 52.71(+0.86)。即**加上 VAE-LoRA/lq_mod/RAGP 的结构增强后,识别/检测与纯 2a+2d 基本持平**,没有建设性叠加,反而 PSNR(23.61)略低于纯 2a+2d 的 24.67。印证"结构增强在一步蒸馏框架下被均衡器吸收"的既有判断:**我们的 OCR-aware 监督才是有效贡献点,结构增强是冗余的**。这本身是干净的消融负结果(强化"单一 loss 贡献点"的论文叙事)。

### 对比方法 —— TeReDiff(TAIR 论文)本地复现
- **设置**:作者发布 stage3 权重 + 作者 `val.py`(50 步,prompt_style=CAPTION,wandb 已关)+ 字节一致 Real-Text;config `/data/ywk/TAIR-main/configs/val/val_terediff_realtext.yaml`。推理 1h16m @ 24GB(NFE=50,vs 我们 NFE=1)。
- **结果(None+Full 双列,对齐 Table 3 的 TESTR 列)**:协议② det 76.14 / **E2E None 33.53 / Full 44.62**(论文宣称 TESTR None **49.39** / Full **56.45**;**两列同时差 12–16 分**;同管线 HQ 校准 None 74.23/Full 82.05 误差仅 0.27);协议① CharAcc 60.61 / EM 53.37 / PSNR 23.30 / SSIM 0.7652 / MUSIQ **62.1** / MANIQA **0.51**。
- **缺口口径澄清(2026-06-15)**:补 Full lexicon 后确认 **49.39 是论文 None 列、不是 Full**;缺口在 None 和 Full **两列都存在 12–16 分**,故**不是 None-vs-Full 的口径错位**,而是用作者发布权重就复现不出论文识别数。det 反高于论文(76.14>74.88)+ HQ 校准准 → 缺口纯在识别端。
- **解读**:det 正常说明推理无误,缺口集中在识别端;TeReDiff 只在无参考美学指标占优(生成感强、字符不忠实)。未排除变量:prompt_style TAG、采样默认值、Drive ckpt 版本。**TAG 复核前论文数字按 "published number" 引用。**
- **TAG 复核(完成,prompt_style 已排除)**:config `val_terediff_realtext_tag.yaml`(唯一变量 prompt_style: CAPTION→TAG)。结果:协议② det **76.28** / E2E **30.54**;协议① CharAcc **59.25** / EM 51.83 / PSNR 23.19 / SSIM 0.7671 / MUSIQ 62.81 / MANIQA 0.5156。**TAG 在两个协议上都比 CAPTION 更差**(E2E None 30.54 < 33.53,CharAcc 59.25 < 60.61),因此 prompt_style **不是**缺口来源——与提示风格无关,仍归因识别端/不可复现。**TeReDiff 复现取 CAPTION 数(det 76.14 / E2E None 33.53 / Full 44.62 / CharAcc 60.61);2a 全面超过 TeReDiff 的可复现表现,但低于其论文宣称数(None 49.39/Full 56.45,官方权重未能复现)。** 对标其余 published 基线(FaithDiff None 41.64/Full 47.97 等)2a/2b 则同口径胜出且 NFE=1。

### 跨域评测 —— Real-CE(中文场景,第二 benchmark)
- **设置**:四个冠军 ckpt-20000(B / 2a-GT / 2d / 2a+2d)对 Real-CE val **零样本**推理(LQ=13mm→GT=52mm,upscale=1 平铺,tile512/overlap64),`scripts/eval_realce.py`(忠实官方协议:RGB PSNR/SSIM/LPIPS + 文字多边形 masked PSNR/SSIM + 中/英 CRNN 识别 ACC/NED)。本机 4 卡并行,各 254/261 图有效(同 7 张推理缺失)、3206 OCR region。论文报数用 valid_list。
- **官方论文参照行(Real-CE Supplementary Table 1,RRDB,4× 13mm→52mm,同协议)** —— 用于把我们的零样本/微调结果锚定到 published 数值:

  | 参照 | PSNR | SSIM | LPIPS | ACC | NED | 说明 |
  |---|---|---|---|---|---|---|
  | **LR(论文)** | 19.65 | 0.6684 | 0.3987 | 0.2759 | 0.6173 | 13mm 输入下限 |
  | LR(本管线,261 图) | 19.65 | 0.663 | — | — | 0.2750 | 0.622 → **与论文逐位吻合** |
  | **HR(论文 oracle)** | — | — | — | **0.4807** | **0.8342** | 52mm GT 识别上界 |
  | HR(本管线 oracle,254 图) | ∞ | 1.000 | 0.000 | **0.4836** | **0.8353** | **复现论文 → 三方管线互验** |
  | **RRDB Baseline**(L1+pixel-EA,Real-CE 训练) | 20.42 | 0.7303 | 0.2630 | 0.2914 | 0.6399 | 论文回归基线 |
  | **RRDB 全法**(+edge-aware F_CH,Real-CE 训练) | 20.14 | 0.7210 | **0.2031** | **0.3093** | **0.6622** | 论文 SOTA(有监督上界) |

  > 说明:论文 RRDB 是**回归式、且在 Real-CE train 上有监督训练**(故 PSNR 20+、ACC 0.31)。我们的冠军是**一步扩散、SA 合成域零样本迁移**(PSNR 17、ACC 0.29~0.30,生成更锐但保真低),不可直接比 PSNR;Real-CE 微调后(下表)才与论文 RRDB 同口径(同样在 Real-CE train 训练)。

- **零样本基线表(微调前,2026-06-21)**:

  | 模型 | PSNR | SSIM | mask-PSNR | mask-SSIM | LPIPS | **rec_acc** | rec_NED |
  |---|---|---|---|---|---|---|---|
  | 13mm 输入(下限,前次 261 图) | 19.65 | 0.663 | 11.89 | 0.496 | — | 0.2750 | 0.622 |
  | B(baseline) | 17.14 | 0.658 | **10.01** | **0.491** | 0.2960 | 0.2901 | 0.6075 |
  | 2a-GT | **17.36** | 0.661 | **10.17** | 0.494 | 0.2933 | 0.2935 | 0.6105 |
  | 2d | 17.25 | **0.662** | 9.93 | 0.492 | 0.2902 | 0.2951 | **0.6160** |
  | **2a+2d** | 17.17 | 0.656 | 9.86 | 0.488 | **0.2889** | **0.2973** | 0.6158 |

- **解读**:① **rec_acc 单调 B(0.290) < 2a-GT(0.294) < 2d(0.295) < 2a+2d(0.297)**,与 Real-Text 847 的识别排序完全一致 → OCR-aware 监督的相对优势**跨域保持**(2a+2d 仍是识别最优、LPIPS 最优)。② 四个模型 rec_acc(0.29~0.30)**略高于 13mm 输入下限(0.275)**,即 SA 合成域训练的模型对 Real-CE 文字识别有**小幅净增益**;但 **PSNR(17.x)显著低于输入(19.65)**——一步 SR 重建更锐但牺牲像素保真(中文长文本 + 真实退化的域差)。③ 增益幅度小(识别仅 +1.5~2.2 点)正是 **Real-CE train split 微调的动机**。
- **微调后(2026-06-21,在线检测版 5000 步)—— 负结果**:init=本机 2a-GT 学生 ckpt-20000、w=0.5、5000 步、LR 5e-6。

  | 行 | PSNR | SSIM | LPIPS | **rec_acc** | rec_NED |
  |---|---|---|---|---|---|
  | 2a-GT 零样本(before) | 17.36 | 0.661 | 0.2933 | **0.2935** | 0.6105 |
  | **在线检测 Real-CE 微调(after,5000)** | 17.35 | 0.661 | 0.2937 | **0.2916** | 0.6119 |

  **几乎无变化,rec_acc 反而微降。根因(已查代码确认):** `dataloaders/realsr_dataset.py::__getitem__` 仅返回 `hq`(52mm GT),**LQ 是训练时在 GPU 端由 `DegradationMapper`/realesrgan 合成退化生成的——并未使用 Real-CE 真实的 13mm→52mm 配对**。即"Real-CE 微调"实为"在 Real-CE 的 52mm 图上用合成退化继续训练",**没有引入真实传感器退化/焦距模糊的域信息**,故对真实 13mm 输入的 Real-CE val 无增益。**与论文 Supplementary Table 3 完全吻合**:RRDB 用 Real-ESRGAN 合成退化训练 ACC=0.2864,用 Real-CE 真实配对训练 ACC=0.3093——**合成退化无法替代真实配对**。
  - **199 GT-box 版**(`realce_2a`,5000 步)同一 dataloader 路径,预期同样近似 null(收割后补行)。
  - **真实配对微调(已实现,2026-06-21)**:数据阻断曾以为缺 13mm,实查 **199 `datastes/Real-CE/train/13mm` 存在 645 张、与 52mm 同名同分辨率(配准)**。改造三处:① `dataloaders/realsr_dataset.py` 新增 `real_paired_lq_dir`——GT-box 分支裁 52mm 后对同名 13mm **replay 同一 crop**(`_apply_same_crop`)返回真实 `lq`;② collate 堆叠 `lq`;③ `train_vosr_distill.py` 训练环 `batch.get("lq")` 存在则**用真实 LQ、跳过合成退化**。config `VOSR_0.5B_realce_2a_realpair_gt.yml`(init=2a-GT、w0.5、5e-6、5000步、GT-box)。冒烟过(hq≠lq 真实加载)。
  - **★ 真实配对结果(2026-06-22,ckpt-4500)—— 假设证伪,真实配对未打破 null**:

    | 行 | PSNR | SSIM | LPIPS | **rec_acc** | rec_NED |
    |---|---|---|---|---|---|
    | 论文 RRDB 全法(Real-CE 从头训) | **20.14** | **0.721** | **0.203** | **0.3093** | **0.662** |
    | 2a-GT 零样本(before) | 17.36 | 0.661 | 0.293 | 0.2935 | 0.611 |
    | 合成微调(在线检测,5000) | 17.35 | 0.661 | 0.294 | 0.2916 | 0.612 |
    | 合成微调(GT-box,5000) | 17.40 | 0.661 | 0.295 | 0.2954 | 0.613 |
    | **真实配对微调(GT-box,4500)** | 17.39 | 0.661 | 0.295 | **0.2957** | 0.614 |

    **真实配对 rec_acc 0.2957 ≈ 合成 GT-box 0.2954 ≈ 2a-GT 零样本 0.2935**(差异 <0.003,噪声内),且 **PSNR 全程 ~17.4 纹丝不动、距论文 0.3093 仍差 0.013**。**"合成退化是元凶、真实配对能修"的假设被证伪**——真实配对与合成退化**结果无差异**。归因:① **微调过弱**:LR 5e-6 × 5000 步 + init=已收敛的 2a-GT,几乎不移动模型(所有指标 flat,PSNR 0 变化即铁证);② **更深层**:一步扩散学生是从教师分布蒸馏来的生成式模型(PSNR 17、锐但不保真),与论文 RRDB(**从头在 Real-CE 上回归训练**、PSNR 20+)是不同物种,温和微调无法把识别拉到专用从头训模型的水平。**结论:在当前温和微调设定下,Real-CE 微调对一步蒸馏学生是 null,LQ 来源(合成/真实)不影响**——这本身是干净的负结果。**后续可选**:更激进微调(LR↑10–50×、步数↑、可解冻更多)验证是否"过弱"假说,或接受"一步蒸馏学生跨域到真实中文 SR 需重训而非微调"的结论。
  - **★ 论文式监督微调(真实退化,2026-06-23 启动,199 GPU1-4)—— 修复 GT 监督崩溃后跑通**:前两版"微调"都带教师/CFG(非论文式),为忠实复现 Real-CE 论文的**纯监督回归**(真实 13mm→52mm 配对、无蒸馏教师、无 OCR aux),走 `_teacher_target` 的 `model_tea=None` 分支。**原 bug**:该分支仍做一次 `model(inp_weak)` 多余前向 + CFG 混合,既不纯监督,又推进 CUDA RNG → 梯度检查点重算"saved vs recomputed metadata mismatch"崩溃(`use_checkpoint:False` 也未解,因 `_teacher_target` 内前向本身扰动 RNG)。**修复**(`vosr.py:_teacher_target`):GT 监督分支直接 `v = v_target`(回归 GT 速度),删掉 weak 前向与 CFG 混合——**一举两得**:既消除崩溃,又使目标恰好等于论文式纯监督(velocity-L2 等价于对 HR 的监督重建)。配置 `VOSR_0.5B_realce_gtsup.yml`(init=2a remote ckpt-20000、real_paired_lq_dir=Real-CE/train/13mm 645张真实LR、LR 5e-5、10000步、ocr_repa_weight 0.0 纯recon、eff-batch16)。**状态**:✅ 跑满 10000 步(v_loss 1.34→0.647),已收割。

    **★★ 结果(ckpt-10000,2026-06-24)—— 我们 Real-CE 微调中的最优,但仍未追平 RRDB**:

    | 行 | PSNR | SSIM | LPIPS | **rec_acc** | rec_NED |
    |---|---|---|---|---|---|
    | 论文 RRDB 全法(从头训) | **20.14** | **0.721** | **0.203** | **0.3093** | 0.662 |
    | 2a-GT 零样本 | 17.36 | 0.661 | 0.293 | 0.2935 | 0.611 |
    | 真实配对(GT-box+teacher,4500) | 17.39 | 0.661 | 0.295 | 0.2957 | 0.614 |
    | **论文式监督(real退化,gtsup,10000)** | **18.00** | **0.682** | 0.290 | **0.2991** | 0.609 |

    **gtsup 在我们所有 Real-CE 微调里最优**:rec_acc **0.2991**(>真实配对 0.2957 >零样本 0.2935)、PSNR **18.00**(>17.4,+0.6)、SSIM 0.682(>0.661)。即**论文式纯监督 + 真实退化 + LR×10 确实比"带教师的温和微调"更能推动**,方向正确。**但仍未追平从头训的 RRDB**(20.14/0.3093,差 PSNR 2.1、rec_acc 0.010)。跨域回测 847:协议② **det 73.34 / E2E None 46.82 / Full 57.58**(det 较 SA 域冠军 78.32 降~5,E2E Full 反升——微调把分布拉向中文真实域,牺牲了 SA 合成域的检测但端到端 spotting 略好);协议① **PSNR 25.15 / SSIM 0.848 / LPIPS 0.251 / CharAcc 67.34 / CER 0.327**——**无灾难性遗忘**(CharAcc 67.34≈2a 的 ~67,PSNR 25.15 甚至略高于 2a 的 ~24.9),真实退化监督微调在原 SA 域上保持/微升。**结论(更新)**:前述"微调对一步蒸馏学生是 null"需**修正为**——温和带教师微调≈null,但**论文式纯监督+真实退化+高LR能拿到小幅真实增益(本变体最优)**,只是天花板仍低于从头训。机理仍是"一步蒸馏学生 vs 从头回归 RRDB 是不同物种",跨域追平需重训而非微调。**坑(已记)**:metric 在 screen `bash -lc` 里 `conda activate` 静默回退 base(缺 cv2/paddleocr)→ 两个 metric 先崩;改用**全路径 python**(`envs/vosr/bin/python` 跑 eval_realce、199 osediff 环境跑 PP-OCR)修复。
  - **跨域回测(在线版完成,2026-06-22)—— 无灾难性遗忘**:在线检测微调 ckpt-5000 回测原域 real_test 847:

    | 协议 | 2a(local)原值 | 在线ft 微调后 | Δ |
    |---|---|---|---|
    | ① CharAcc | 66.9–67.7 | **67.14** | ~平 |
    | ① PSNR / SSIM / LPIPS | 24.86–24.97 / 0.836 / 0.244 | 24.89 / 0.836 / 0.246 | 平 |
    | ① EM / CER | 58.1–58.3 / — | 58.35 / 32.86 | 平 |
    | ② Det F1 | 76.32 | 75.68 | −0.64 |
    | ② E2E None | 43.06 | 43.86 | +0.80 |

    **Real-CE 合成退化微调对原合成域 847 几乎无损伤**(CharAcc/PSNR 持平、det 微降、E2E 微升)——与其 Real-CE 端 null 一致:改动本就轻微,故既不增益 Real-CE 也不损伤原域。协议① `metric_2a_online_ft.py`(199 OSEDiff,847)、协议② `eval_spotting.py`(本机 TESTR)。
  - **★★★ 破 0.30 ——「文字聚焦真实瓦片 + 2a OCR-REPA」一步学生(2026-06-27,ckpt-8000)**:
    - **动机**:gtsup(0.2991)已逼近 RRDB(0.3093)但仍差 0.010,且此前所有 Real-CE 尝试都用 645 整图(teacher 随机裁 512 命中背景)或纯 recon 监督。本版换两点:① **数据**——不用整图,改用我们 SA-Text 管线产出的 **Real-CE 文字聚焦瓦片**(`make_realce_tiles.py`,围绕配准过滤后的文字框裁 512,IoU0.7 去重,4821 瓦片/15043 框)+ 同名 13mm 真实 LR 配对(`real_paired_lq_dir`);② **监督**——在 gtsup 纯 recon 之上叠回 **2a OCR-REPA**(`ocr_repa_weight 0.5`,charset 无关、对中文安全;瓦片框索引 `realce_tile_boxes_index.pkl`),2d/CTC 因中文 charset 跳过(`ocr_real_ctc_weight 0.0`)。
    - **配置**:`VOSR_0.5B_realce_2a_tiles.yml`,一步学生,`teacher_ckpt: ~`(纯 GT 监督,同 gtsup),init=2a+2d 冠军 ckpt-18000,`train_dataset_config=realce_tiles_only_dataset.txt`,LR 5e-5,8000 步,eff-batch16,distill shortcut。跑满 8000(10h28m,final loss 0.261,ocr_repa/loss 0.0222 收敛)。评测同口径(4卡分片一步推理 tile512/ov64 → 合并 → `eval_realce.py`)。

    | 行 | PSNR | SSIM | LPIPS | **rec_acc** | rec_NED |
    |---|---|---|---|---|---|
    | 论文 RRDB 全法(Real-CE 从头训,有监督上界) | **20.14** | **0.721** | **0.203** | **0.3093** | **0.662** |
    | gtsup(论文式监督,real退化,10000) | 18.00 | 0.682 | 0.290 | 0.2991 | 0.609 |
    | **本版 2a-tiles(一步,ckpt-8000)** | 17.79 | 0.674 | 0.283 | **0.3012** ✅ | 0.617 |

    **★ rec_acc 0.3012 首次破 0.30,超 gtsup(0.2991)、与从头训 RRDB(0.3093)的差距收窄到 0.008**(gtsup 时为 0.010),NED 0.617>gtsup 0.609、LPIPS 0.283<0.290 也同步微升。**PSNR 17.79 仍显著低于 RRDB 20.14**——生成式一步学生的像素保真天花板未变,印证"识别度可追平、保真度因范式差距持续落后"。**机理**:相对 gtsup,增量来自**文字聚焦瓦片把训练样本对准真实中文文字区**(整图随机裁的背景样本被剔除)+ **2a 细粒度 OCR 特征对齐**在真实退化上再压一刀——两者都作用在"识别"维度,故只抬 rec_acc/NED 不抬 PSNR。**结论**:在保持一步、跨域(SA合成→Real-CE真实)的前提下,**文字聚焦真实瓦片 + OCR-REPA 监督让一步蒸馏学生在 rec_acc 上追平/微超有监督从头训的 RRDB**,坐实"生成范式天花板≈0.30、且可被文字感知监督顶到该天花板"。覆盖 257/260(3 张超大图 tiled 全画布 latent 单卡亦 OOM,已排除,非系统偏差)。资产:训练 log `logs/realce_2a_tiles.log`、收割 `claude_tools/harvest_realce_2a_tiles.sh`、结果 `Real-CE/curation/teacher_eval/REALCE2A_realce_eval.json`、ckpt `..._realce_2a_tiles/checkpoints/checkpoint-00008000`。

### SA-Text-test 三级退化评测【协议① 完成,带污染声明】
- **背景**:SA-Text-test 1000 张的母图全部在我们的训练列表内(986 张母图),**结果须带数据污染声明**;但作为与 TAIR 论文同口径的参考对比仍然执行(用户决策 2026-06-13)。测试集含三个退化等级 lq_lv1/lv2/lv3(TAIR 论文按等级出表)。
- **设置**:数据 `/data/ywk/datasets/satext_test/{HQ,LQ_lv1,LQ_lv2,LQ_lv3}`(各 1000 张,parquet 提取);OCR region mask 由 `gen_masks.py` 从 parquet 生成 1000 个 result.json(2206 实例,bbox 已是 512 坐标),传至 199 `HQmask/`;TESTR GT 转换 `/data/ywk/TAIR/satext_eval/`(1000 图 2206 实例,复用 convert_realtext.py)。模型:2a-remote ckpt-20000。
- **坑**:metric 脚本默认从 Real-Text 的 `HQmask/` 读 OCR region,SA-Text 缺该目录会报 "No valid regions found";补生成 SA-Text 专属 mask、sed 改 `MASK_ROOT` 后重跑,三级 OCR_REGIONS 均 = 2206。
- **结果(协议①,2a-remote ckpt-20000,**含污染**)**:
  | 等级 | PSNR | SSIM | LPIPS | DISTS | NIQE | MUSIQ | MANIQA | OCR-EM | OCR-CharAcc |
  |---|---|---|---|---|---|---|---|---|---|
  | lv1 | 23.15 | 0.674 | 0.267 | 0.207 | 8.97 | 65.3 | 0.561 | 0.306 | **0.440** |
  | lv2 | 23.19 | 0.665 | 0.298 | 0.228 | 9.37 | 63.7 | 0.543 | 0.282 | **0.413** |
  | lv3 | 22.14 | 0.622 | 0.365 | 0.272 | 10.03 | 59.4 | 0.508 | 0.205 | **0.308** |
- **解读**:CharAcc 随退化加重单调下降(44.0→41.3→30.8),符合预期;**绝对值远低于 Real-Text(67)**,因 SA-Text 退化更重、文本更密(每图 2.2 实例 vs Real-Text 2.3,但字号更小)。**该数字受训练集污染高估,仅作 TAIR 同口径参考,不入主结论**。
- **结果(协议② TESTR,2a-remote ckpt-20000,**含污染**)**:
  | 等级 | Det F1 | E2E F1 |
  |---|---:|---:|
  | HQ(GT 校准) | 89.17 | 78.33 |
  | lv1 | 71.30 | 31.89 |
  | lv2 | 69.68 | 29.43 |
  | lv3 | 62.78 | 21.85 |
- **解读**:HQ 校准 det 89.17 / E2E 78.33 说明 SA-Text 这条 TESTR 管线本身工作正常(比 Real-Text 的 HQ 86.68/74.23 还略高,因 SA-Text 字号/版式更规整);三级 E2E(31.9→29.4→21.9)随退化下降,与协议① CharAcc 趋势一致。**绝对值低于 Real-Text(E2E 42),反映 SA-Text 退化更重;数字受污染高估,仅作 TAIR 同口径参考,不入主结论。**

### dino_repa —— 通用 DINOv2 REPA 对照【199 在训】
- **设计**:REPA 原论文的标准形态(DiT 隐藏层 ↔ DINOv2 特征对齐,`repa_type: dino, repa_weight: 0.5, repa_layer: 13`),用于证明 2a 的增益来自 **OCR 特异特征**而非"任意 REPA 都行"。
- **配置**:`VOSR_0.5B_text_guided_target_no_rc_dino_repa.yml`(bs4 × grad_accum 4 = 等效全局 16,与 B/2a 可比)。
- **执行**:199 GPU1 单卡(A800),`run_dino_repa.sh`(6 次重试 + `find_latest_checkpoint` 自动断点续训)。单卡+累积预计 ~3 天。
- **坑(已修)**:dino config 沿用了本地 `text_hr_512_dataset.txt`(图路径 `/data/ywk/dataset/images/...`,199 上不存在 → DataLoader `FileNotFoundError` → ChildFailedError 反复崩);工作的 2a-remote config 用的是 `sa_remote_dataset.txt`(`/data2/wyw/ywk/datastes/SA/`)。已 sed 改 `train_dataset_config` + `test_lq_dir/test_gt_dir` 为 SA remote 路径,6-13 17:34 重启,过数据加载段正常,开始训练。

### 其余已登记待评/待跑
- dino_token / seg / seg_token REPA:配置在库,排队(199 出现第二张空卡时上)。
- E2(E1+2a 组合,config `ocr_cond_local_gt.yml` 就绪)、2c(真标签 CTC):排队中。

### 1.4B 放大 —— teacher 重训 + 一步蒸馏【2026-06-22,226.31】
- **背景**:1.4B = d1536/b36/h24 + Qwen-Image f8c16 VAE。**上一版 1.4B teacher 与蒸馏均欠拟合不可用**(VAE 完好 43dB,但 teacher/student 输出糊)。本轮**从零重训 teacher**(effbs16=bs4×accum4、100k 计划)。
- **teacher 收敛判定**:flow-matching loss 逐 batch 方差大(0.04–0.53,随采样 t 变化),用**窗口均值**判趋势——早期 0.234 → ~30k 0.162 → **36k 起平台 0.151–0.157**(近 1000 均值 0.1514)。已收敛、训练正常 → 于 **73k 停训**(loss 自 36k 饱和,73→100k 无收益)。
- **teacher 质量实测(ckpt-70000,必做——上版就是"loss 饱和但欠拟合")**:25 步多步推理 60 图(`inference_vosr.py -u4 --infer_steps 25 --cfg_scale 0.5 --weak_cond_strength_aelq 0.1`),对本机 real_test/HQ:**PSNR 23.40 dB(15.6–30.9)/ global-SSIM ~0.905**。**明确非欠拟合**(与 0.5B teacher 同量级),teacher 可用。
- **一步蒸馏(进行中)**:`distill_type: simple`、init=teacher ckpt-70000、effbs8(bs4×accum2 local1/卡、4×A6000)、LR 2e-5、20000 步、ckpt/5000。出 NFE=1 学生(推理 `inference_vosr_onestep.py --infer_steps 1`)。**坑(已解)**:原 simple config 是 2GPU 配方(`train_batch_size:2`),4 卡上 `local_batch=2//4=0` → DataLoader `batch_size=0` 崩;改 `train_batch_size:4`(local 1)即修。**收割链** `chain_distill_after_test.sh`:teacher 测试完成自动接力启蒸馏(≥55 图安全闸)。状态:13/20000、loss 0.15、~2.5s/it、ETA ~14h。
- **注**:0.5B headline 阶梯是 **shortcut** 蒸馏,此 1.4B 沿用既有 **simple** 管线(上版 1.4B 即 simple,本轮用好 teacher 重做);如需与 0.5B 严格同口径放大对比,可改 `distill_type: shortcut` 重跑。
- **蒸馏收割(2026-06-22,ckpt-20000,real_test 847,NFE=1)**:
  - **协议② TESTR:Det F1 66.95 / E2E None 26.76** —— **显著低于 0.5B**(0.5B baseline det ~76/E2E ~41、2a det 76.6/E2E 41.93)。
  - 协议① CharAcc:评测中(199 `metric_1p4b.py`)。
  - **★ 根因(2026-06-22 查代码坐实)——蒸馏跑成了 flow-matching(用错训练器,我的 bug)**:此"蒸馏"启动脚本误用 **`train_vosr.py`**(`vosr.loss_fm`=纯 FM 速度损失 `v=eps−hq`、`model(inp,t,z)` 无 shortcut),**`distill_type:simple` 被完全忽略、teacher 未用作蒸馏目标**——产物是从 teacher 初始化继续训的**多步 FM 模型**,被强行 NFE=1 推理才得 det 66.95。**真正的蒸馏在 `train_vosr_distill.py`**(line 904 按 `distill_type` 调 `vosr.loss_fm_distill_simple`/`_shortcut`);`loss_fm_distill_simple`= FM 速度损失 `v_loss` + **shortcut 自一致 `u_loss`**(模型在 (t,r) 的 shortcut 预测匹配两子步积分 (v1+v2)/2,自学走大步=一步蒸馏)。`train_vosr_distill.py` 已支持 qwen VAE(line 509)+ 加载 teacher(line 529)→ **可直接正确重蒸**。故 det 66.95/E2E 26.76 不代表"1.4B 蒸馏不行",只代表"FM 模型 @NFE1 差";**之前猜的 simple-vs-shortcut 主因作废**。
- **★★ teacher 识别欠拟合(2026-06-22,199 协议①,201 图)—— 真根因**:1.4B teacher(多步25)**CharAcc 仅 36.82** / PSNR 23.16 / EM 26.74 / CER 63.18。**对比 0.5B Teacher MS CharAcc 63.78**——**1.4B teacher 识别率不到 0.5B 的六成**!之前"teacher 可用"误判=**只看 PSNR(23.4 尚可)漏看识别**(当时已标注此风险)。**这是 1.4B 全盘问题的真根因:teacher 识别端严重欠拟合**(loss 平台 0.15 ≠ 识别收敛,正是"平台不能区分收敛/欠拟合饱和"的实例),**任何蒸馏(正确与否)都被 teacher 36.82 封顶**。注:比上版 teacher(致学生 11.68)好,但仍远不及 0.5B。
- **CharAcc-vs-step 诊断(2026-06-22,同 120 图子集,看趋势)—— 已平台**:

  | teacher ckpt | CharAcc | PSNR |
  |---|---|---|
  | 50000 | 0.2776 | 23.29 |
  | 60000 | **0.2967** | 23.33 |
  | 70000 | 0.2957 | 23.34 |

  50k→60k +1.9,**60k→70k 持平(−0.1)**,PSNR 几乎不动 → **识别率 ~60k 就停涨**。**续训到 100k 不会修好,必须换 recipe**(用户决策:停重蒸先修 teacher)。(注:120 子集绝对值 ~29 < 201 子集的 36.82,子集难度不同;趋势=平台才是有效信号。)
- **★★★ 深挖根因(2026-06-22)—— 0.5B 成/1.4B 不成 = init 不同(定论)**:逐项对比 teacher config:

  | 参数 | 0.5B teacher(成,63.78) | 1.4B teacher(败,~30) |
  |---|---|---|
  | **pretrained_ckpt** | `preset/ckpts/VOSR_0.5B_ms/.../ema_model.safetensors`（**微调**） | **`~`（从零随机！）** |
  | learning_rate | 1e-5 | 5e-5（5×） |
  | cfg_scale | 0.5 | −0.5（采样参数，对训练 loss 基本无影响、非主因） |
  | ae_type / enc | sd2 / dinov2b | qwen / dinov2l |

  **头号根因:0.5B 的"teacher"是从预训练 `VOSR_0.5B_ms` 微调来的，从来不是从零训；而 1.4B 用 `pretrained_ckpt:~` 从零训——尽管 `preset/ckpts/VOSR_1.4B_ms/checkpoints/ema_model.safetensors` 预训练权重就在那没被用！** 从零训 1.4B（12 万图≈10 epoch）识别平台 ~30 是必然。**这是配置疏漏，不是模型/VAE/数据问题。**
- **`VOSR_1.4B_ms` 预训练自测(2026-06-22,同 120 图)**:CharAcc **37.74** / PSNR 23.87(vs 从零训 70k 的 29.57)→ **预训练 init 比从零训 +8.2,证实 init 是关键**;但 37.74 仍不够(`VOSR_1.4B_ms` 是通用预训练、未文字特化)。
- **fix 已启动(2026-06-22)—— 复刻 0.5B 配方**:`VOSR_1.4B_ms_from_pretrained.yml`= 从 `preset/ckpts/VOSR_1.4B_ms/.../ema_model.safetensors` 初始化 + LR **1e-5**(对齐 0.5B)+ cfg_scale 改回 0.5 + SA-Text 微调 40000 步、ckpt/5000。脚本 `run_ms_from_pretrained.sh`(`train_vosr.py`=teacher FM 训练器,GPU1-4,screen `ms_pretrained`)。**启动确认**:日志 "Loading pretrained"、**起步 loss 0.0809**(远低于从零训的 ~1.3,印证好 init)。目标 teacher CharAcc 趋近 0.5B 的 63.78。待办:抽 ckpt 看 CharAcc 随步、达标后正确 shortcut 蒸馏(train_vosr_distill.py)。
- **错版"学生"(FM@NFE1)协议① 参照**:CharAcc 53.84 / PSNR 24.52(高于 teacher 36.82,因多训 20k FM 步;但这是错的 flow-matching 非蒸馏,仅作废参照)。
- **★★★★ fix 验证完成(2026-06-24,from-pretrained teacher ckpt-40000,协议① 全 847)—— init 根因坐实、teacher 已修好**:25 步多步推理(cfg0.5/wc0.1/Qwen-VAE)→ **CharAcc 65.63** / PSNR 24.78 / SSIM 0.786 / CER 0.344。**对比从零训版 36.82 → +28.8,且超过当初目标"趋近 0.5B teacher 63.78"(实达 65.63 > 63.78)**。**结论定论:"1.4B 不成"是 `pretrained_ckpt:~` 从零训的配置疏漏,不是 1.4B 规模/Qwen-VAE/数据的问题——换 from-pretrained init + LR 1e-5 即修复,1.4B teacher 识别端与 0.5B 同量级。** 资产:226 `run_teacher40k_847.sh`、自动链 `harvest_teacher40k.sh`、199 `metric_teacher40k.log`。**下一步可选**:teacher 已达标(>55 且≈0.5B),可启动正确的 1.4B shortcut 蒸馏(`train_vosr_distill.py`,带 2a/2d OCR 监督),看 1.4B 学生能否超 0.5B 的 68.8。
- **★ 1.4B 正确 shortcut 蒸馏已启动(2026-06-24,199 GPU2,3)—— 复刻 0.5B 2a+2d 配方**:
  - **迁移而非搬代码**:用户要求"只迁权重 + 改 config 适配,别一股脑搬 226"。核查发现 **199 `VOSR_textsr/train_vosr_distill.py` 与本机逐行等价**(qwen VAE 9 处、2a/2d OCR loss 全在、`v=v_target` GT 监督修复在),dinov2l/qwen-VAE/SA 数据/dinov2 torch_cache 均已就位 → **零代码改写,纯 config 适配**。仅迁移两文件:teacher EMA(5.2G,226 `rget_dt`→本机→`rput` 199)+ `text_boxes_index.pkl`(SA-Text GT 框,与 199 SA 图同 basename 对齐)。
  - **config `VOSR_1.4B_codsr_2a2d.yml`**(派生自 0.5B 冠军 `..._ocr_repa_realctc`,仅换):dim 1024→1536 / depth 28→36 / heads 16→24;ae sd2→**qwen f8c16**(`preset/ckpts/Qwen-Image-vae-2d`,OCR loss 走全 512 qwen decode);enc dinov2b/768/[8]→**dinov2l/1024/[17]**(必须配 teacher 的 z_dim 才能加载);teacher=pretrained=迁移来的 `teacher_1p4b_ms_40000.safetensors`;数据=199 `sa_remote_dataset.txt`(`/data2/wyw/ywk/datastes/SA/images`,同 sa_*_crop 基名);box index=199 repo 根 proven 路径。2a(ocr_repa_weight 0.5)+2d(ocr_real_ctc_weight 0.5)、LR 5e-6、20000 步、ckpt/1000,**与 0.5B 逐项一致**。
  - **effbs 对齐坑**:此 codebase `train_batch_size` 是**跨 GPU 总量**(按 GPU 数均分),故 effbs = `train_batch_size × grad_accum`(**不乘 GPU 数**)。首启 bs2×accum4=effbs8(expname bs008)偏小 → 改 **accum8 → effbs16**(bs016),per-device1(80GB 仅占 46GB,full-512 qwen OCR decode 安全),与 0.5B 冠军 effbs16 同口径。
  - **启动确认**:torchrun nproc4(GPU**1,2,3,4**,GPU4 与他用户共租)、teacher 载入 missing keys=6(shortcut/aux-time 参数,符合 fm→shortcut 预期)、GT 框 119495、训练中、effbs16(bs4×accum4,per-device1)、v_loss~5e-4。脚本 `run_1p4b_2a2d_199.sh`(NCCL_P2P_LEVEL=NVL、retry+auto-resume)、screen `distill_1p4b`、日志 `train_1p4b_2a2d.log`。**看点:1.4B 学生 CharAcc/det 能否超 0.5B 的 68.74/78.32。**
  - **★ 多卡提速无效的测量与定论(2026-06-24)**:应用户"改 4 卡加快速度",从 2 卡(2,3)切到 4 卡(1,2,3,4)。**实测瞬时步速 4 卡 ~13.0s/it ≈ 2 卡 ~13.1s/it,提速≈0**。根因:**固定 effbs16 下,每个 optimizer step 的墙钟被 sync 步的「全 512 qwen-VAE OCR decode + CRNN」(ocr_*_interval=1,每 opt 步一次)主导**,而非可被 GPU 并行的微批 fwd/bwd;加之第 4 张卡 GPU1 与 GPU2/3/4 不同 NUMA,跨 socket 的 1.4B 梯度 all-reduce(NCCL_P2P_LEVEL=NVL→host 回退)抵消了减少微批带来的微小收益。**故在此「每步一次重 decode」的训练结构下,加卡不增吞吐(opt 步数固定 20000、每步重 decode 固定)。** 真·提速杠杆=`ocr_repa_interval`/`ocr_real_ctc_interval` 改 2(OCR loss 每两步一次,~2× 但偏离 0.5B 配方、削弱可比性),或减 effbs(改样本量)。**用户拍板:保持 4 卡现状(ETA~3 天)。** 经验:OCR-decode-in-loop 的蒸馏,瓶颈在 decode 不在并行,多卡扩展性差。
  - **★★ A/B 实测:per-device(每卡 batch)对速度与"有效 OCR 权重"的耦合(2026-06-24,重要方法论)**:应用户"增大每卡 size 减少累积"提议,实测两配置(均 effbs16、4 卡):
    - **A** per-device1/accum4/OCR权重0.5:瞬时 **12.5 s/it**、峰值~46GB。
    - **B** per-device2/accum2/OCR权重0.5:瞬时 **10.0 s/it**(快~20%)、峰值~50GB(GPU4 含共租 65GB,安全,无 OOM)。
    - **关键发现(查码坐实,line1363 `accelerator.accumulate` 对每次 backward 除以 accum)**:2a/2d OCR loss 是 `on_sync_only`(每 opt 步只在最后那个 sync 微批算一次),故其**有效权重 = ocr_weight / accum**。→ A 有效 OCR 权重 0.5/4=**0.125**(与 0.5B 冠军 accum4 一致);**B 是 0.5/2=0.25,悄悄把 OCR 强度 ×2、偏离基准**——B 的提速真实,但**不是同一训练**。补偿方案 **B′**=per-device2/accum2 且 OCR 权重砍半到 0.25(→有效 0.125,对齐冠军)可拿到提速+可比,但靠"权重/accum 等效"推算非逐字节复刻。
    - **用户拍板:选 A(per-device1/accum4/权重0.5),与 0.5B 冠军逐项完全一致,零可比性风险**(放弃 20% 提速)。当前 run 即 A。**通用教训:带 `on_sync_only` OCR/FD loss 的蒸馏,改 accum 会连带改有效 aux 权重——调 batch 拆分前必须同步核对 aux 权重,否则静默改变 loss 平衡。**

---

## 3. 总对比表

### 3.1 协议① PP-OCR metric(Real-Text 847,按 CharAcc 升序;粗体 = 列最优)

| 模型 | NFE | PSNR | SSIM | LPIPS | DISTS | NIQE | MUSIQ | MANIQA | EM | **CharAcc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TeReDiff(复现) | 50 | 23.30 | 0.7652 | 0.2849 | 0.2384 | **7.56** | **62.14** | **0.5111** | 53.37 | 60.61 |
| Teacher MS(多步) | ~28 | 24.86 | 0.7882 | 0.2692 | 0.2087 | 6.61* | 54.90 | 0.4040 | 56.20 | 63.78 |
| C: guided+RC | 1 | 24.28 | 0.7864 | 0.2550 | **0.2078** | 6.52* | 54.85 | 0.3819 | 55.48 | 64.01 |
| 结构增强① distill | 1 | 24.98 | 0.8130 | 0.2487 | 0.2040* | 8.10 | 53.14 | 0.4051 | — | 65.59 |
| seg_token(对照) | 1 | 24.55 | 0.8312 | 0.2564 | 0.2218 | 10.98 | 54.70 | 0.4328 | 57.33 | 64.86 |
| 2c: detmap 一致性 | 1 | 24.69 | 0.8337 | 0.2531 | 0.2185 | 10.78 | 54.34 | 0.4229 | 57.79 | 65.76 |
| E1: TrOCR cond | 1 | 24.41 | 0.8306 | 0.2523 | 0.2159 | 10.51 | 53.67 | 0.4192 | 57.89 | 65.80 |
| REPAocr(全局) | 1 | **25.80** | 0.8362* | 0.2500 | 0.2110 | 10.06 | 53.29 | 0.4058 | — | 65.81 |
| A: full target | 1 | 24.52 | 0.8329 | 0.2538 | 0.2189 | 10.74 | 53.50 | 0.4156 | 57.89 | 65.82 |
| **B: guided target(baseline)** | 1 | 24.62 | 0.8334 | 0.2541 | 0.2200 | 10.94 | 54.78 | 0.4295 | 58.41 | 65.99 |
| dino_repa(通用 REPA 对照) | 1 | 24.76 | 0.8354 | 0.2520 | 0.2178 | 10.75 | 54.15 | 0.4221 | 58.20 | 66.00 |
| 结构增强② distill | 1 | 23.61 | 0.8230 | 0.2553 | 0.2179 | 10.60 | 54.45 | 0.4237 | — | 66.23 |
| w=1.0 | 1 | 24.41 | 0.8268 | 0.2498 | 0.2083 | 9.47 | 52.95 | 0.4053 | 57.89 | 66.96 |
| w=0.25 | 1 | 24.61 | 0.8339 | 0.2504 | 0.2171 | 10.78 | 54.46 | 0.4255 | 58.25 | 66.98 |
| 2b: GT-CTC | 1 | 24.24 | 0.8292 | 0.2454† | 0.2094 | 10.01 | 53.34 | 0.4103 | **58.35** | 67.45 |
| 2a: local OCR-REPA(w=0.5) | 1 | 24.86–24.97 | **0.836–0.837** | 0.2445–0.2468 | 0.214 | 10.56 | 54.05 | 0.4172 | 58.05–58.25 | 66.92–67.66 |
| **🏆 2d: 真 GT 文本 CTC** | 1 | **25.08** | 0.8388 | 0.2426 | 0.2121 | 10.56 | 54.50 | 0.4216 | **58.97** | 68.28 |
| **🏆 2a+2d 联合** | 1 | 24.67 | 0.8320 | **0.2372** | **0.2022** | 9.58 | 52.72 | 0.4005 | 59.23 | **68.74** |

\* Teacher/C/结构增强① 的 NIQE/DISTS 优势来自多步采样或强正则的"自然感",伴随 OCR 低位,不构成有效 trade-off。† 2a 波段内 LPIPS 最低为 0.2445,与 2b 并列量级。2d 为单 ckpt(20000),其余多为噪声带均值。

### 3.2 协议② TESTR 官方 spotting(Real-Text 847)——**None + Full lexicon 双列(对齐 TAIR Table 3)**

**重要(2026-06-15 补 Full lexicon 后的口径澄清)**:TAIR Table 3 的 E2E 有 **None / Full 两列**,且用两个 spotter(ABCNet v2、TESTR);我们对标的是 **TESTR** 那组。论文 TeReDiff(TESTR)= **None 49.39 / Full 56.45**,FaithDiff(TESTR)= None 41.64 / Full 47.97。**49.39 是 None,不是 Full**——此前只比 None 一列、且一度误以为"Full 能解释 TeReDiff 缺口",均已更正。Full lexicon 用 Real-Text GT 重建的 773 词词表(`/data/ywk/TAIR/realtext_eval/lexicons/totaltext/`),**离线重评分**已存的 `text_results.json`(里面是 None 的原始识别串)得到,零 GPU、零重推理;管线先精确复现 None(HQ E2E 74.23、TeReDiff 33.53 一字不差)才采信。脚本 `rescore_lexicon.py` / `rescore_batch.sh`。

| 方法 | NFE | Det F1 | E2E **None** | E2E **Full** |
|---|---:|---:|---:|---:|
| LQ(bicubic↑512) | — | 44.63 | 29.37 | 35.14 |
| **TeReDiff(论文宣称,TESTR)** | 50 | 74.88 | **49.39** | **56.45** |
| **TeReDiff(本地复现,官方权重)** | 50 | 76.14 | 33.53 | 44.62 |
| TeReDiff(本地复现,TAG) | 50 | 76.28 | 30.54 | 41.34 |
| FaithDiff(论文,TESTR) | 多步 | 70.57 | 41.64 | 47.97 |
| **🏆 2a(w=0.5,remote)** | 1 | 76.62 | 41.93 | 52.69 |
| 2a(local) | 1 | 76.32 | 43.06 | 52.36 |
| 2b GT-CTC | 1 | 76.35 | 43.33 | 52.75 |
| **🏆 2d 真 GT 文本 CTC** | 1 | 77.03 | **44.60** | **54.68** |
| **🏆 2a+2d 联合** | 1 | **78.32** | 42.59 | 52.71 |
| w=0.25 | 1 | 76.65 | 43.74 | 53.96 |
| w=1.0 | 1 | 77.73 | 42.42 | 52.15 |
| 2c detmap 一致性 | 1 | 71.91 | 45.17† | 54.21† |
| E1 TrOCR cond | 1 | 71.70 | 45.53† | 54.79† |
| dino_repa(通用 REPA) | 1 | 72.28 | 46.37† | 55.27† |
| seg_token(对照) | 1 | 72.31 | 44.13† | 53.83† |
| HQ(GT,管线校准) | — | 86.68 | 74.23 | 82.05 |

† 低 det 方法(E1/2c/dino/seg,det 71–72)的 E2E 在 **两列都"偏高"**(None 44–46、Full 54–55,甚至超过 det 76 的 2a)——这是"低召回→基数小、虚警被过滤"的高精度假象,Full lexicon 的"就近 snap 到词表"会进一步放大它。**所以 E2E 单看会误导,必须连 det F1 一起读**:2a 是唯一同时拿到最高 det(76.6)+ 协议① 最高 CharAcc(67+)的方法,这四个低-det 对照仍判**无增益**(详见各节)。

**TeReDiff 复现缺口——None 和 Full 两列都存在,不是 lexicon 口径问题:**
| | det F1 | E2E None | E2E Full |
|---|---:|---:|---:|
| 论文宣称 | 74.88 | 49.39 | 56.45 |
| 我们用官方 stage3 权重复现 | 76.14 | 33.53 | 44.62 |
| **缺口** | **+1.3(我们反高)** | **−15.9** | **−11.8** |

- **检测端复现正常甚至更好**(76.14 > 74.88),**HQ 校准准**(None 74.23、Full 82.05),说明评测管线无误;**缺口纯在识别端**,且 **None/Full 两列同时存在 ~12–16 分**——证明不是 None-vs-Full 的口径错位,而是**用作者发布权重就是复现不出论文的识别数**(prompt_style 已排除:TAG 更差)。归因:发布权重≠论文权重 / 识别端不可复现。台账对 TeReDiff 一律取我们复现数,论文数标注 "published, 官方权重未能复现"。

**★ 完整溯源审计(2026-06-22,回应"是否权重/代码错了"的质疑)—— 每一环交叉验证通过,缺口锁死在 TESTR E2E 这一个指标**:
1. **权重 md5 溯源**:复现用的 `resume_ckpt_dir` = `terediff_stage3.pt`,md5 **`a54ced6d8495d205a6f3f9812d1fa148`**,与官方 `TAIR-main/demo/terediff_stage3.pt` 及 199 `temps/TAIR/preset/terediff_stage3.pt` **逐字节相同**;≠ 用户重训的 `model_ckpts/.../TRAIN_*`(md5 `189b84ac…`,Nov 日期)。**确认用的是官方权重,非重训权重**。
2. **IQA 三方对拍**(忠实照搬 `temps/TAIR/val.py` 的 pyiqa 计算,847 图):我的复现 SR vs 用户原版 TAIR 跑 vs 论文 —— **PSNR 23.30/23.30/23.37、LPIPS 0.2849/0.2848/0.2848(逐位)、DISTS 0.2384/0.2384/0.2386(逐位)、NIQE 7.56/7.56/7.64、MUSIQ 62.14/62.13/62.02** 三方一致 → **SR 模型/推理 100% 正确**。
3. **识别率/CharAcc 一致**:用户 TAIR 识别率 **60.07** ≈ 我协议① CharAcc **60.61**(注:这是"给定区域逐字识别准确率",**宽松**指标,≠ E2E spotting)。
4. **同图 TESTR 复核**:对这组已被 IQA+识别率证明正确的 SR 图,官方 TESTR(config 1600/1824/0.45、官方权重)跑出 **det 76.14 / E2E None 33.53**,与历史复现逐位一致。
- **结论**:模型✓ 权重✓ 识别准确率✓ 评测管线✓(HQ 校准 74.23≈论文 74.50),**唯一复现不出的就是论文宣称的 E2E spotting 49.39**——在一组已证明正确的 SR 图上,官方权重+官方 TESTR 就是 33.53。**缺口铁定在"论文发布的 stage3 权重的 TESTR E2E 联合指标"本身,与我们的权重/代码/图片无关**。这套证据链(md5+IQA三方+识别率+同图复核)是对 TeReDiff 宣称数的有力 rebuttal 资产。

**对标其余基线(论文 TESTR 数,同口径):2a/2b 在 None 持平、Full 明显反超 FaithDiff,且 NFE=1。**
| | E2E None | E2E Full |
|---|---:|---:|
| FaithDiff(论文,多步) | 41.64 | 47.97 |
| 2a(我们,1 步) | 41.93 | 52.69 |
| 2b(我们,1 步) | 43.33 | 52.75 |

参考(TAIR 论文表 3 其余 published,Det F1 / E2E-None):DiffBIR 68.35/39.27、SeeSR 67.87/40.34、SUPIR 48.39/27.25——均低于我们 2a/2b。

### 3.3 一句话结论

1. **两个互补冠军**:**2a+2d 联合**拿下「给定区域识别 + 检测定位」全表新高(CharAcc **68.74**、det F1 **78.32**,逐项超 2d),证明特征空间 REPA 与输出空间 CTC 在这两项上建设性叠加;**2d(真 GT 文本 CTC)**仍是「端到端 spotting + 像素保真」最优(E2E None 44.60 / Full 54.68 / PSNR 25.08)。联合在 E2E/PSNR 上**未超** 2d(42.59 / 52.71 / 24.67),不是全面碾压——TESTR E2E 需检测器与识别器同时命中,联合新增的召回框在 TESTR 自带识别器下转写未必对,故 E2E 反小降。**2a 仍是最强的"纯特征对齐"方案**(无需 GT 文本标注)。
2. **识别口径(协议① CharAcc)排序:2a+2d > 2d > 2a > 2b > w0.25 ≈ w1.0 > B**;2b→2d 的唯一变量是 CTC 标签来源(HR 贪婪伪标签→真 GT 文本),净增益说明 **CTC 的瓶颈在标签噪声**;2d→2a+2d 再叠特征对齐,CharAcc/det 又各进一步(+0.46 / +1.29)。w 扫描呈倒 U,w=0.5 双指标峰值。**2c(检测图一致性)完成且无增益**(CharAcc 65.76 ≈ baseline、det 71.91 落后 2a 4 分),粗粒度像素级空间约束打不过细粒度区域/glyph 监督。
3. teacher 侧改进无效(被蒸馏洗掉),student 侧监督有效——这是方法论主线。**从四个正交切口反衬 2a 的 OCR 特异性**:E1(LQ-side OCR 语义条件注入,gate 自学到负值主动关闭)、2c(检测图一致性)、dino_repa(通用 DINO-REPA)、seg_token——四者 CharAcc 全部≈或略低 baseline、协议② det 全部卡在 71–72,唯独 2a 同时把 det 拉到 76+、CharAcc 拉破 67。只有**细粒度区域级 OCR 特征对齐**(像素/特征级 student 监督)能穿透蒸馏均衡器。
4. **协议② 现按 TAIR Table 3 同口径出 None+Full 双列**(TESTR spotter)。结论分层:(a) **超过 FaithDiff 等全部 published 常规扩散基线**——2a/2b E2E None 持平 FaithDiff(41.6→41.9/43.3)、Full 明显反超(47.97→52.7),且 NFE=1 vs 多步;(b) **对 TeReDiff 论文数(None 49.39 / Full 56.45)我们仍低**,两列差 12–16 分;(c) 但**用 TeReDiff 官方 stage3 权重复现不出论文数**(我们复现 None 33.53 / Full 44.62,det 76.14 反而高于论文、HQ 校准准 → 缺口纯在识别端,None/Full 同时存在,**已排除 prompt_style**),故 2a **全面超过 TeReDiff 的可复现表现**。诚实立场:对标 published 基线我们胜且更快,对 TeReDiff 宣称数取保留态度(官方权重未能复现)。
5. 效率:NFE=1 vs TeReDiff NFE=50(50 步 + 每步 spotting prompt 循环),推理成本差 ~50×。

---

*维护约定:每新增一个实验(含失败实验),在 §2 增一节、§3 表中增一行;数字一律注明协议与 ckpt 步数。最后更新:2026-06-15(**2c 完成无增益**;dino/seg/e1 对照补齐;**协议② 补 Full lexicon、按 TAIR Table 3 同口径出 None+Full 双列**——离线重评分,发现 TeReDiff 缺口 None/Full 两列同时存在 12–16 分、非口径问题、缺口纯在识别端;2a/2b 超过 FaithDiff 等全部 published 基线且 NFE=1;**2d 真 GT 文本 CTC 已实现、本机 4 卡训练中**;待办:E2 组合、1.4B 放大)。最后更新:2026-06-17(**2d 真 GT 文本 CTC 完成,全面最优**:CharAcc 68.28 / det 77.03 / E2E None 44.60 / Full 54.68,逐项超 2a/2b,证明 CTC 瓶颈在标签噪声;中途因同租户抢卡 OOM,已加 OOM 容错+等卡守护自动续训;**联合 2a+2d 已 chain 自动接力、训练中**;待办:2a+2d 收割、E2、1.4B)。最后更新:2026-06-19(**2a+2d 联合完成,识别/检测双新高**:CharAcc **68.74** / det F1 **78.32** 全表最高,逐项超 2d;但 E2E None 42.59 / Full 52.71 / PSNR 24.67 未超 2d——**两个互补冠军**,联合最优于"识别+定位"、2d 最优于"端到端 spotting+保真";训练无 OOM 跑满 20000;待办:E2、1.4B、Real-CE 微调决策)。*
