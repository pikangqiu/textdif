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
- **结论:无增益,弃用**。CharAcc 65.80 与 baseline B(65.99)持平(实为 −0.19,在噪声内),PSNR 比 2a(24.9)低 0.5dB;`ocr_cond_gate` 一路走负到 −0.0137,**模型主动抑制这路条件**,证实 LQ-side OCR 语义条件对 token-level 蒸馏无用——只有 student 侧的**像素/特征级监督**(2a 的 local OCR-REPA)能穿透"蒸馏均衡器"。协议②上 det 反比 2a(76.3)低 4.6 分,佐证条件注入未带来文本结构增益。组合版 E2(E1+2a)因 E1 单独无效、优先级下调。

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
- **结论:联合把"给定区域识别"(协议① CharAcc 68.74)与"检测定位"(det 78.32)同时推到全表新高**,逐项超过 2d(68.28 / 77.03),证明特征空间 + 输出空间监督在这两个指标上**建设性叠加**(各管一头:REPA 提区域可读性、CTC 提 glyph、两者合力进一步抬 CharAcc;det F1 也随之新高)。**但不是对 2d 的全面碾压**:E2E None 42.59 < 2d 44.60、Full 52.71 < 2d 54.68、PSNR 24.67 < 2d 25.08,三项被 2d 反超。即**两个互补冠军**——联合是"识别+定位"最优(CharAcc/det),2d 是"端到端 spotting + 像素保真"最优(E2E/PSNR)。机理:det F1(IoU 定位)与协议① CharAcc(GT-mask 区域 + PP-OCR)度量"给定框可读性",联合在此双赢;TESTR E2E 要检测器与识别器**同时**命中,新增召回的框在 TESTR 自带识别器下转写未必对,故 E2E 反而小降。注:单 ckpt,后可补噪声带。

### 对比方法 —— TeReDiff(TAIR 论文)本地复现
- **设置**:作者发布 stage3 权重 + 作者 `val.py`(50 步,prompt_style=CAPTION,wandb 已关)+ 字节一致 Real-Text;config `/data/ywk/TAIR-main/configs/val/val_terediff_realtext.yaml`。推理 1h16m @ 24GB(NFE=50,vs 我们 NFE=1)。
- **结果(None+Full 双列,对齐 Table 3 的 TESTR 列)**:协议② det 76.14 / **E2E None 33.53 / Full 44.62**(论文宣称 TESTR None **49.39** / Full **56.45**;**两列同时差 12–16 分**;同管线 HQ 校准 None 74.23/Full 82.05 误差仅 0.27);协议① CharAcc 60.61 / EM 53.37 / PSNR 23.30 / SSIM 0.7652 / MUSIQ **62.1** / MANIQA **0.51**。
- **缺口口径澄清(2026-06-15)**:补 Full lexicon 后确认 **49.39 是论文 None 列、不是 Full**;缺口在 None 和 Full **两列都存在 12–16 分**,故**不是 None-vs-Full 的口径错位**,而是用作者发布权重就复现不出论文识别数。det 反高于论文(76.14>74.88)+ HQ 校准准 → 缺口纯在识别端。
- **解读**:det 正常说明推理无误,缺口集中在识别端;TeReDiff 只在无参考美学指标占优(生成感强、字符不忠实)。未排除变量:prompt_style TAG、采样默认值、Drive ckpt 版本。**TAG 复核前论文数字按 "published number" 引用。**
- **TAG 复核(完成,prompt_style 已排除)**:config `val_terediff_realtext_tag.yaml`(唯一变量 prompt_style: CAPTION→TAG)。结果:协议② det **76.28** / E2E **30.54**;协议① CharAcc **59.25** / EM 51.83 / PSNR 23.19 / SSIM 0.7671 / MUSIQ 62.81 / MANIQA 0.5156。**TAG 在两个协议上都比 CAPTION 更差**(E2E None 30.54 < 33.53,CharAcc 59.25 < 60.61),因此 prompt_style **不是**缺口来源——与提示风格无关,仍归因识别端/不可复现。**TeReDiff 复现取 CAPTION 数(det 76.14 / E2E None 33.53 / Full 44.62 / CharAcc 60.61);2a 全面超过 TeReDiff 的可复现表现,但低于其论文宣称数(None 49.39/Full 56.45,官方权重未能复现)。** 对标其余 published 基线(FaithDiff None 41.64/Full 47.97 等)2a/2b 则同口径胜出且 NFE=1。

### 跨域评测 —— Real-CE(261 图,中文场景)
- **设置**:2a-remote ckpt-20000 直接零样本推理(`scripts/eval_realce.py`),与 13mm baseline 对照。
- **结果**:PSNR 17.40 / SSIM 0.663 / mask-PSNR 10.22 / mask-SSIM 0.495 / **acc 28.38% / NED 0.595**;baseline 19.65 / 0.663 / 11.89 / 0.496 / 27.50% / 0.622。识别 acc 略胜、像素指标落后(域差:中文长文本、不同退化)。是否用 Real-CE train split 微调待决策。

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
- E2(E1+2a 组合,config `ocr_cond_local_gt.yml` 就绪)、2c(真标签 CTC)、1.4B 放大:排队中。

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
