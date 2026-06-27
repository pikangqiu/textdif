# 任务 03 —— Real-CE：按领域标准"混入训练"(非顺序微调) + 两 setting + 一致性表

> **2026-06-22 重写**：查实领域做法后改向。最新最强(TADiSR/TIGER/TEXTS-Diff)**全是把 Real-CE train 混进自己的合成/mask 数据联合训练**(并把基线也在 Real-CE 微调),无一用"顺序微调"。DiffTSR 则纯零样本评 generalization。详 `../related_work_scan.md` §6。
> 关键事实：**Real-CE 自带 GT box + 文字转录**(train 1935对/23,547行)→ 我们 2a-GT 框 OCR-REPA + 2d GT-text CTC **完全可用**,是主场优势。
> 风险定位：Real-CE 零样本 PSNR 17<输入19.65 像退化;此前顺序微调=弱结果。详 `../story_evidence_audit.md`。

## Part 0 —— 前置：用 TAIR 数据管道为 Real-CE 构建标注 + 建索引（阻断点）
- 现状：`/data/ywk/datasets/Real-CE/train/{13mm,52mm}` 各 645 对；Real-CE 自带逐行 GBK 标注**不够具体/不好用** → 用 TAIR 管道**自动构建**。
- **管道(已 clone 本机 `/data/ywk/SA-Text_Dataset/`)**：正确仓 `paulcho98/SA-Text_Dataset`。11 阶段:`start`(bridge_stage1全图检测)→`cropping`(按检测裁512,sliding/adaptive)→`bridge_stage2`(crop重检测)→`vlm1`(OVIS2)→`vlm2`(Qwen2.5-VL)→过滤/取一致/模糊过滤→`final_formatting`→`dataset.json`(框+转录)。**自带从零检测,不依赖原标注,任意图像可跑。**
- **本机构建(选本机:远端禁改env,本机有外网+4×24GB)**：
  - env:`conda create -n dataset_curation python=3.10`;torch2.5/cu124;`transformers==4.51.3`+qwen_vl_utils+accelerate;`flash-attn --no-build-isolation`;`Bridging-Text-Spotting/detectron2` 与本体 `python setup.py build develop`(setuptools==59.5.0)。
  - 权重:Bridge `Bridge_tt.pth`+DiG(Google Drive,gdown);VLM `AIDC-AI/Ovis2-8B`(~16-18GB bf16)+`Qwen/Qwen2.5-VL-7B-Instruct`(~16GB,需flash-attn2),HF 自动下。**两VLM顺序跑,峰值单卡一个→24GB够**。
  - 改 `dataset_curation/config.yaml`:`sa1b_base_dir`/`sa1b_subfolder`指向 Real-CE 52mm;`bridge_repo_dir`/`bridge_weights_file`;输出目录。run:`python dataset_curation/main_pipeline.py --config ... --sa1b_subfolder <realce> --output_suffix _realce`。
  - **⚠️ 中文检测风险(Bridge config=TotalText英文)**:先在 5-10 张 Real-CE 验证全图检测召回。召回OK→全量645跑;**召回差→hybrid:Real-CE自带框裁crop+喂vlm阶段schema,`--start_from vlm1_recognition` 绕开英文检测器,只用Qwen/OVIS中文识别**(需Real-CE train框)。
- **关键认知**:我们 2d 的"GT text"(`text_boxes_index.pkl` texts)**本就是此管道在 SA 上产的标签,非人工GT** → 在 Real-CE 用同管道**标签同源、无故事降级、无方法不一致**。2d 贡献=双VLM交叉验证文本>单识别器伪标签(2b),Real-CE 照样成立。**写作把"GT text"诚实定义为"dual-VLM 交叉验证标注"**(SA 上亦然)。
- **在 HR(52mm) 上跑管道**(标注清晰图,TAIR 即如此);配准的 13mm 用同 crop/box 坐标 replay(复用 `_apply_same_crop`)→ 得 LR-HR 裁剪对+框+转录。
- **⚠️ 中文检测风险**:Bridge Spotter 多在英文(TotalText系)训,Real-CE 中文密集长文本召回或掉。缓解(推荐):**框用 Real-CE 自带行级框(人工,可靠),VLM 只做识别/转录**,绕开英文检测器弱项;或先验 Bridge Spotter 中文检测质量再定。
- 建索引(仿 `preset/text_boxes_index.pkl`/`restoration_dataset.json`)供 2a-GT/2d 取框取文本;滤错配对(参考 TADiSR/TIGER 1935→337)。

## Part A —— S2 域内：把 Real-CE train 混入蒸馏训练（正解，替代顺序微调）
- **做法**：训练集 = SA 合成 + **Real-CE train(带框/转录)**,对 Real-CE 样本照常上 2a-GT + 2d 监督。按行级框裁 512 区域喂入(与 SA 裁剪管线一致)。
- **配比**：Real-CE 仅 337~645 对 vs SA 119k → **上采样/加权 Real-CE**,防被淹没(扫 2-3 个配比)。
- **init**：从现有学生 ckpt(B/2a-GT/2a+2d) 继续混训,或并入主蒸馏。
- **评测**：`scripts/eval_realce.py`(valid_list 260,官方 CRNN ACC/NED + PSNR/SSIM/LPIPS)。
- **判据**：识别接近/达 RRDB 0.3093 且 PSNR 回到可辩护(不离谱低于输入)→ S2 可作正文"域内竞争"正面结果。
- **公平性**：S2 基线也应 train-on-Real-CE → 诚实主对标 **RRDB(可引)** + 我们;TADiSR/TIGER 同设定但不开源,仅点名。
- 废弃:旧 `VOSR_0.5B_realce_2a_realpair_*.yml` 顺序微调路线(留作消融对照"顺序微调 vs 混训",非主结果)。

## Part B —— S1 泛化：零样本一致性表（最稳的牌，必做）
- 训练仅 SA → 测 Real-CE 零样本,对标 DiffTSR 式 generalization。
- 用 **PP-OCR CharAcc 同一内轴**,把 B/2a/2d/2a+2d 在 **847** 与 **Real-CE** 的识别排序并列,证 **B<2a<2d<2a+2d 跨域保持**(已验证一致)。
- 微调/混训行单列,勿与零样本混。

## Part C —— 可选：Real-CE 补 1-2 现代扩散对照点
- Real-CE 论文(2023)表无扩散方法。**若**要 Real-CE 表放扩散对照:跑 **OSEDiff**(199 已有 env,~半小时)±**SeeSR**,同协议。先问用户。

## 工作量
- Part 0：取标注+建索引+滤错配 ~半天-1天(取决于标注获取)。
- Part A：混训(数小时,等卡)+配比扫 + 评测 ~1-2 天。
- Part B：纯评测/汇总 ~半天。
- Part C：OSEDiff ~半天。

## 结果（回填）
> train 标注获取/索引: ______
> S2 混训 ACC/NED/PSNR / 配比: ______ 是否救回观感: ______
> S1 一致性表: ______
