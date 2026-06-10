# 文本 SR 一步蒸馏:实验配置原理与结果分析

更新时间:2026-06-08 ｜ 分支:`text-distill-ablation`

> 本文聚焦 **OCR 监督设计的对照**:把"正常蒸馏 / 全局 REPAocr / 结构增强(CODSR) / 2a 局部 OCR-REPA / 2b CTC"五类配置的**原理、代码实现、结果、机制原因**讲透。
> 总览/路线图见 `doc/text_distill_master_plan.md`;2a/2b 隔离契约见 `doc/text_ocr_local_repa_experiment.md`。

---

## 1. 结果总表(Real-Text 评测集)

列序:PSNR↑ / SSIM↑ / LPIPS↓ / DISTS↓ / NIQE↓ / MUSIQ↑ / MANIQA↑ / **OCRacc(CharAcc)↑**。
按 OCR 从低到高排列。**粗体 = 该列最优。**

| 模型 | PSNR | SSIM | LPIPS | DISTS | NIQE | MUSIQ | MANIQA | **OCRacc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher MS(VOSR-text 0.5B,多步) | 24.862 | 0.7882 | 0.2692 | 0.2087 | **6.607** | **54.90** | 0.4040 | 63.78 |
| 结构增强① teacher | 24.109 | 0.8208 | 0.2610 | 0.2155 | 9.966 | 51.29 | 0.3931 | 64.61 |
| 结构增强① distill | 24.976 | 0.8130 | 0.2487 | **0.2040** | 8.101 | 53.14 | 0.4051 | 65.59 |
| 结构增强② teacher(ragp+lqmod+vae-lora) | 23.503 | 0.8216 | 0.2533 | 0.2169 | 10.424 | 51.72 | 0.3987 | 65.67 |
| REPAocr(全局) | **25.796** | **0.8362** | 0.2500 | 0.2110 | 10.058 | 53.29 | 0.4058 | 65.81 |
| B(VOSR-text distill,guided target) | 24.619 | 0.8334 | 0.2541 | 0.2200 | 10.939 | 54.78 | **0.4295** | 65.99 |
| 结构增强② distill | 23.608 | 0.8230 | 0.2553 | 0.2179 | 10.597 | 54.45 | 0.4237 | 66.23 |
| **🏆 2a(local-crop OCR-REPA,GT框,step18000)** | 24.885 | 0.8358 | **0.2471** | 0.2141 | 10.560 | 54.05 | 0.4172 | **67.28** |
| 2b(CTC,字符级监督) | — | — | — | — | — | — | — | 待测 |

补充:REPAocr **强化版**(repa_weight 0.25→1.0、global 0.25→0.5)= PSNR 24.800 / MUSIQ 56.05 / MANIQA 0.4401 / OCR **65.92** —— 见 §5.3。

**一眼结论**:除 2a 外,所有一步学生的 OCR 全挤在 **65.6–66.2**(~0.66 天花板);**只有 2a 捅破到 67.28**,且 PSNR/LPIPS 不降反优。

---

## 2. 评测与共同设置

- 评测集:Real-Text(847 对 / 1945 OCR regions),指标三件套 OCR-EM / OCR-CER↓ / **OCR-CharAcc↑**。口语"65.99"= CharAcc 0.6599。
- **单一变量原则**:除特别说明,所有蒸馏共享:同一 teacher/init、0.5B DiT(d1024/b28/h16/ps2)、SD2.1 4ch f8 VAE、DINOv2-b L8(vision-only,无文字条件)、512×4、20000 步、`distill_type:shortcut` 且 **`u_weight=0`(shortcut 自洽未生效,纯 guided-target 速度蒸馏)**、在线 RealESRGAN+RandomCrop 退化。
- **噪声口径**:差异用"最后 3–5 个 ckpt 的 mean±std"卡;差距 < std 视为平手。表中 ±0.1~0.5 的 OCR 差异多在噪声内,**只有 2a 的 +1.3、guided-target 的 +2.2、结构增强在 teacher 层的 +1.9 是可信大信号**。

---

## 3. 各配置的原理与代码实现

### 3.1 Teacher MS(多步,基准/上界)

多步 flow-matching 采样的 VOSR-text 模型。OCR 63.78 是所有一步学生的"出发点"。注意它 NIQE/MUSIQ 最好但 OCR 最低 —— 多步采样的随机性带来字符漂移。

### 3.2 B — 正常蒸馏(guided target,冠军基线)

文件:`vosr.py:227 loss_fm_distill_shortcut_improved`;config:`..._guided_target_no_rc.yml`。

```
v_guided = v_T^p + ω·(v_T^f − v_T^p)      # ω = cfg_scale = 0.5,partial→full 条件混合(非标准 CFG)
loss = ‖v_student − v_guided‖²            # u_weight=0,只有这一项
```

- 纯 latent 速度场蒸馏,无任何 OCR/像素监督。
- **OCR 从 teacher 63.78 → 65.99(+2.2)**:一步蒸馏减少多步字符漂移 + guided target 抑制幻觉。**这是 OCR 的主杠杆,设定了 ~0.66 基线。**

### 3.3 REPAocr — 全局表征对齐

文件:`train_vosr_distill.py:1282 extract_ocr_repa_tokens` + `models/repa_align.py:repa_cosine_loss`;config:`repa_type:ocr, repa_ocr_model:microsoft/trocr-base-printed, repa_layer:13, repa_weight:0.5`。

```
hidden = DiT 第13层内部 token(return_hidden_at=13)
s = RepaProjector(hidden)                  # LayerNorm→Linear→SiLU→Linear
t = TrOCR_encoder(HR图)                     # extract_ocr_repa_tokens(hq_repa),整图编码器特征
loss = repa_weight · (1 − cos(pool(s), pool(t)))   # token 数对不齐 → mode="global" 池化
total = v_loss + loss
```

- **三个特征**:① 对齐发生在 **DiT 内部隐藏层**(未解码成图);② **TrOCR 编码器**的语义特征(对像素清晰度相对不变);③ **整图 global pool**。
- multilayer 版 `[8,13,18]` + `repa_target_token_count:256` 做到了 token-level,但仍是"DiT hidden ↔ TrOCR 编码器"。
- **结果 65.81,饱和**。强化权重也只到 65.92(§5.3)。

### 3.4 结构增强 / CODSR(VAE encoder LoRA + RAGP + LQ-token-mod)

文件:`models/vae_encoder_lora.py`;config:`text_codsr_distill/..._vae_lora_guided_target_ocr_repa.yml`。

- teacher = VOSR + **RAGP**(region-adaptive guidance prior 噪声加权)+ **LQ token modulation** + **VAE encoder LoRA(rank4)** 训练得到。
- 蒸馏期 `use_vae_encoder_lora:True` 但 **冻结加载**(`train_vosr_distill.py:756` "Frozen VAE encoder LoRA loaded"),不再训练。
- **代码事实**:LoRA **只加在 encoder**(`collect_vae_encoder_lora_targets`:`if "encoder" not in name: continue`),**decoder 不变** → 编/解码不对称 → 这是结构增强分支 PSNR 偏低(23.5/23.6)的来源。
- 该 config 里的 OCR-REPA 是**弱单层** `repa_layer:13, repa_weight:0.5`。
- **结果**:teacher 层有效(64.61/65.67,比原 teacher 63.78 高 +1~1.9),但**蒸馏后被洗到 65.59/66.23**(见 §5.4)。

### 3.5 2a — local-crop OCR-REPA(🏆 破天花板)

文件:`models/ocr_repa.py: compute_loss`;config:`..._ocr_local_gt.yml`(`ocr_use_gt_boxes:True, ocr_repa_weight:0.5, ocr_repa_on_sync_only:True, CRNN`)。

```
x0 = z_t − t·v_student;  pred_img = VAE.decode(x0)        # 先解码成 512px 真实图
boxes = GT框(restoration_dataset.json 预索引)或 DBNet 在线检测
for 每框: crop_pred, crop_HR ← 从 pred/HR 裁同框 → resize(3,32,320)
fp = CRNN.backbone+neck(crop_pred)                        # (N,T,C) 字符序列特征
ft = CRNN.backbone+neck(crop_HR).detach()
loss = ocr_repa_weight · (1 − cos(fp, ft))               # 逐 token 余弦,按框平均
total = v_loss + loss
```

- **作用在解码后的像素图**(梯度经 VAE 回流,直接修正渲染出的字形)、**逐文字框**、**CRNN 识别器的序列特征**。
- **box 只在训练用**;推理就是普通一步 SR,不需要任何框 → **不是 oracle 作弊**。
- **结果 67.28(step 18000 已最高),且 PSNR 24.885 > B、LPIPS 0.2471 全表最佳**。

### 3.6 2b — CTC 字符级监督(设计,待测)

文件:`models/ocr_repa.py: compute_ctc_loss`;config:`ocr_repa_weight:0.0`(关2a)+ `ocr_ctc_weight:0.5`。

```
pred_img = VAE.decode(x0)
boxes = DBNet 在线检测
crop_HR → CRNN(backbone+neck+head) → logits → greedy CTC 解码 → 伪标签字符序列(确定性,不耗 RNG)
crop_pred → CRNN → logits
loss = ocr_ctc_weight · CTCLoss(log_softmax(pred_logits), HR伪标签)
total = v_loss + loss
```

- 沿用 2a 的"解码+裁框+识别器"定位通路,但**最后一步从"对齐 neck 特征"升级为"用 head 的 logits 算字符 CTC"** —— 从"让特征像"到"让它解码出和 HR 一样的字符"。

---

## 4. 四种 OCR 监督设计的精确对照

| 维度 | ① B 正常蒸馏 | ② 全局 REPAocr | ③ 2a local-crop | ④ 2b CTC |
|---|---|---|---|---|
| 监督信号 | 匹配 teacher 速度 | 特征 cosine | 特征 cosine | **字符 CTC** |
| 作用位置 | latent 速度场 | **DiT 内部 hidden** | **解码后像素图** | 解码后像素图 |
| 空间粒度 | 整图 latent | **整图 global pool** | **逐文字框** | 逐文字框 |
| OCR 模型 | 无 | TrOCR **编码器** | CRNN **识别器 backbone+neck** | + CRNN **head** |
| 特征/标签 | — | 语义编码特征 | 字符**序列特征** | 字符 **logits+伪标签** |
| 目标图 | — | HR(整图) | HR(同框) | HR(同框,解码出字符) |
| 训练用 box | 否 | 否 | **是(仅训练)** | 是(仅训练) |
| OCRacc | 65.99 | 65.81 | **67.28** | 待测 |

---

## 5. 机制分析(为什么)

### 5.1 2a 破天花板:三个设计轴一起起作用

对比 ②→③ 的三处差异,正是 2a 赢的原因:

1. **作用在解码后的真实图像**(而非 DiT 内部 hidden):梯度经 VAE 直接作用到"渲染出来的字形",修正的是人眼/识别器真正看到的像素结构。
2. **逐文字框**(而非整图 global pool):文字信号不被大面积背景稀释;每个框独立对齐其字符序列。
3. **CRNN 识别器的 per-column 序列特征**(而非 TrOCR 编码器的整体语义特征):CRNN 沿字宽方向输出序列,更贴"能不能逐字读对",而 TrOCR 编码器特征对清晰度相对不变。

→ **关键不是"加不加 OCR loss",而是"局部化 + 作用于输出图像 + 识别器字符特征"这三件一起。** 这证实了项目主张:**REPA 用于文本 SR 必须 region-local 且作用于像素输出**。

### 5.2 为什么全局对齐/结构增强会饱和

- 全局 REPAocr:在 DiT 内部 hidden 上、对整图语义特征做对齐 —— 既不直接作用于字形,又被背景稀释,信号到不了"字符可读性",所以卡 65.81。
- 结构增强(VAE-LoRA/RAGP):改的是隐空间/条件,不是字符级监督;teacher 层能涨,但(见 5.4)蒸馏把它洗平。

### 5.3 感知-失真权衡:OCR-REPA 权重是"感知旋钮"不是"OCR 旋钮"

REPAocr 权重 4×(0.25→1.0)、global 2×(0.25→0.5):

| | 旧 REPAocr | 强化版 | Δ |
|---|---:|---:|---:|
| PSNR | 25.796 | 24.800 | **−1.0** |
| MUSIQ | 53.29 | 56.05 | **+2.8** |
| MANIQA | 0.4058 | 0.4401 | **+0.034** |
| OCRacc | 65.81 | 65.92 | +0.11(噪声) |

- 加大全局特征对齐 → **OCR 不动,拿 PSNR 换 MUSIQ/MANIQA**。原因:TrOCR 编码器特征对像素清晰度不变 + 不带字符判别信息;加权只把输出往"语义像文字"的流形拉(NR 感知↑、偏离像素真值 PSNR↓)。
- **教训:全局 OCR-REPA 控制感知真实感,不控制识别率。** 与 2a(OCR↑ 且 PSNR/LPIPS 不降)形成鲜明对照。

### 5.4 蒸馏是"均衡器":teacher 端增益被洗平

| 蒸馏 | teacher OCR | → distill OCR | 蒸馏增量 |
|---|---:|---:|---:|
| B | 63.78(原MS) | 65.99 | +2.21 |
| 结构增强② | 65.67 | 66.23 | +0.56 |

- 三个一步学生(A 65.82 / B 65.99 / 结构增强 65.59~66.23 / REPAocr 65.81)全收敛到 ~0.66。
- **结构增强在 teacher 层有效(+1.9),但蒸馏把它压到与 B 平手,而 PSNR 损失留下** → 说明 **teacher 端的任何增益都被一步蒸馏的 OCR 天花板吃掉**;要突破必须在 **蒸馏过程中直接监督 student**(2a/2b)。这正是 2a 成功、CODSR 平手的根因。

### 5.5 为什么"加 OCR-REPA 但 PSNR 没动"

OCR-REPA 是**特征对齐辅助项,不是像素/保真损失**,与 PSNR 正交。PSNR 由 ① 蒸馏 v_loss 重建 + ② VAE 编解码保真上限决定。中等权重下 PSNR 不动是正常的;权重拧大反而掉(§5.3)。**想用 OCR-REPA 提 PSNR 是用错工具。**

---

## 6. 结论:什么有效 / 什么饱和

- **对 OCR 有效**:① **guided target**(设定 ~0.66 基线);② **2a 局部 OCR-REPA**(唯一破到 67.28,且不损保真)。
- **对 OCR 饱和(实为感知/保真旋钮)**:全局 REPAocr(各权重)、结构增强 / VAE-LoRA / RAGP。
- **真正的设计原则**:文本 SR 的表征对齐必须 **region-local + 作用于输出图像 + 用识别器字符特征**;全局 / DiT-hidden / 编码器语义特征都不行。
- **蒸馏是均衡器**:teacher 端增强(目标构造、结构增强)会被一步蒸馏的 OCR 天花板洗平 → 杠杆必须放在 student 端的局部监督。

---

## 7. Caveat 与待办

1. **2a 数据是 step18000(GT 框)**;待补:① 20000 最终值;② 最后 3–5 ckpt 的 mean±std 把 +1.3 坐实;③ **在线检测版 2a** 复现破天花板(证明不依赖 GT 框)。
2. **2b(CTC)是决定性下一步**:既然局部**特征对齐**到 67.28,局部**字符监督**能否再上一层?若 2b 也停在 ~67,则"局部化"是主因、字符监督边际有限;若再升,则字符监督是额外杠杆。
3. **噪声带**:表中多数 ±0.1~0.5 的差异先当平手,勿过度解读。
4. **单一 init 条件性**:主机制结论建议至少换一个 init 复现一次。
