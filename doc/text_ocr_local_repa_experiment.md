# Local-crop OCR-REPA 实验说明 (2a)

更新时间：2026-06-07

## 动机

现有 `repa_type: ocr`(表中的 `REPAocr`)把整张图过 TrOCR,再把所有 token **全局 mean-pool 成一个向量** 做 cosine 对齐。文字信息是局部 glyph 级的,全局池化恰好抹掉了空间结构——这既不符合 REPA 原论文(per-patch 对齐),对文本也不敏感。结果上 `REPAocr` 的 OCR(65.81)和纯 B baseline(65.99)基本持平,说明全局表征对齐对 OCR 已饱和。

本实验(2a)用文本 SR 的"正确"做法替代:

```text
HR 图  --det(no_grad)-->  文字 bbox
pred 图 ┐
HR  图  ┴ 裁同一批 bbox -> resize -> 冻结 recognizer backbone+neck -> 逐区域序列特征
L_ocr = 1 - cosine( feat(pred_crops), feat(HR_crops).detach() )
总损失: L = L_guided_distill + ocr_repa_weight * L_ocr
```

关键点:
- bbox 在 **HR(干净图)** 上检测(LQ 上会漏检)。
- 训练用 RandomCrop + 在线退化,所以 bbox 必须**在线检测**当前 HR crop,无法按文件名离线缓存。
- 检测器/识别器全部冻结,梯度只**穿过** recognizer 回流到预测图(再回到 student)。
- 复用 FD-loss 已有的一步解码通路 `x0 = z_t - t·v_student → VAE decode`。

## 新增 / 改动

```text
models/ocr_repa.py                  # OCRRepaSystem: 检测 + 可微裁剪 + 识别特征对齐
train_vosr_distill.py               # 构建 + 按 interval/start_step 门控 + 计入 loss + 日志
configs/.../VOSR_0.5B_text_guided_target_no_rc_ocr_local_repa.yml
scripts/train_text_repa_ablation.sh # 新增 ocr_local 模式
tests/test_ocr_repa.py              # 检测/可微/无文字 三个用例
requirements.txt                    # + shapely, pyclipper (PaddleOCR DB 后处理依赖)
```

OCR 模型用仓库内 vendored 的 PaddleOCR2Pytorch,**不依赖联网**:
- det: `PaddleOCR2Pytorch/weights/ch_ptocr_server_v2.0_det_infer.pth`
- rec: `PaddleOCR2Pytorch/weights/ch_ptocr_server_v2.0_rec_infer.pth`(CRNN, neck_out 为 `(N,T,256)` 序列特征)

## 配置项 (YAML)

```yaml
ocr_repa_weight: 0.5          # >0 即启用;0/缺省则关闭
ocr_repa_interval: 1          # 每隔多少 step 算一次
ocr_repa_start_step: 0        # 从第几步开始
ocr_repa_on_sync_only: False
ocr_repa_rec_algorithm: CRNN
ocr_repa_rec_image_shape: "3,32,320"
ocr_repa_det_limit_side_len: 960
ocr_repa_det_db_box_thresh: 0.6
ocr_repa_max_boxes: 16        # 每图保留最大的若干文字框
ocr_repa_min_box_size: 8
ocr_repa_loss_type: cosine    # cosine | mse
# 可选: ocr_repa_det_model_path / ocr_repa_rec_model_path / *_yaml_path / ocr_repa_paddle_root
```

## 运行

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_local
```

测试:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_ocr_repa.py -q
```

## 与论文消融的关系(locality ablation)

| 设置 | OCR 特征粒度 | 配置 |
|---|---|---|
| global pool | 整图 1 向量 | `repa_type: ocr`(REPAocr, 65.81) |
| region pool(本实验) | 每文字框 1 段序列 | `ocr_repa_*`(ocr_local) |
| dense/CTC(2b, 待做) | glyph 级序列 | TODO |

预期:region/CTC > global,直接论证"REPA 用于文本 SR 必须局部化"。这是把"表征对齐"推进到"直接文字监督"的第一步;2b(HR 伪标签 + CTC 识别 loss)在此基础上扩展。

---

## 2b：HR 伪标签 + CTC 识别 loss

2a 是"对齐特征",2b 是"直接监督识别结果"——把可读性写进梯度。

```text
HR 图  --det(no_grad)-->  文字 bbox
       --crop+rec(no_grad)-->  greedy CTC 解码 -> 伪标签 index 序列
pred 图 --crop+rec--> CTC logits  --CTCLoss(伪标签)-->  L_ctc
总损失: L = L_guided_distill + ocr_ctc_weight * L_ctc
```

要点:
- 伪标签来自 HR crop 的 recognizer 贪心 CTC 解码(blank=0,合并重复),**确定性、不消耗 RNG**。
- recognizer 完全冻结;`_rec_logits` 复用 2a 的 `_rec_features`,再手动过 `head.fc` 取原始 logits(eval 模式 head 会自带 softmax,故绕过)。
- 无可解码文字(伪标签全空)→ 返回 0 loss,不崩。
- `zero_infinity=True`,并 clamp `target_length <= T`,避免 CTC inf。

### 与 2a 的隔离保证(已用测试钉死)

2b 是**纯加法、默认关**:
- 独立键 `ocr_ctc_weight`(缺省 0);2a 的 yaml 不含它 → `ctc_active` 恒 False。
- `ocr_active` 加了 `args.ocr_repa_weight > 0` 守卫,使 CTC-only 跑法不会触发 2a。
- 不改 `compute_loss` / `_detect_boxes` / `_crop_and_stack`,只新增 `compute_ctc_loss` / `_rec_logits` / `_ctc_greedy_targets`。
- `tests/test_ocr_repa.py::test_2b_does_not_perturb_2a` 断言:同输入下计算 2b 前后 2a 的 loss 完全相等。

### 配置项 (YAML)

```yaml
ocr_repa_weight: 0.0     # 关掉 2a,单独看 2b 信号
ocr_ctc_weight: 0.5      # >0 即启用
ocr_ctc_interval: 1
ocr_ctc_start_step: 0
ocr_ctc_on_sync_only: False
# det/rec 配置与 2a 共用同一组 ocr_repa_* 键
```

### 运行

```bash
NPROC_PER_NODE=4 bash scripts/train_text_repa_ablation.sh ocr_ctc
```

2a 与 2b 互不冲突,也可在同一份 yaml 里同时设 `ocr_repa_weight` 和 `ocr_ctc_weight` 做组合(检测会各跑一次,开销翻倍)。

## 已知注意事项

- recognizer 在 `accelerator.autocast()` 下运行;crop 与特征已显式 `.float()`,pred/HR 走同一路径保持一致。
- 无文字图返回 0 loss(`pred.sum()*0`),不会破坏计算图或崩溃。
- 每个 DDP 进程各自构建一份冻结 OCR 系统;det 用 numpy 逐图运行,batch 大时按 `ocr_repa_max_boxes` 截断防 OOM。
- 当前用 CRNN(中英)recognizer;如需多语种/更强 glyph 表征可换 v4 rec(需提供对应 yaml)。
```
