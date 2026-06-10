# 流程图生成提示词(已按代码修正)— 2a Local-Crop OCR-REPA

> 用途:生成 "2a Local-Crop OCR-REPA Loss for One-Step Text Image Super-Resolution Distillation" 的方法流程图(论文/组会)。
> 本文件是**已修正版**:对照 `models/ocr_repa.py: compute_loss`、`decode_x0_from_distill_aux`、`train_vosr_distill.py:1408` 校准。
>
> **相对初版的修正:**
> 1. 🔴 `λ_ocr` 只出现一次(原稿在 loss 定义和 total 里各乘一次 = 重复)。
> 2. 🟡 余弦对齐对 **box 和 sequence token 一起平均**(N×T),非仅 "mean over boxes"。
> 3. 🟡 `x₀` 是**干净 latent**(非图像),VAE decode 后才成图。
> 4. 🟡 解码器写明 **SD2.1 VAE Decoder**(训练期此通路用全 VAE;推理用轻量 lwdecoder)。
> 5. 🟡 两个 crop 过的是**同一个、共享权重的冻结 CRNN**。

---

## 提示词

请生成一张清晰、专业、适合论文和组会展示的深度学习方法流程图,主题为:

**"2a Local-Crop OCR-REPA Loss for One-Step Text Image Super-Resolution Distillation"**

### 整体风格

- 白色背景
- 现代扁平化科研信息图
- 清晰的模块方框、箭头和数学符号
- 蓝色表示 Student 主干网络
- 绿色表示 HR 监督分支
- 橙色表示 OCR 特征提取和 Loss
- 灰色虚线表示冻结模块或停止梯度(加雪花/锁图标)
- 红色反向箭头表示梯度回传
- 横向布局,从左到右展示;比例 16:9 宽屏论文流程图
- 字体简洁,适合学术论文插图
- 不要复杂装饰,不要真实人物,不要照片风格

### 图标题

**"Local-Crop OCR-REPA Supervision"**

主流程分为上下两条并行分支。

---

### 上方:Student Prediction Branch(蓝色)

1. 左侧输入模块:**"Low-Quality Image"**,下面标注 **"LQ image"**。

2. 输入进入蓝色模块:**"One-Step SR Student"**,下面小字 **"DiT / Flow Student"**。

3. Student 输出速度预测:**"Predicted Velocity"**,数学符号 **v_student**。

4. 公式模块恢复干净 **latent**(画成 latent 方块,标明是 latent 不是图像):
   **"Recover Clean Latent"**,公式 **x₀ = zₜ − t · v_student**。

5. 输入冻结的解码器:**"Frozen SD2.1 VAE Decoder"**(灰色虚线边框 + 雪花/锁图标)。

6. 输出:**"Predicted SR Image"**(示意一张模糊文字被恢复成清晰文字的图)。

---

### 下方:HR Reference Branch(绿色)

1. 输入:**"Ground-Truth HR Image"**(一张清晰高分辨率文本图)。

2. 在 Predicted SR Image 与 HR Image 上绘制**完全对应的矩形文字框**:
   标注 **"Same Text Boxes"**;
   来源 **"GT boxes during training"**;
   旁注 **"No boxes required at inference"**。

3. 从两张图分别裁剪相同区域:
   上方 **"Predicted Text Crop"**,下方 **"HR Text Crop"**。

4. 两个 crop 同样预处理:**"Resize to 3 × 32 × 320"**。

5. 两个 crop 输入**同一个、共享权重的**冻结 CRNN OCR 网络(橙色模块、灰色虚线边框):
   **"Frozen OCR Recognizer (shared weights)"**,内部分 **"CRNN Backbone" → "CRNN Neck"**。

6. CRNN 输出**沿文字宽度方向排列的序列特征**(画成一排横向 token `[f₁, f₂, f₃, …, f_T]`):
   预测分支 **F_pred ∈ ℝ^(N×T×C)**;
   HR 分支 **F_HR ∈ ℝ^(N×T×C)**,旁标 **"Stop Gradient (detach)"**。

7. 将 F_pred 和 F_HR 输入橙色 Loss 模块:**"Local OCR-REPA Loss"**,显示公式:

   **L_OCR-REPA = mean_{boxes N, tokens T} [ 1 − cosine( F_pred , stopgrad(F_HR) ) ]**

   突出文字:
   - "Token-wise sequence feature alignment"
   - "Average over text boxes **and sequence tokens**"

8. 与基础蒸馏损失合并(λ_ocr **只在此处**出现一次):

   **L_total = L_guided-distill + λ_ocr · L_OCR-REPA**

   - L_guided-distill 蓝色(可附小字 `= ‖v_student − v_guided‖²`)
   - L_OCR-REPA 橙色

9. 从 Local OCR-REPA Loss 画一条**红色反向梯度箭头**,依次经过:

   CRNN feature extraction → Predicted Text Crop → Predicted SR Image → Frozen SD2.1 VAE Decoder → x₀ (latent) → v_student → One-Step SR Student

   红色箭头旁标注 **"Gradient updates Student only"**。
   冻结的 CRNN 与 HR 特征不更新(灰色虚线 + 停止梯度)。

---

### 底部:三个核心优势标签

1. **"Decoded-image supervision"**(作用于解码后真实图像,而非 DiT hidden feature)
2. **"Region-local text alignment"**
3. **"Character-sequence-aware features"**

### 右下角:推理阶段说明

**"Inference: LQ Image → One-Step SR Student → SR Image"**
并注明:**"No OCR model"** / **"No text boxes"** / **"No extra inference cost"**

---

### 视觉重点(必须准确表现)

- 预测图像与 HR 图像使用**相同文字框**裁剪。
- CRNN 输出是**沿文字宽度方向排列的序列 token**。
- HR 特征**停止梯度**。
- OCR Loss 作用于**解码后的真实图像**,而非 DiT hidden feature(本方法与全局 REPAocr 的核心差异,务必突出)。
- OCR 模块**只在训练阶段使用**。
- `x₀` 是 **latent**,VAE 之后才是图像。
- `λ_ocr` **只乘一次**。
- 箭头顺序准确,模块名称不能拼错。
