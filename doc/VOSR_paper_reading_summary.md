# VOSR 论文阅读总结

> 论文：VOSR: A Vision-Only Generative Model for Image Super-Resolution  
> 状态：CVPR 2026 accepted  
> 论文链接：https://arxiv.org/pdf/2604.03225  
> 代码链接：https://github.com/cswry/VOSR  
> 当前仓库来源：https://github.com/pikangqiu/textdif  

本文整理 VOSR 的主要思路、实现细节、训练 recipe、指标体系、对比方式，以及和当前 text-SR one-step distillation 项目的关系。

## 1. 一句话概括

VOSR 想回答的问题是：图像超分是否一定要依赖 Stable Diffusion / SDXL / SD3 这类 text-to-image 预训练模型？论文的答案是：不一定。只用视觉数据训练一个 restoration-oriented generative SR 模型，也可以在感知质量、结构忠实度、速度和训练成本之间取得很强的平衡。

它的核心组合是：

```text
LR image
  -> VAE encoder -> structural latent condition
  -> DINOv2 / vision encoder -> visual semantic condition
  -> LightningDiT / flow matching backbone
  -> multi-step teacher
  -> one-step distilled student
```

相比 T2I-based SR，VOSR 不使用文本 prompt、不训练 text encoder、不需要 image-text pair，而是把 SR 重新建模成低质量输入条件下的视觉恢复生成任务。

## 2. 论文的主要动机

传统 generative SR 方法大致有三类：

| 路线 | 优点 | 问题 |
|---|---|---|
| pixel / distortion-oriented SR | PSNR、SSIM 高 | 纹理容易过平滑 |
| GAN / perceptual SR | 图像锐利 | 对抗训练不稳定，容易有伪影 |
| T2I diffusion-based SR | 生成先验强 | 文本/多模态先验和 LR fidelity 有结构性冲突 |

VOSR 认为 SR 不是自由生成，而是 LR input-conditioned restoration。T2I 模型虽然强，但它的语义先验来自文本或 text-aligned feature，空间上不够细，容易在细结构、文字、符号处出现 hallucination。

因此 VOSR 的设计重点是：

1. 用视觉语义特征替代文本语义特征。
2. 用低质量图像 latent 保留结构条件。
3. 改造 CFG，使 guidance 分支仍然被 LR 输入锚定。
4. 先训练 multi-step 模型，再蒸馏成 one-step 模型，兼顾质量和速度。

## 3. 方法细节

### 3.1 条件设计

VOSR 使用两类条件：

| 条件 | 来源 | 作用 | 代码对应 |
|---|---|---|---|
| structural condition | LR 图像经过 VAE encoder 后的 latent | 锚定 LR 输入结构，保证 fidelity | `encode_lq_hq_latents()`、`vosr.py` 中 `lq` |
| visual semantic condition | LR 图像经过 DINOv2 等视觉 encoder 后的 token feature | 提供视觉语义和上下文，帮助解决局部歧义 | `load_dinov2()`、`forward_with_features()`、`LightningDiT` cross-attn |

在当前代码里，0.5B 默认用 SD2.1 VAE，latent channel 为 4；1.4B 用 Qwen-Image 2D VAE，latent channel 为 16。DINOv2 特征会经过投影后送进 DiT block 的 cross-attention。

### 3.2 Backbone：LightningDiT

主干是 DiT / SiT 风格的 latent flow model。输入是当前 noisy latent 和 LR latent 的拼接：

```text
input = concat(lq_latent, z_t)  # channel 维拼接
output = velocity / flow direction
```

关键结构：

| 模块 | 作用 |
|---|---|
| `PatchEmbed` | 把 latent patch 化成 token |
| timestep embedding | 编码 flow matching 的时间 `t` |
| auxiliary timestep `r` | one-step / shortcut / RCGM 蒸馏时使用 |
| cross-attention | 注入 DINOv2 visual semantic token |
| RoPE / RMSNorm / QK norm / SwiGLU | Transformer 稳定训练和表示增强 |
| `FinalLayer + unpatchify` | 输出回 latent feature map |

当前仓库的 `models/lightningdit.py` 中，`forward()` 和 `forward_flexible()` 都支持 `z` 作为 visual semantic condition；`forward_flexible()` 还会在推理分辨率变化时动态生成 RoPE。

### 3.3 Flow Matching 训练目标

训练时从 HQ latent 和噪声之间插值得到：

```text
z_t = (1 - t) * hq + t * eps
target velocity = eps - hq
```

模型学习预测 velocity。当前实现对应 `vosr.py::loss_fm()`：

```text
v_current = model(concat(lq_mixed, z_t), t, z=z_mixed)
loss = mean((v_current - (eps - hq))^2)
```

训练中 LQ 不是离线读入，而是由 HQ 图像在线经过 Real-ESRGAN degradation pipeline 生成，这一点在 `train_vosr.py` / `train_vosr_distill.py` 中都一样。

## 4. Restoration-Oriented CFG

### 4.1 标准 CFG 为什么不适合 SR

标准 CFG 通常是：

```text
v_cfg = v_uncond + s * (v_cond - v_uncond)
```

问题在于，SR 是输入条件恢复任务。如果 unconditional branch 完全不看 LR，它就变成了从头学一个通用图像生成器。对于从零训练的 SR 模型来说，这个分支很难学好；如果 `v_uncond` 不可靠，guidance 方向也会不稳定。

### 4.2 VOSR 的 partial conditioning

VOSR 把 unconditional branch 换成 partially conditioned branch：

| 分支 | structural condition | semantic condition |
|---|---|---|
| full branch | 强 LR latent condition | 有 DINOv2 semantic condition |
| partial branch | 弱 LR latent condition | 无 semantic condition |

guided prediction 写作：

```text
v_g = v_p + omega * (v_f - v_p)
```

其中 `v_f` 是 fully conditioned prediction，`v_p` 是 partially conditioned prediction。

当前代码对应：

- 训练条件混合：`vosr.py::_prepare_cfg_conditions()`
- 蒸馏条件混合：`vosr.py::_prepare_cfg_conditions_distill()`
- teacher target 构造：`vosr.py::_teacher_target()`
- multi-step inference CFG：`vosr.py::sample_multistep_fm()`

### 4.3 CFG 消融

论文在 LSDIR 上比较了三种 guidance：

| Guidance design | LPIPS ↓ | MUSIQ ↑ |
|---|---:|---:|
| Full condition only | 0.3752 | 67.29 |
| Standard CFG | 0.4053 | 50.78 |
| Ours partial conditioning | 0.3772 | 69.26 |

结论：standard CFG 明显退化；partial conditioning 的 MUSIQ 最好，说明 input-anchored auxiliary branch 更适合 SR。

## 5. One-Step Distillation

VOSR 的 one-step 模型不是直接训练一个新模型，而是从 multi-step teacher 蒸馏得到。蒸馏目标不是裸的 full branch prediction，而是 restoration-oriented guidance 后的 teacher target：

```text
v_teacher_guided = v_p + omega * (v_f - v_p)
```

论文附录比较了 shortcut-based 和 RC-based distillation：

| Method | LPIPS ↓ | MUSIQ ↑ |
|---|---:|---:|
| Teacher VOSR-0.5B-ms | 0.3069 | 68.93 |
| Shortcut-based distillation | 0.2913 | 68.21 |
| RC-based distillation | 0.2856 | 69.78 |

通用图像 SR 上，RC-based distillation 的 LPIPS 和 MUSIQ 都更好，因此是论文最终偏好的 one-step recipe。

当前仓库里两种路线都在 `vosr.py`：

| 蒸馏方式 | 函数 | 机制 |
|---|---|---|
| shortcut | `loss_fm_distill_shortcut_improved()` | teacher guided target + midpoint consistency |
| rcgm | `loss_fm_distill_rcgm_improved()` | teacher guided target + recursive consistency / trajectory compression |

需要注意：在当前 text-SR 实验里，RCGM 对通用感知指标有帮助，但会伤 OCR CharAcc；所以文本超分不应机械照搬 VOSR 的 RC 最终 recipe。

## 6. 模型规模与配置

| 项目 | VOSR-0.5B | VOSR-1.4B |
|---|---:|---:|
| Diffusion backbone | LightningDiT | LightningDiT |
| hidden dim | 1024 | 1536 |
| depth | 28 | 36 |
| heads | 16 | 24 |
| patch size | 2 | 2 |
| MLP ratio | 4 | 4 |
| VAE | SD2.1 VAE | Qwen-Image 2D VAE |
| latent channels | 4 | 16 |
| semantic encoder | DINOv2-Base | DINOv2-Large |

当前配置文件对应：

- 0.5B multi-step：`configs/train_yml/multi_step/VOSR_0.5B.yml`
- 1.4B multi-step：`configs/train_yml/multi_step/VOSR_1.4B.yml`
- 0.5B one-step：`configs/train_yml/one_step/VOSR_0.5B.yml`
- 1.4B one-step：`configs/train_yml/one_step/VOSR_1.4B.yml`

## 7. 训练细节

论文训练采用 progressive recipe：

| 阶段 | 分辨率 | 步数 | global batch | learning rate |
|---|---:|---:|---:|---:|
| multi-step pretrain | 256x256 | 400K | 1024 | 1.0e-4 |
| multi-step finetune | 512x512 | 400K | 256 | 5.0e-5 |
| one-step distillation | 512x512 | 50K | 32 | 2.0e-5 |

统一设置：

| 项 | 值 |
|---|---:|
| optimizer | AdamW |
| beta1 / beta2 | 0.9 / 0.95 |
| weight decay | 0.01 |
| warmup | 0 |
| gradient clip | 1.0 |
| EMA decay | 0.9999 |

当前仓库默认训练配置和论文 recipe 接近，但不完全相同。例如 `configs/train_yml/one_step/VOSR_0.5B.yml` 中 `max_train_steps: 100001`，比论文 one-step 的 50K 更长；text-SR 分支的很多实验又改成了 20K、bs16、lr5e-6，用于受控消融。

## 8. 数据与 benchmark

### 8.1 训练数据

论文收集约 100M web images，经过以下过滤：

- gradient-based filtering
- image entropy filtering
- resolution filtering
- category-level de-duplication and balancing

训练 LR-HR pair 由 Real-ESRGAN degradation pipeline 合成。当前代码里也是在线退化：

```text
HQ image -> RealESRGAN_degradation.degrade_process() -> LQ image
```

### 8.2 评测数据

| 数据集 | 类型 | 用法 |
|---|---|---|
| LSDIR val | synthetic paired | 250 images，Real-ESRGAN degradation，x4 |
| RealSR | real-world paired | 经典真实 SR benchmark |
| ScreenSR | real-world paired | 论文新建，通过 screen re-photography pipeline 得到 |

LSDIR 和 RealSR 使用 512x512 GT center crop，并生成 128x128 LR，构成 x4 SR 协议。ScreenSR 的 LQ 和 GT 都是 512x512。

ScreenSR 最终包含 130 paired samples，覆盖室内、室外、人物、动物、植物、艺术品、中英文文字、静态/动态场景等。

## 9. 指标体系

论文使用 full-reference 和 no-reference 两类指标：

| 指标类型 | 指标 |
|---|---|
| distortion fidelity | PSNR、SSIM，论文中在 YCbCr 的 Y channel 上算 |
| reference-based perceptual quality | LPIPS、DISTS、AFINE-FR |
| no-reference perceptual quality | NIQE、MUSIQ、MANIQA、AFINE-NR、TOPIQ-NR |
| human preference | user study |

从论文结果看，VOSR 不追求最高 PSNR，而是强调 perceptual quality 和 input faithfulness 的平衡。对 text-SR 来说，这也提醒我们：PSNR/SSIM/MUSIQ 不能替代 OCR CharAcc / E2E spotting，需要另设文本可读性指标。

## 10. 主要量化结果

### 10.1 VOSR 在 RealSR 上的关键结果

| Method | Setting | Type | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| StableSR | multi-step | T2I | 24.6426 | 0.7079 | 0.3004 | 0.2140 | 5.8838 | 65.8802 | 0.6229 |
| PASD | multi-step | T2I | 25.2423 | 0.7223 | 0.2988 | 0.2065 | 5.2047 | 65.3484 | 0.5960 |
| SeeSR | multi-step | T2I | 25.1480 | 0.7211 | 0.3007 | 0.2224 | 5.3973 | 69.8179 | 0.6451 |
| ResShift | multi-step | VO | 26.2630 | 0.7405 | 0.3468 | 0.2495 | 7.1790 | 58.4687 | 0.5343 |
| VOSR-0.5B-ms | multi-step | VO | 25.4361 | 0.7125 | 0.3069 | 0.2260 | 5.7070 | 68.9277 | 0.6429 |
| VOSR-1.4B-ms | multi-step | VO | 25.2886 | 0.7150 | 0.2961 | 0.2226 | 6.2614 | 70.4718 | 0.6510 |
| OSEDiff | one-step | T2I | 25.1517 | 0.7341 | 0.2920 | 0.2128 | 5.6401 | 69.0830 | 0.6335 |
| PiSA-SR | one-step | T2I | 25.5030 | 0.7418 | 0.2672 | 0.2044 | 5.5033 | 70.1421 | 0.6551 |
| SinSR | one-step | VO | 26.2766 | 0.7347 | 0.3188 | 0.2352 | 6.2900 | 60.7849 | 0.5413 |
| VOSR-0.5B-os | one-step | VO | 25.4189 | 0.7220 | 0.2856 | 0.2110 | 5.2790 | 69.7775 | 0.6347 |
| VOSR-1.4B-os | one-step | VO | 25.2284 | 0.7175 | 0.2732 | 0.2054 | 5.4303 | 70.5813 | 0.6443 |

重点观察：

1. VOSR-1.4B-ms 在 RealSR 上 MUSIQ 最高。
2. VOSR-1.4B-os 在 one-step VO 模型中感知质量明显强于 SinSR。
3. VOSR 没有追最高 PSNR，但 LPIPS/MUSIQ/MANIQA 等感知指标很有竞争力。

### 10.2 复杂度与速度

论文在 A100、batch size 1、FP16、512x512 output 下测 runtime：

| Method | 类型 | Params (M) | Time (s) |
|---|---|---:|---:|
| ResShift | multi-step VO | 174.0 | 0.534 |
| VOSR-0.5B-ms | multi-step VO | 619.4 | 1.226 |
| VOSR-1.4B-ms | multi-step VO | 1742.9 | 2.114 |
| StableSR | multi-step T2I | 1540 | 10.036 |
| PASD | multi-step T2I | 1680 | 2.751 |
| SeeSR | multi-step T2I | 2510 | 3.943 |
| DiT4SR | multi-step T2I | 21380 | 12.550 |
| SinSR | one-step VO | 174.0 | 0.116 |
| VOSR-0.5B-os | one-step VO | 620.7 | 0.094 |
| VOSR-1.4B-os | one-step VO | 1745.7 | 0.095 |
| OSEDiff | one-step T2I | 1760 | 0.104 |
| PiSA-SR | one-step T2I | 1290 | 0.091 |

论文强调：VOSR multi-step 明显快于多步 T2I-based SR；one-step VOSR 和现有 one-step T2I 方法速度接近。

### 10.3 语义条件消融

RealSR 上，不同 semantic encoder 的消融：

| Method | LPIPS ↓ | MUSIQ ↑ |
|---|---:|---:|
| w/o SVE | 0.3011 | 63.74 |
| w/ SVE (CLIP) | 0.2788 | 63.82 |
| w/ SVE (SigLIPv2) | 0.2817 | 64.81 |
| w/ SVE (DINOv3) | 0.2858 | 67.51 |
| w/ SVE (DINOv2) | 0.2872 | 68.23 |

CLIP 的 LPIPS 最好，但 DINO 系列 MUSIQ 更强。论文最终选 DINOv2，是在整体表现和稳定性之间折中。

### 10.4 ScreenSR GT 质量

论文比较了不同真实 paired SR benchmark 的 GT no-reference 质量：

| Benchmark | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | AFINE-NR ↓ | TOPIQ-NR ↑ |
|---|---:|---:|---:|---:|---:|
| ScreenSR | 3.7719 | 72.1500 | 0.7187 | -1.2093 | 0.7363 |
| RealSR | 6.1167 | 57.4564 | 0.6016 | -0.9088 | 0.4140 |
| DRealSR | 6.7909 | 50.5644 | 0.5588 | -0.7731 | 0.3932 |

这个表用于支持 ScreenSR 的评测可靠性：参考图像质量更高，真实 paired 评测更可信。

## 11. 和当前 text-SR 项目的关系

当前项目并不是只复现 VOSR，而是在 VOSR 的 one-step restoration distillation 基础上研究文本超分：

```text
通用图像 SR 目标：perceptual quality + input faithfulness
文本 SR 目标：perceptual quality + glyph / character fidelity + OCR recognizability
```

已有实验表明：

1. one-step distillation 本身能减少多步采样的字符漂移，提升 OCR。
2. guided teacher target 比 full teacher target 更适合 text-SR。
3. 原始 VOSR 中对通用图像有利的 RC/RCGM，在当前文本设置里会损害 OCR CharAcc。
4. 真正能突破 OCR 上限的是 student 侧 text-aware supervision，例如 local OCR-REPA、真 GT 文本 CTC，或二者联合。

因此，当前论文故事线可以写成：

> VOSR 证明了 vision-only restoration distillation 能替代 T2I prior 做通用 generative SR；我们的 text-SR 工作进一步说明，文本区域是检验 restoration fidelity 的敏感场景。对于文字，one-step student 不只要继承通用感知质量，还要通过 text-aware target / representation / recognition supervision 保住字符身份。

## 12. 当前代码阅读索引

| 主题 | 文件 / 函数 |
|---|---|
| multi-step 训练入口 | `train_vosr.py` |
| one-step 蒸馏入口 | `train_vosr_distill.py` |
| FM loss | `vosr.py::loss_fm()` |
| shortcut distillation | `vosr.py::loss_fm_distill_shortcut_improved()` |
| RCGM distillation | `vosr.py::loss_fm_distill_rcgm_improved()` |
| CFG 条件构造 | `vosr.py::_prepare_cfg_conditions()` |
| guided teacher target | `vosr.py::_teacher_target()` |
| multi-step sampling | `vosr.py::sample_multistep_fm()` |
| one-step sampling | `vosr.py::sample_onestep()` |
| DiT backbone | `models/lightningdit.py::LightningDiT` |
| DINOv2 feature extraction | `train_vosr.py::load_dinov2()` |
| RealESRGAN online degradation | `dataloaders/realesrgan_gpu.py::RealESRGAN_degradation` |
| text-aware OCR supervision | `models/ocr_repa.py` |

## 13. 可复现命令入口

官方 README 给出的典型命令：

```bash
# multi-step 0.5B
torchrun --nproc_per_node=8 train_vosr.py \
  --config configs/train_yml/multi_step/VOSR_0.5B.yml

# one-step 0.5B
torchrun --nproc_per_node=8 train_vosr_distill.py \
  --config configs/train_yml/one_step/VOSR_0.5B.yml

# multi-step inference
python inference_vosr.py \
  -c preset/ckpts/VOSR_0.5B_ms \
  -i preset/datasets/inp_data \
  -o preset/results \
  -u 4

# one-step inference
python inference_vosr_onestep.py \
  -c preset/ckpts/VOSR_0.5B_os \
  -i preset/datasets/inp_data \
  -o preset/results \
  -u 4
```

当前 text-SR 分支的实验命令和结果以 `doc/result_exp.md` 为单一事实来源。

## 14. 读论文后的关键 takeaways

1. VOSR 的真正贡献不是“把 U-Net 换成 DiT”，而是把 SR 从 T2I adaptation 拉回 restoration-native generative modeling。
2. 视觉语义条件和结构 latent 条件互补；DINOv2 的空间 grounded token 比文本语义更适合细结构恢复。
3. partial conditioning 是 VOSR 的关键设计，解决了标准 CFG 的 unconditional branch 在 SR 中不好学的问题。
4. one-step distillation 是部署关键；通用图像上 RC-based distillation 最好，但文本 SR 上要重新评估，因为 OCR fidelity 对 trajectory compression 更敏感。
5. 论文指标体系偏通用图像质量，当前 text-SR 必须额外报告 OCR / spotting 指标，否则无法说明字符是否真的被恢复正确。
