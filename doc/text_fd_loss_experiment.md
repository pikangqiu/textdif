# FD-Loss 文本蒸馏实验说明

本文档记录将 [Jiawei-Yang/FD-Loss](https://github.com/Jiawei-Yang/FD-Loss) 的训练机制迁移到当前 VOSR 文本蒸馏实验中的设置。

## 核心定位

FD-Loss 的目标不是替代 VOSR 的 guided target distillation，而是在已经可以输出一步 SR 图像的蒸馏阶段，额外约束生成图像在冻结表征模型中的分布：

```text
text-aware teacher checkpoint
    -> B guided target distillation
    -> FD image feature distribution regularization
```

因此当前推荐把它作为 B baseline 的附加项，而不是加到普通 multi-step FM 预训练早期。

## 迁移内容

本项目的实现保留 FD-Loss 仓库中的关键机制：

- frozen representation judge；
- 真实 HR 图像的参考统计 `.npz`：`mu` / `sigma`；
- generated feature queue / EMA statistics；
- checkpoint 中保存和恢复 `fd_queue_states.pt`；
- DDP 可微 `all_gather`；
- full-covariance differentiable Frechet distance；
- 原仓库的归一化形式：`fid / (fid.detach() + eps)`。

当前不是“batch 内 HR vs pred 的特征 MSE/对角方差近似”，而是仓库式的 generated distribution -> real distribution 对齐。

训练时的图像路径是：

```text
VOSR distillation aux:
    z_t, t, v_student
        -> x0_pred_latent = z_t - t * v_student
        -> VAE decode
        -> x_pred
        -> FD judge features
        -> FID(x_pred feature queue, HR reference stats)
```

总损失：

```text
L = L_distill + lambda_fd * L_fd
```

## 新增文件

```text
models/fd_loss.py
scripts/compute_vosr_fd_ref_stats.py
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_no_rc_fd.yml
configs/train_yml/one_step/text_distill_ablation/VOSR_0.5B_text_guided_target_shortcut_fd.yml
```

## 第一步：计算参考统计

FD-Loss 必须先对真实 HR 训练集计算 frozen judge 的参考分布统计。当前配置默认使用原仓库支持的 `convnext` judge，对应 `convnextv2_base.fcmae_ft_in22k_in1k`。

单卡：

```bash
python scripts/compute_vosr_fd_ref_stats.py \
  --train-dataset-config configs/train_txt/text_hr_512_dataset.txt \
  --resolution 512 \
  --model convnext \
  --target-size 224 \
  --num-images 20000 \
  --batch-size 64 \
  --num-workers 8 \
  --output preset/fd_stats/convnext_text_hr_512_stats.npz
```

多卡：

```bash
torchrun --nproc_per_node=4 scripts/compute_vosr_fd_ref_stats.py \
  --train-dataset-config configs/train_txt/text_hr_512_dataset.txt \
  --resolution 512 \
  --model convnext \
  --target-size 224 \
  --num-images 20000 \
  --batch-size 64 \
  --num-workers 8 \
  --output preset/fd_stats/convnext_text_hr_512_stats.npz
```

这里默认用 `--num-images 20000` 做 4*4090 上的可运行版本；脚本会用固定随机种子从训练列表中抽样。首次运行如果本地没有 timm / HuggingFace 缓存，可能会下载 `convnextv2_base.fcmae_ft_in22k_in1k` 权重。

## 第二步：训练

在 B baseline 上加 FD-Loss：

```bash
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_no_rc_fd
```

在 shortcut consistency 版本上加 FD-Loss：

```bash
NPROC_PER_NODE=4 bash scripts/train_text_distill_ablation.sh guided_target_shortcut_fd
```

## 关键参数

```yaml
fd_loss_weight: 0.01
fd_repr_models: [convnext]
fd_repr_stats_paths: [preset/fd_stats/convnext_text_hr_512_stats.npz]
fd_repr_weights: [1.0]
fd_repr_pool_types: [cls]
fd_target_sizes: [224]
fd_queue_size: 256
fd_fid_norm_eps: 0.01
fd_eigvalsh: True
fd_ema_beta: 0.999
fd_loss_interval: 50
fd_loss_on_sync_only: True
fd_loss_start_step: 50
fd_decode_latent_size: 32
use_checkpoint: False
```

说明：

- `fd_queue_size`：原仓库默认是 `50000`。VOSR 每次 queue fill 需要跑 degradation、DiT、VAE decode 和 judge；在 4*4090 上默认先用 `256` 做轻量趋势实验，后续如果趋势有效再升到 `1024/4096`。
- `fd_ema_beta: 0.999`：对应 FD-Loss 仓库常用脚本中的 EMA stats 方式。
- `fd_eigvalsh: True`：使用仓库提供的更快 full-covariance trace 计算路径。
- `fd_loss_interval: 50`：每 50 个 optimizer step 才计算一次 FD-Loss。
- `fd_loss_on_sync_only: True`：只在梯度累积的最后一个 micro-batch 上计算 FD-Loss。当前 `gradient_accumulation_steps=4`，因此实际 FD 额外开销约为每 200 个 micro-batch 触发一次。
- `fd_decode_latent_size: 32`：FD 分支把 64x64 latent 下采样到 32x32 后解码成 256 图像，再送入 ConvNeXt 的 224 输入。这保留 image-space FD 机制，同时避免 512 VAE decode 反向在 24GB 4090 上 OOM。
- `use_checkpoint: False`：当前 `LightningDiT` 内部有 `@torch.compile` 路径，DDP + checkpoint 在 shortcut/FD 的多次 student forward 下不稳定；4*4090 默认配置用低频 FD 和 256 decode 控制显存，而不打开 DiT checkpoint。

## 实验对照

| ID | 方法 | 主要变量 | 目的 |
|---|---|---|---|
| Teacher | text-aware multi-step teacher | 蒸馏前 | 基础文本能力 |
| A | full target no RC | `cfg_scale=1.0`, `u_weight=0` | 验证 full teacher target |
| B | guided target no RC | `cfg_scale=0.5`, `u_weight=0` | 当前 OCR 最好 baseline |
| B+Shortcut | guided target + shortcut consistency | `cfg_scale=0.5`, `u_weight=1` | 验证 shortcut consistency 是否有益 |
| B+FD | guided target + FD-Loss | repository-style FD | 验证图像表征分布对齐是否提升文字 |
| B+Shortcut+FD | guided target + shortcut + FD-Loss | 两种附加约束叠加 | 验证 FD 与 shortcut consistency 是否互补 |

重点指标仍然是：

```text
OCR-EM / OCR-CER / OCR-CharAcc
LPIPS / DISTS / NIQE / MUSIQ / MANIQA
```
