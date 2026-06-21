# Real-CE 基准评测流程(ICCV 2023)

> 建立日期:2026-06-10。Real-CE = 中英真实场景文本 SR 基准(13mm/26mm/52mm 三焦段配准对,val 300 张、官方 valid_list 260 张)。
> 论文用途:第二个 benchmark,与自建 real_test(847 张)互补;注意数据集协议为非商用研究用途、不得再分发。

## 资产位置

| 内容 | 路径 |
|---|---|
| 官方 repo(已克隆+打补丁) | `/data/ywk/Real-CE`(补丁:recognition.py 硬编码路径→模块相对路径、`np.int`→`int`) |
| 识别器权重(官方 CRNN 中/英) | `/data/ywk/Real-CE/basicsr/metrics/{scene_base_CRNN.pth, crnn.pth}` |
| 数据 val 集(本机) | `/data/ywk/datasets/Real-CE/val/{13mm,26mm,52mm,annos,valid_list.txt}` |
| 数据(199,含 train) | `/data2/wyw/ywk/datastes/Real-CE/` |
| 评测脚本(自包含,复用官方指标实现) | `scripts/eval_realce.py` |
| 推理脚本(本机/199) | `scripts/infer_realce_local.sh` / 199:`VOSR_textsr/infer_realce_remote.sh` |

## 任务定义(×4)

- **LQ = 13mm,GT = 52mm,两者已配准到同一分辨率**(如 2680×1872)→ 我们的模型用 `upscale=1` + **平铺推理**(`--tile_size 512 --tile_overlap 64`,tile 即训练分辨率)。
- 标注:GBK 编码,每行 8 个多边形坐标 + 转录文字。

## 指标(忠实官方协议)

PSNR / SSIM / LPIPS(crop_border=2, RGB)+ **文字区域 masked PSNR/SSIM**(多边形掩码)+ **识别 ACC / NED**(官方 minAreaRect 裁剪 → 中文 scene_base_CRNN / 英文 crnn 按语言分流 → 大小写不敏感、去空格 → 精确匹配率 + `(max_len−editdistance)/max_len`)。

## 用法

```bash
# 1) 推理(本机,等 GPU 空闲;TAG 任取)
EXP=exp_vosr_text_distill_ablation/ldit_distill_..._ocr_local_gt STEP=00020000 GPU=0 TAG=2a_local \
  bash scripts/infer_realce_local.sh

# 2) 评测(输出 sd2_steps1_seed42_shortcut 子目录)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_realce.py \
  --sr_dir preset/results/realce/2a_local_step00020000/sd2_steps1_seed42_shortcut \
  --out_json preset/results/realce/2a_local_step00020000.json

# 输入下限行(13mm 自身):
python scripts/eval_realce.py --sr_dir /data/ywk/datasets/Real-CE/val/13mm
```

3 张图 sanity(13mm vs 52mm):PSNR 21.99 / mask_PSNR 14.06 / rec_acc 0.545 / NED 0.821 —— 管线验证通过。

## 排队计划

1. ⏳ **2a-remote ckpt-20000**:199 GPU3 推理中(screen `infer_realce`,日志 `VOSR_textsr/infer_realce.log`),完成后拉回本机评测;
2. 2b/E1 让出本机 GPU 后:B、2a-local(18k/20k)、teacher(多步)依次跑 `infer_realce_local.sh`;
3. 全部出齐后汇成 Real-CE 主表(行:bicubic/13mm 下限、teacher、B、2a、2b、E1/E2)。

## 注意事项

- 评测脚本对训练中的机器友好:eval 占 ~1.5GB 显存,可与训练共存(挑余量最大的卡)。
- 识别评测会在终端打印每条 `pred || gt` 对照(官方行为),日志会比较长。
- `crop_images` 用 minAreaRect 透视矫正裁剪;竖排文本(高>宽)官方会旋转处理,无需我们干预。
- 论文报数用 valid_list 的 260 张(评测脚本默认),推理跑全部 300 张无妨。
