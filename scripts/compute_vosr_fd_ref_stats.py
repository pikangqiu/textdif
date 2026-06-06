#!/usr/bin/env python3
"""Compute FD-Loss reference statistics for VOSR text training images.

This is the VOSR counterpart of Jiawei-Yang/FD-Loss `compute_repr_stats.py`.
It reads the same txt-list dataset config used by VOSR training, extracts
frozen judge-model features from real HR crops, and saves an `.npz` containing
`mu`/`sigma` plus optional `avg_mu`/`avg_sigma`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.fd_loss import load_repr_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VOSR FD-Loss reference stats")
    parser.add_argument("--train-dataset-config", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model", type=str, default="convnext")
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--honor-repeat",
        action="store_true",
        help="Use repeat counts from the dataset config. By default each HR path is used once.",
    )
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def read_dataset_config(config_path: str, honor_repeat: bool):
    config_path = Path(config_path)
    paths = []
    with config_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            list_path = Path(parts[0])
            repeat = int(parts[1]) if honor_repeat and len(parts) == 2 else 1
            if not list_path.is_absolute():
                list_path = ROOT / list_path
            with list_path.open("r") as list_file:
                image_paths = [x.strip() for x in list_file if x.strip()]
            for _ in range(repeat):
                paths.extend(image_paths)
    return paths


class CenterCropResize:
    def __init__(self, image_size: int):
        self.image_size = int(image_size)

    def __call__(self, pil_image: Image.Image):
        if min(pil_image.size) < self.image_size:
            scale = self.image_size / min(pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        crop_y = (pil_image.height - self.image_size) // 2
        crop_x = (pil_image.width - self.image_size) // 2
        return pil_image.crop(
            (
                crop_x,
                crop_y,
                crop_x + self.image_size,
                crop_y + self.image_size,
            )
        )


class TxtImageDataset(Dataset):
    def __init__(self, paths: list[str], resolution: int):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                CenterCropResize(resolution),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


@torch.inference_mode()
def extract_stats(model, loader, feat_dim, device, rank, world_size, max_images_per_rank=None):
    cls_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    cls_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    avg_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    avg_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    has_avg = torch.zeros(1, dtype=torch.long, device=device)
    count = 0

    pbar = tqdm(
        loader,
        desc="extracting FD stats",
        disable=rank != 0,
    )
    for images in pbar:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            cls_token, avg_token = model(images)

        cls64 = cls_token.double()
        cls_sum.add_(cls64.sum(0))
        cls_outer.addmm_(cls64.T, cls64)

        if avg_token is not None:
            has_avg[0] = 1
            avg64 = avg_token.double()
            avg_sum.add_(avg64.sum(0))
            avg_outer.addmm_(avg64.T, avg64)

        count += cls_token.shape[0]
        pbar.set_postfix({"images": count})
        if max_images_per_rank is not None and count >= max_images_per_rank:
            break

    count_t = torch.tensor([count], dtype=torch.long, device=device)
    if world_size > 1:
        dist.reduce(cls_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(cls_outer, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_outer, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(has_avg, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)

    if rank != 0:
        return None

    total = int(count_t.item())
    if total < 2:
        raise RuntimeError(f"Need at least 2 images to compute covariance, got {total}")

    def finish(sum_vec, outer_mat):
        s_np = sum_vec.cpu().numpy()
        mu = s_np / total
        sigma = (outer_mat.cpu().numpy() - np.outer(s_np, s_np) / total) / (total - 1)
        return mu, sigma

    mu, sigma = finish(cls_sum, cls_outer)
    stats = {"mu": mu, "sigma": sigma, "num_images": np.array(total, dtype=np.int64)}
    if int(has_avg.item()) > 0:
        avg_mu, avg_sigma = finish(avg_sum, avg_outer)
        stats["avg_mu"] = avg_mu
        stats["avg_sigma"] = avg_sigma
    return stats


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    paths = read_dataset_config(args.train_dataset_config, honor_repeat=args.honor_repeat)
    if args.num_images is not None and args.num_images < len(paths):
        rng = random.Random(args.sample_seed)
        paths = rng.sample(paths, args.num_images)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 images, got {len(paths)}")

    dataset = TxtImageDataset(paths, args.resolution)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    if rank == 0:
        print(
            f"Computing FD reference stats: model={args.model}, "
            f"images={len(paths)}, crop={args.resolution}, world_size={world_size}"
        )

    model, feat_dim, _, target_size = load_repr_model(
        args.model,
        device=device,
        target_size=args.target_size,
    )
    if rank == 0:
        print(f"Judge feature dim={feat_dim}, target_size={target_size}")

    max_per_rank = None
    stats = extract_stats(
        model,
        loader,
        feat_dim,
        device,
        rank,
        world_size,
        max_images_per_rank=max_per_rank,
    )

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **stats)
        print(f"Saved {output} (n={int(stats['num_images'])}, feat_dim={stats['mu'].shape[0]})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
