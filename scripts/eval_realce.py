#!/usr/bin/env python3
"""Evaluate SR results on the Real-CE benchmark (ICCV 2023) with the OFFICIAL
metric implementations, without installing the full BasicSR package.

Metrics (faithful to options/test/ESRGAN/*.yml + sr_model.py aggregation):
  - PSNR / SSIM        (crop_border=2, RGB, official psnr_ssim.py)
  - masked PSNR / SSIM (text-region polygons from annos, official region_*)
  - LPIPS              (official: lpips alex)
  - recognition ACC / NED (case-insensitive, official CRNN CHN/ENG weights,
    per-region exact match + (max_len - editdistance) / max_len)

Usage:
  python scripts/eval_realce.py \
      --sr_dir  <dir with SR images named like the 13mm inputs> \
      --gt_dir  /data/ywk/datasets/Real-CE/val/52mm \
      --anno_dir /data/ywk/datasets/Real-CE/val/annos \
      [--list /data/ywk/datasets/Real-CE/val/valid_list.txt] \
      [--no_rec]  # skip recognition (no CRNN weights needed)
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import cv2
import editdistance
import numpy as np

REALCE_ROOT = "/data/ywk/Real-CE"


def _load_realce_metrics():
    """Import the official metric modules from the Real-CE repo by faking the
    basicsr package skeleton (the real basicsr __init__ pulls in the whole
    training framework, which we neither need nor have installed)."""

    def fake_pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
        return mod

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    fake_pkg("basicsr", f"{REALCE_ROOT}/basicsr")
    fake_pkg("basicsr.utils", f"{REALCE_ROOT}/basicsr/utils")
    fake_pkg("basicsr.metrics", f"{REALCE_ROOT}/basicsr/metrics")
    load_file("basicsr.utils.registry", f"{REALCE_ROOT}/basicsr/utils/registry.py")
    load_file("basicsr.metrics.metric_util", f"{REALCE_ROOT}/basicsr/metrics/metric_util.py")
    # crnn is a sub-package with a real __init__.py; give it a proper spec.
    crnn_pkg = fake_pkg("basicsr.metrics.crnn", f"{REALCE_ROOT}/basicsr/metrics/crnn")
    init = f"{REALCE_ROOT}/basicsr/metrics/crnn/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "basicsr.metrics.crnn", init, submodule_search_locations=[f"{REALCE_ROOT}/basicsr/metrics/crnn"]
    )
    crnn_pkg = importlib.util.module_from_spec(spec)
    sys.modules["basicsr.metrics.crnn"] = crnn_pkg
    spec.loader.exec_module(crnn_pkg)

    psnr_ssim = load_file("basicsr.metrics.psnr_ssim", f"{REALCE_ROOT}/basicsr/metrics/psnr_ssim.py")
    recognition = load_file("basicsr.metrics.recognition", f"{REALCE_ROOT}/basicsr/metrics/recognition.py")
    return psnr_ssim, recognition


def read_annos(path):
    """Real-CE anno: per line 8 polygon coords + transcript (GBK encoded)."""
    lines = []
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("gbk")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def polys_from_annos(lines):
    polys, strs = [], []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 9:
            continue
        coords = [int(float(v)) for v in parts[:8]]
        polys.append(np.array(coords, dtype=np.int32).reshape(4, 2))
        strs.append(",".join(parts[8:]))
    return polys, strs


def build_mask(shape_hw, polys):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr_dir", required=True)
    ap.add_argument("--gt_dir", default="/data/ywk/datasets/Real-CE/val/52mm")
    ap.add_argument("--anno_dir", default="/data/ywk/datasets/Real-CE/val/annos")
    ap.add_argument("--list", default="/data/ywk/datasets/Real-CE/val/valid_list.txt")
    ap.add_argument("--crop_border", type=int, default=2)
    ap.add_argument("--no_rec", action="store_true", help="skip recognition ACC/NED")
    ap.add_argument("--no_lpips", action="store_true", help="skip LPIPS (full-res LPIPS needs ~3GB VRAM)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    psnr_ssim, recognition = _load_realce_metrics()
    if not args.no_rec:
        # Point the official loaders at the downloaded weights.
        recognition.opt["saved_model"] = f"{REALCE_ROOT}/basicsr/metrics/scene_base_CRNN.pth"
        recognition.opt["saved_model_eng"] = f"{REALCE_ROOT}/basicsr/metrics/crnn.pth"

    if args.list and os.path.isfile(args.list):
        names = [l.strip() for l in open(args.list) if l.strip()]
    else:
        names = sorted(os.listdir(args.gt_dir))

    agg = {k: [] for k in ["psnr", "ssim", "mask_psnr", "mask_ssim", "lpips"]}
    n_match, n_regions, ned_sum = 0, 0, 0.0
    missing = []

    for idx, name in enumerate(names):
        sr_path = os.path.join(args.sr_dir, name)
        if not os.path.isfile(sr_path):
            base = os.path.splitext(name)[0]
            hits = [f for f in os.listdir(args.sr_dir) if os.path.splitext(f)[0] == base]
            if not hits:
                missing.append(name)
                continue
            sr_path = os.path.join(args.sr_dir, hits[0])
        sr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(args.gt_dir, name), cv2.IMREAD_COLOR)
        if sr.shape != gt.shape:
            sr = cv2.resize(sr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        anno_path = os.path.join(args.anno_dir, os.path.splitext(name)[0] + ".txt")
        anno_lines = read_annos(anno_path) if os.path.isfile(anno_path) else []
        polys, _ = polys_from_annos(anno_lines)
        mask = build_mask(gt.shape[:2], polys)

        agg["psnr"].append(float(psnr_ssim.calculate_psnr(sr, gt, args.crop_border)))
        agg["ssim"].append(float(psnr_ssim.calculate_ssim(sr, gt, args.crop_border)))
        if not args.no_lpips:
            agg["lpips"].append(float(psnr_ssim.calculate_lpips(sr, gt, args.crop_border)))
        if mask.sum() > 0:
            agg["mask_psnr"].append(float(psnr_ssim.calculate_region_psnr(sr, gt, args.crop_border, mask)))
            agg["mask_ssim"].append(float(psnr_ssim.calculate_region_ssim(sr, gt, args.crop_border, mask)))

        if not args.no_rec and anno_lines:
            # official get_polys expects each anno wrapped in a list (anno[0])
            preds, gts = recognition.calculate_recognition_case_insensitive(
                sr, gt, args.crop_border, [[l] for l in anno_lines]
            )
            for p, g in zip(preds, gts):
                n_regions += 1
                if p == g:
                    n_match += 1
                max_len = max(len(p), len(g))
                ned_sum += (max_len - editdistance.eval(p, g)) / float(max_len + 1e-10)

        if (idx + 1) % 25 == 0:
            print(f"[{idx + 1}/{len(names)}] running psnr={np.mean(agg['psnr']):.3f}", flush=True)

    result = {k: float(np.mean(v)) for k, v in agg.items() if v}
    result["n_images"] = len(agg["psnr"])
    if missing:
        result["n_missing_sr"] = len(missing)
    if n_regions:
        result["rec_acc"] = n_match / n_regions
        result["rec_ned"] = ned_sum / n_regions
        result["n_regions"] = n_regions

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
