import os
import random
import numpy as np
import torch
import glob
import webdataset as wds
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T


class CenterCropTestTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image):
        if min(*pil_image.size) < self.image_size:
            scale = self.image_size / min(*pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        crop_y = (pil_image.height - self.image_size) // 2
        crop_x = (pil_image.width - self.image_size) // 2
        pil_image = pil_image.crop((
            crop_x, crop_y,
            crop_x + self.image_size, crop_y + self.image_size
        ))

        return pil_image


class RandomCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image):
        if min(pil_image.size) < self.image_size:
            scale = self.image_size / min(pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        max_x = pil_image.width - self.image_size
        max_y = pil_image.height - self.image_size
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)

        pil_image = pil_image.crop((
            crop_x, crop_y,
            crop_x + self.image_size, crop_y + self.image_size
        ))

        return pil_image


class RandomCropWithParams:
    """Like RandomCropTransform but also returns the transform params so callers
    can map original-image annotations (GT text boxes) into the crop's pixels."""

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image):
        scale = 1.0
        if min(pil_image.size) < self.image_size:
            scale = self.image_size / min(pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        max_x = pil_image.width - self.image_size
        max_y = pil_image.height - self.image_size
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)
        pil_image = pil_image.crop((
            crop_x, crop_y,
            crop_x + self.image_size, crop_y + self.image_size
        ))
        return pil_image, scale, crop_x, crop_y


def gt_boxes_collate_fn(batch):
    """Collate for the GT-box OCR-REPA path: stack hq, keep gt_boxes as a
    per-image list of [x1,y1,x2,y2] (variable length, matches the format that
    OCRRepaSystem._detect_boxes would otherwise produce)."""
    batch = [b for b in batch if b is not None]
    out = {"hq": torch.stack([b["hq"] for b in batch], dim=0)}
    if batch and "gt_boxes" in batch[0]:
        out["gt_boxes"] = [b["gt_boxes"] for b in batch]
    return out


class TxtPairDataset(Dataset):
    """
    Outputs HQ images as Tensors in [0,1]. Degradation is applied
    in the main training loop as a batched GPU operation.
    """
    def __init__(self, split=None, args=None):
        super().__init__()
        self.max_retry = 10000
        self.args = args
        self.to_tensor = transforms.ToTensor()

        self.gt_list = []
        self.split = split

        # Opt-in GT text-box path (experiment 2a-GT): use real annotated boxes
        # instead of the online DBNet detector. Default off -> behaves exactly as
        # before, so the online-detection runs are unaffected.
        self.use_gt_boxes = bool(getattr(args, 'ocr_use_gt_boxes', False)) and split == 'train'
        self.gt_boxes_index = None
        if self.use_gt_boxes:
            self.gt_crop = RandomCropWithParams(args.resolution)
            self.gt_min_box_size = int(getattr(args, 'ocr_repa_min_box_size', 8))
            idx_path = getattr(args, 'ocr_boxes_index_path', 'preset/text_boxes_index.pkl')
            import pickle
            with open(idx_path, 'rb') as f:
                self.gt_boxes_index = pickle.load(f)
            print(f'=====> loaded GT text-box index: {len(self.gt_boxes_index)} images from {idx_path}')

        if self.split == 'train':
            self.random_crop_preproc = transforms.Compose([
                RandomCropTransform(args.resolution),
            ])
            assert len(args.train_dataset_txt_paths_list) == len(args.train_dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list)):
                with open(args.train_dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.train_dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')
        elif self.split == 'test':
            self.test_center_crop_preproc = transforms.Compose([
                CenterCropTestTransform(args.resolution),
            ])
            assert len(args.test_dataset_txt_paths_list) == len(args.test_dataset_prob_paths_list)
            for idx_dataset in range(len(args.test_dataset_txt_paths_list)):
                with open(args.test_dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.test_dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def _load_rgb(self, path):
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")

    def _transform_boxes(self, path, scale, crop_x, crop_y):
        """Map original-image GT boxes into the current 512px crop. Returns a list
        of int [x1,y1,x2,y2] kept only if they survive clipping at >= min size."""
        stem = os.path.splitext(os.path.basename(path))[0]
        rec = self.gt_boxes_index.get(stem)
        if not rec:
            return []
        S = self.args.resolution
        out = []
        for (x1, y1, x2, y2) in rec["boxes"]:
            nx1 = min(max(x1 * scale - crop_x, 0.0), S)
            ny1 = min(max(y1 * scale - crop_y, 0.0), S)
            nx2 = min(max(x2 * scale - crop_x, 0.0), S)
            ny2 = min(max(y2 * scale - crop_y, 0.0), S)
            ix1, iy1, ix2, iy2 = int(nx1), int(ny1), int(round(nx2)), int(round(ny2))
            if (ix2 - ix1) >= self.gt_min_box_size and (iy2 - iy1) >= self.gt_min_box_size:
                out.append([ix1, iy1, ix2, iy2])
        return out

    def __getitem__(self, idx):
        for retry in range(self.max_retry):
            try:
                current_idx = (idx + retry) % len(self.gt_list)
                path = self.gt_list[current_idx]

                gt_img = self._load_rgb(path)
                if self.split == 'train' and self.use_gt_boxes:
                    gt_img, scale, crop_x, crop_y = self.gt_crop(gt_img)
                    boxes = self._transform_boxes(path, scale, crop_x, crop_y)
                    return {"hq": self.to_tensor(gt_img), "gt_boxes": boxes}
                if self.split == 'train':
                    gt_img = self.random_crop_preproc(gt_img)
                elif self.split == 'test':
                    gt_img = self.test_center_crop_preproc(gt_img)

                hq_tensor = self.to_tensor(gt_img)
                return {"hq": hq_tensor}

            except Exception as e:
                if retry == 0:
                    print(f"Warning: Failed to load {self.gt_list[(idx + retry) % len(self.gt_list)]}: {str(e)}")
                if retry == self.max_retry - 1:
                    raise RuntimeError(f"Failed to load data after {self.max_retry} retries. Last error: {str(e)}")
                continue

        raise RuntimeError(f"Unexpected error in __getitem__ for idx {idx}")


class DegradationMapper:
    """
    Reads images, applies geometric transforms, and outputs [0,1] Tensors.
    Degradation is handled in the GPU training loop.
    """
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.resolution = args.resolution

        if self.split == 'train':
            self.random_crop = RandomCropTransform(args.resolution)
        else:
            self.test_center_crop = CenterCropTestTransform(args.resolution)

        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        if not isinstance(sample, dict):
            return None

        img_key = None
        for k in sample.keys():
            if k in ("jpg", "png", "jpeg", "bmp", "webp"):
                img_key = k
                break
        if img_key is None:
            return None

        try:
            gt_img = sample[img_key]

            if isinstance(gt_img, np.ndarray):
                gt_img = Image.fromarray(gt_img)
            else:
                if gt_img.mode != 'RGB':
                    gt_img = gt_img.convert("RGB")

            if self.split == 'train':
                gt_img = self.random_crop(gt_img)
            elif self.split == 'test':
                gt_img = self.test_center_crop(gt_img)

            hq_tensor = self.to_tensor(gt_img)
            return {"hq": hq_tensor}

        except Exception:
            return None


def build_webdataset_pipeline(args, split='train'):
    if split == 'train':
        folders = args.train_dataset_txt_paths_list
        ratios = args.train_dataset_prob_paths_list
    else:
        folders = args.test_dataset_txt_paths_list
        ratios = args.test_dataset_prob_paths_list

    total_ratio = sum(ratios)
    probs = [r / total_ratio for r in ratios]

    shuffle_buffer = getattr(args, 'shuffle_buffer', 5000)

    datasets = []
    for folder in folders:
        tar_files = sorted(glob.glob(os.path.join(folder, "*.tar")))
        if not tar_files:
            print(f"Warning: No tar files found in {folder}")
            continue

        ds = wds.DataPipeline(
            wds.ResampledShards(tar_files, deterministic=False),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_buffer),
            wds.decode("rgb8", handler=wds.warn_and_continue),
        )
        datasets.append(ds)

    if not datasets:
        raise RuntimeError("No valid datasets created.")

    if len(datasets) > 1:
        mixed_loader = wds.RandomMix(datasets, probs)
    else:
        mixed_loader = datasets[0]

    mapper = DegradationMapper(args, split=split)

    loader = wds.DataPipeline(
        mixed_loader,
        wds.map(mapper),
        wds.select(lambda x: x is not None),
    )

    total_tars = sum([len(glob.glob(os.path.join(f, "*.tar"))) for f in folders])
    num_samples_est = total_tars * 100

    if split == 'train':
        loader = loader.with_length(1000000000)
    else:
        loader = loader.with_length(num_samples_est)

    return loader
