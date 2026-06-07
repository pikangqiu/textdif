"""Local-crop OCR-REPA loss (text-region-aware representation alignment).

Motivation
----------
The existing ``repa_type: ocr`` path feeds the *whole* image through a frozen
OCR encoder (TrOCR) and then mean-pools every token into a single global
vector before the cosine alignment.  Text information is local and glyph-level,
so global pooling throws away exactly the spatial structure that matters for
readability.  This module replaces that with the "correct" formulation for
text: detect text boxes on the (clean) HR image, crop the *same* boxes out of
both the student's one-step prediction and the HR target, and align the frozen
recognizer's per-region sequence features.

It is structured like ``models/fd_loss.py``: a frozen ``nn.Module`` system that
the distillation loop calls with the decoded one-step prediction and the HR
image, gated by ``start_step`` / ``interval``.

Pipeline (per training step where active)::

    HR image (m11)  --det(no_grad)-->  text boxes
    pred image (m11) ┐
    HR image  (m11)  ┴ crop same boxes -> resize -> frozen rec backbone+neck
    loss = 1 - cosine( feat(pred_crops), feat(HR_crops).detach() )

The OCR detector/recognizer come from the vendored ``PaddleOCR2Pytorch`` and
are fully frozen; gradients only flow *through* the recognizer back into the
predicted image (and hence the student).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _no_autocast():
    """Run the frozen OCR det/rec in full precision.

    The loss is computed under the trainer's ``accelerator.autocast()`` (bf16),
    which makes PaddleOCR's internal convs emit bf16 tensors; its numpy-based
    postprocess then dies with "unsupported ScalarType BFloat16". Gradients to
    the prediction image still flow with autocast disabled.
    """
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    else:
        yield


logger = logging.getLogger("ocr_repa")

# m11 normalization (2*x01 - 1) equals PaddleOCR's rec normalization (x01-0.5)/0.5,
# so an image already in [-1, 1] can be fed to the recognizer directly.


@dataclass
class OCRRepaConfig:
    paddle_root: str
    det_model_path: str
    rec_model_path: str
    det_yaml_path: str | None = None
    rec_yaml_path: str | None = None
    rec_algorithm: str = "CRNN"
    rec_image_shape: tuple = (3, 32, 320)
    det_limit_side_len: float = 960.0
    det_db_box_thresh: float = 0.6
    max_boxes: int = 16
    min_box_size: int = 8
    feat_key: str = "neck_out"
    loss_type: str = "cosine"  # "cosine" | "mse"
    bgr_input: bool = True


def _resolve_default_paddle_root() -> str:
    # models/ocr_repa.py -> repo root -> PaddleOCR2Pytorch
    return str(Path(__file__).resolve().parent.parent / "PaddleOCR2Pytorch")


def _ensure_paddle_on_path(paddle_root: str) -> None:
    paddle_root = str(Path(paddle_root).resolve())
    if not os.path.isdir(paddle_root):
        raise FileNotFoundError(f"PaddleOCR2Pytorch root not found: {paddle_root}")
    if paddle_root not in sys.path:
        sys.path.insert(0, paddle_root)


def _build_predictor_args(config: OCRRepaConfig, use_gpu: bool):
    """Build the argparse-style namespace expected by Text{Detector,Recognizer}."""
    from tools.infer import pytorchocr_utility as utility

    args = utility.init_args().parse_args([])
    args.use_gpu = use_gpu
    # detection
    args.det_algorithm = "DB"
    args.det_model_path = config.det_model_path
    args.det_yaml_path = config.det_yaml_path
    args.det_limit_side_len = config.det_limit_side_len
    args.det_db_box_thresh = config.det_db_box_thresh
    args.det_box_type = "quad"
    # recognition
    args.rec_algorithm = config.rec_algorithm
    args.rec_model_path = config.rec_model_path
    args.rec_yaml_path = config.rec_yaml_path
    args.rec_image_shape = ",".join(str(int(v)) for v in config.rec_image_shape)
    return args


class OCRRepaSystem(nn.Module):
    """Frozen local-crop OCR representation alignment."""

    def __init__(self, config: OCRRepaConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        _ensure_paddle_on_path(config.paddle_root)

        from tools.infer.predict_det import TextDetector
        from tools.infer.predict_rec import TextRecognizer

        use_gpu = torch.cuda.is_available()
        pred_args = _build_predictor_args(config, use_gpu)

        # Detector is only used under no_grad to produce boxes.
        self.detector = TextDetector(pred_args)
        # Recognizer: we only need its differentiable backbone(+neck).
        self.recognizer = TextRecognizer(pred_args)

        self.rec_net = self.recognizer.net
        self.rec_net.eval()
        for p in self.rec_net.parameters():
            p.requires_grad_(False)
        # The detector net is likewise frozen.
        self.detector.net.eval()
        for p in self.detector.net.parameters():
            p.requires_grad_(False)

        self._rec_dtype = next(self.rec_net.parameters()).dtype
        # CTC blank is index 0 in PaddleOCR recognizers (see CTCLabelDecode).
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # ---- detection -------------------------------------------------------
    @torch.no_grad()
    def _detect_boxes(self, hr_image_m11):
        """Return a list (len B) of int boxes [[x1, y1, x2, y2], ...] in HR pixels."""
        b, _, h, w = hr_image_m11.shape
        imgs01 = (hr_image_m11.detach().float() * 0.5 + 0.5).clamp(0, 1)
        boxes_per_image = []
        for i in range(b):
            rgb = (imgs01[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            bgr = rgb[:, :, ::-1].copy()  # PaddleOCR expects BGR (cv2 order)
            try:
                with _no_autocast():
                    dt_boxes, _ = self.detector(bgr)
            except Exception as exc:  # detection must never crash training
                logger.warning("OCR-REPA detection failed on sample %d: %s", i, exc)
                dt_boxes = None
            boxes_per_image.append(self._quads_to_int_boxes(dt_boxes, h, w))
        return boxes_per_image

    def _quads_to_int_boxes(self, dt_boxes, h, w):
        boxes = []
        if dt_boxes is None or len(dt_boxes) == 0:
            return boxes
        areas = []
        cand = []
        for quad in dt_boxes:
            quad = np.asarray(quad).reshape(-1, 2)
            x1 = int(np.floor(quad[:, 0].min()))
            y1 = int(np.floor(quad[:, 1].min()))
            x2 = int(np.ceil(quad[:, 0].max()))
            y2 = int(np.ceil(quad[:, 1].max()))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                continue
            cand.append([x1, y1, x2, y2])
            areas.append((x2 - x1) * (y2 - y1))
        if not cand:
            return boxes
        # keep the largest boxes (most reliable / most legible regions)
        order = np.argsort(areas)[::-1][: self.config.max_boxes]
        return [cand[i] for i in order]

    # ---- cropping --------------------------------------------------------
    def _crop_and_stack(self, image, boxes_per_image):
        """Differentiable crop of every box; resize to rec shape; stack to (N,3,H,W)."""
        _, imgH, imgW = self.config.rec_image_shape
        crops = []
        for i, boxes in enumerate(boxes_per_image):
            for (x1, y1, x2, y2) in boxes:
                crop = image[i : i + 1, :, y1:y2, x1:x2]
                crop = F.interpolate(
                    crop.float(), size=(imgH, imgW), mode="bilinear", align_corners=False
                )
                crops.append(crop)
        if not crops:
            return None
        return torch.cat(crops, dim=0)

    # ---- features --------------------------------------------------------
    def _rec_features(self, crops_m11):
        x = crops_m11
        if self.config.bgr_input:
            x = x[:, [2, 1, 0], :, :]  # RGB -> BGR to match the pretrained recognizer
        x = x.to(self._rec_dtype)
        # cuDNN's RNN backward refuses to run while the (frozen) recognizer is in
        # eval mode; disable cuDNN so the native LSTM, which supports eval-mode
        # backward, is used and gradients flow back to the prediction crops.
        with _no_autocast(), torch.backends.cudnn.flags(enabled=False):
            feat = self.rec_net.backbone(x)
            if getattr(self.rec_net, "use_neck", False):
                feat = self.rec_net.neck(feat)
        if isinstance(feat, dict):
            feat = feat.get(self.config.feat_key, next(iter(feat.values())))
        if feat.ndim == 4:  # (N, C, H, W) -> (N, H*W, C)
            feat = feat.flatten(2).transpose(1, 2)
        return feat  # (N, T, C)

    # ---- public API ------------------------------------------------------
    def _prepare_given_boxes(self, boxes_per_image, h, w):
        """Validate/clip externally provided GT boxes and cap to max_boxes (keeps
        the largest), so they match the format produced by the detector path."""
        out = []
        for boxes in boxes_per_image:
            cand, areas = [], []
            for box in (boxes or []):
                x1, y1, x2, y2 = (int(v) for v in box)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                    continue
                cand.append([x1, y1, x2, y2])
                areas.append((x2 - x1) * (y2 - y1))
            if cand:
                order = np.argsort(areas)[::-1][: self.config.max_boxes]
                out.append([cand[i] for i in order])
            else:
                out.append([])
        return out

    def compute_loss(self, pred_image_m11, hr_image_m11, boxes_per_image=None):
        """Return (loss, logs). loss is a zero scalar (no grad) when no text found.

        If ``boxes_per_image`` is given (a per-image list of [x1,y1,x2,y2] in HR
        pixels, e.g. GT annotations), the online detector is skipped."""
        if pred_image_m11.shape[-2:] != hr_image_m11.shape[-2:]:
            pred_image_m11 = F.interpolate(
                pred_image_m11.float(),
                size=hr_image_m11.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if boxes_per_image is None:
            boxes_per_image = self._detect_boxes(hr_image_m11)
        else:
            h, w = hr_image_m11.shape[-2:]
            boxes_per_image = self._prepare_given_boxes(boxes_per_image, h, w)
        n_boxes = sum(len(b) for b in boxes_per_image)
        if n_boxes == 0:
            zero = pred_image_m11.sum() * 0.0
            return zero, {"ocr_repa/loss": 0.0, "ocr_repa/n_boxes": 0.0}

        pred_crops = self._crop_and_stack(pred_image_m11, boxes_per_image)
        with torch.no_grad():
            hr_crops = self._crop_and_stack(hr_image_m11.detach(), boxes_per_image)

        pred_feat = self._rec_features(pred_crops)
        with torch.no_grad():
            target_feat = self._rec_features(hr_crops).detach()

        if self.config.loss_type == "mse":
            loss = F.mse_loss(pred_feat.float(), target_feat.float())
        else:  # token-wise cosine, averaged over the sequence and the regions
            p = F.normalize(pred_feat.float(), dim=-1)
            t = F.normalize(target_feat.float(), dim=-1)
            loss = 1.0 - (p * t).sum(dim=-1).mean()

        logs = {
            "ocr_repa/loss": float(loss.detach()),
            "ocr_repa/n_boxes": float(n_boxes),
        }
        return loss, logs

    # ---- CTC recognition loss (2b) --------------------------------------
    def _rec_logits(self, crops_m11):
        """Raw CTC logits (N, T, num_classes); the head softmax is bypassed."""
        feat = self._rec_features(crops_m11)  # (N, T, C_neck)
        head = self.rec_net.head
        if getattr(head, "mid_channels", None) is None:
            logits = head.fc(feat)
        else:
            logits = head.fc2(head.fc1(feat))
        return logits

    @torch.no_grad()
    def _ctc_greedy_targets(self, logits):
        """Greedy CTC decode -> per-sample index sequences (blank=0, repeats collapsed)."""
        best = logits.argmax(dim=-1)  # (N, T)
        targets, lengths = [], []
        for seq in best.tolist():
            prev = -1
            out = []
            for idx in seq:
                if idx != 0 and idx != prev:
                    out.append(idx)
                prev = idx
            targets.append(out)
            lengths.append(len(out))
        return targets, lengths

    def compute_ctc_loss(self, pred_image_m11, hr_image_m11):
        """CTC loss of the prediction against HR-derived pseudo transcripts.

        Returns (loss, logs); a zero scalar (no grad) when no decodable text.
        """
        if pred_image_m11.shape[-2:] != hr_image_m11.shape[-2:]:
            pred_image_m11 = F.interpolate(
                pred_image_m11.float(),
                size=hr_image_m11.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        boxes_per_image = self._detect_boxes(hr_image_m11)
        n_boxes = sum(len(b) for b in boxes_per_image)
        zero_logs = {"ocr_ctc/loss": 0.0, "ocr_ctc/n_boxes": 0.0, "ocr_ctc/n_used": 0.0}
        if n_boxes == 0:
            return pred_image_m11.sum() * 0.0, zero_logs

        with torch.no_grad():
            hr_crops = self._crop_and_stack(hr_image_m11.detach(), boxes_per_image)
            hr_logits = self._rec_logits(hr_crops)
            targets, target_lengths = self._ctc_greedy_targets(hr_logits)

        keep = [i for i, length in enumerate(target_lengths) if length > 0]
        if not keep:
            return pred_image_m11.sum() * 0.0, zero_logs

        pred_crops = self._crop_and_stack(pred_image_m11, boxes_per_image)
        pred_logits = self._rec_logits(pred_crops)[keep]  # (Nk, T, K)
        log_probs = F.log_softmax(pred_logits.float(), dim=-1).transpose(0, 1)  # (T, Nk, K)

        timesteps, n_used = log_probs.shape[0], log_probs.shape[1]
        device = log_probs.device
        flat_targets = torch.tensor(
            [idx for i in keep for idx in targets[i]], dtype=torch.long, device=device
        )
        tgt_lengths = torch.tensor([target_lengths[i] for i in keep], dtype=torch.long, device=device)
        # CTC needs input_length >= target_length; clamp pathological long targets.
        tgt_lengths = torch.clamp(tgt_lengths, max=timesteps)
        in_lengths = torch.full((n_used,), timesteps, dtype=torch.long, device=device)

        loss = self.ctc_loss(log_probs, flat_targets, in_lengths, tgt_lengths)
        logs = {
            "ocr_ctc/loss": float(loss.detach()),
            "ocr_ctc/n_boxes": float(n_boxes),
            "ocr_ctc/n_used": float(n_used),
        }
        return loss, logs


def ocr_repa_enabled(args) -> bool:
    return float(getattr(args, "ocr_repa_weight", 0.0)) > 0.0


def ocr_ctc_enabled(args) -> bool:
    return float(getattr(args, "ocr_ctc_weight", 0.0)) > 0.0


def build_ocr_repa_system(args, device) -> OCRRepaSystem:
    paddle_root = getattr(args, "ocr_repa_paddle_root", None) or _resolve_default_paddle_root()
    default_det = os.path.join(paddle_root, "weights", "ch_ptocr_server_v2.0_det_infer.pth")
    default_rec = os.path.join(paddle_root, "weights", "ch_ptocr_server_v2.0_rec_infer.pth")

    shape_str = str(getattr(args, "ocr_repa_rec_image_shape", "3,32,320"))
    rec_image_shape = tuple(int(v) for v in shape_str.replace(" ", "").split(","))

    config = OCRRepaConfig(
        paddle_root=paddle_root,
        det_model_path=str(getattr(args, "ocr_repa_det_model_path", None) or default_det),
        rec_model_path=str(getattr(args, "ocr_repa_rec_model_path", None) or default_rec),
        det_yaml_path=getattr(args, "ocr_repa_det_yaml_path", None),
        rec_yaml_path=getattr(args, "ocr_repa_rec_yaml_path", None),
        rec_algorithm=str(getattr(args, "ocr_repa_rec_algorithm", "CRNN")),
        rec_image_shape=rec_image_shape,
        det_limit_side_len=float(getattr(args, "ocr_repa_det_limit_side_len", 960.0)),
        det_db_box_thresh=float(getattr(args, "ocr_repa_det_db_box_thresh", 0.6)),
        max_boxes=int(getattr(args, "ocr_repa_max_boxes", 16)),
        min_box_size=int(getattr(args, "ocr_repa_min_box_size", 8)),
        feat_key=str(getattr(args, "ocr_repa_feat_key", "neck_out")),
        loss_type=str(getattr(args, "ocr_repa_loss_type", "cosine")),
        bgr_input=bool(getattr(args, "ocr_repa_bgr_input", True)),
    )
    return OCRRepaSystem(config, device=device)
