"""Tests for the local-crop OCR-REPA loss (models/ocr_repa.py).

These require the vendored PaddleOCR2Pytorch weights
(ch_ptocr_server_v2.0_{det,rec}_infer.pth) to be present; they are skipped
otherwise. They run on CPU to avoid contending with training GPUs.
"""

import os
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.ocr_repa import _resolve_default_paddle_root, build_ocr_repa_system


_PADDLE_ROOT = _resolve_default_paddle_root()
_DET = os.path.join(_PADDLE_ROOT, "weights", "ch_ptocr_server_v2.0_det_infer.pth")
_REC = os.path.join(_PADDLE_ROOT, "weights", "ch_ptocr_server_v2.0_rec_infer.pth")
_HAVE_WEIGHTS = os.path.isfile(_DET) and os.path.isfile(_REC)

pytestmark = pytest.mark.skipif(
    not _HAVE_WEIGHTS, reason="PaddleOCR2Pytorch det/rec weights not available"
)


def _text_image(size=512):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for y in (80, 180, 280, 380):
        draw.text((40, y), "HELLO WORLD 12345", fill=(240, 240, 240))
    arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return (arr * 2 - 1).unsqueeze(0)  # m11, (1,3,H,W)


@pytest.fixture(scope="module")
def system():
    args = types.SimpleNamespace(ocr_repa_weight=0.5)
    return build_ocr_repa_system(args, device="cpu")


def test_detects_text_boxes(system):
    hr = _text_image()
    boxes = system._detect_boxes(hr)
    assert len(boxes) == 1
    assert len(boxes[0]) > 0  # found at least one text region


def test_loss_is_differentiable_wrt_prediction(system):
    hr = _text_image()
    pred = torch.nn.functional.avg_pool2d(hr, 5, 1, 2).clone().detach().requires_grad_(True)
    loss, logs = system.compute_loss(pred, hr)
    assert loss.item() > 0
    assert logs["ocr_repa/n_boxes"] > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0  # gradient actually flows back to the image


def test_no_text_returns_zero_without_crashing(system):
    blank = torch.zeros(1, 3, 256, 256)
    pred = torch.zeros(1, 3, 256, 256, requires_grad=True)
    loss, logs = system.compute_loss(pred, blank)
    assert loss.item() == 0.0
    assert logs["ocr_repa/n_boxes"] == 0.0


def test_ctc_loss_is_differentiable_wrt_prediction(system):
    hr = _text_image()
    pred = torch.nn.functional.avg_pool2d(hr, 5, 1, 2).clone().detach().requires_grad_(True)
    loss, logs = system.compute_ctc_loss(pred, hr)
    assert logs["ocr_ctc/n_boxes"] > 0
    assert logs["ocr_ctc/n_used"] > 0
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_ctc_no_text_returns_zero(system):
    blank = torch.zeros(1, 3, 256, 256)
    pred = torch.zeros(1, 3, 256, 256, requires_grad=True)
    loss, logs = system.compute_ctc_loss(pred, blank)
    assert loss.item() == 0.0
    assert logs["ocr_ctc/n_boxes"] == 0.0


def test_2b_does_not_perturb_2a(system):
    """Isolation guarantee: computing the CTC loss (2b) must not change the 2a
    REPA loss for the same inputs (no shared state, no RNG)."""
    hr = _text_image()
    pred = torch.nn.functional.avg_pool2d(hr, 5, 1, 2).clone().detach()

    repa_before, _ = system.compute_loss(pred.clone(), hr)
    _ = system.compute_ctc_loss(pred.clone().requires_grad_(True), hr)
    repa_after, _ = system.compute_loss(pred.clone(), hr)
    assert torch.allclose(repa_before, repa_after, atol=0, rtol=0)
