import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ChannelwiseSobel(nn.Module):
    """CODSR-style channelwise Sobel magnitude."""

    def __init__(self, mode="fast"):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        )
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.mode = mode

    def forward(self, x):
        kernel_x = self.kernel_x.repeat(x.size(1), 1, 1, 1).to(device=x.device, dtype=x.dtype)
        kernel_y = self.kernel_y.repeat(x.size(1), 1, 1, 1).to(device=x.device, dtype=x.dtype)

        padded_x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        padded_y = F.pad(x, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded_x, kernel_x, padding=0, groups=x.size(1))
        grad_y = F.conv2d(padded_y, kernel_y, padding=0, groups=x.size(1))

        if self.mode == "accurate":
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        else:
            magnitude = torch.abs(grad_x) + torch.abs(grad_y)

        return torch.mean(magnitude, dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)


class SFTLayer(nn.Module):
    """CODSR SFT layer: conv -> SiLU -> zero-initialized conv for scale/shift."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.SFT_scale_conv1 = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.SFT_shift_conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.SFT_shift_conv1 = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.conv_act = nn.SiLU()

    def forward(self, cond):
        scale = self.SFT_scale_conv1(self.conv_act(self.SFT_scale_conv0(cond)))
        shift = self.SFT_shift_conv1(self.conv_act(self.SFT_shift_conv0(cond)))
        return scale, shift


class LQTokenModulation(nn.Module):
    """Adapt CODSR LQ SFT modulation from UNet conv features to DiT tokens."""

    def __init__(self, hidden_size, downscale_factor=8):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.sft = SFTLayer(3 * downscale_factor * downscale_factor, hidden_size)

    def forward(self, tokens, lq_image):
        if lq_image is None:
            return tokens
        if lq_image.ndim != 4 or lq_image.size(1) != 3:
            raise ValueError(f"lq_image must be BCHW RGB tensor, got shape={tuple(lq_image.shape)}")

        b, n, c = tokens.shape
        h = w = int(n**0.5)
        if h * w != n:
            raise ValueError(f"DiT token count must be square for LQ modulation, got n={n}")

        cond = self.unshuffle(lq_image.to(device=tokens.device, dtype=tokens.dtype))
        scale, shift = self.sft(cond)
        if scale.shape[-2:] != (h, w):
            scale = F.interpolate(scale, size=(h, w), mode="bilinear", align_corners=False)
            shift = F.interpolate(shift, size=(h, w), mode="bilinear", align_corners=False)

        scale = scale.flatten(2).transpose(1, 2)
        shift = shift.flatten(2).transpose(1, 2)
        return tokens * (scale + 1) + shift


def rgb_to_gray(rgb_map):
    r, g, b = rgb_map[:, 0], rgb_map[:, 1], rgb_map[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.unsqueeze(1)


def graygrad_to_weight_patchwise_sobel(grad_map, target_hw8=None):
    """CODSR gradient-to-noise-weight mapping."""
    b, c, h, w = grad_map.shape
    device = grad_map.device
    dtype = grad_map.dtype
    block_size = 16

    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h or pad_w:
        grad_map = F.pad(grad_map, (0, pad_w, 0, pad_h), mode="replicate")

    patch_avg = F.avg_pool2d(grad_map, kernel_size=block_size, stride=block_size)
    result_patch = torch.zeros_like(patch_avg)
    low_mask = patch_avg <= 0.15
    mid_mask = (patch_avg > 0.15) & (patch_avg <= 0.25)
    high_mask = patch_avg > 0.25

    result_patch[low_mask] = 0.3
    result_patch[mid_mask] = 7.0 * (patch_avg[mid_mask] - 0.15) + 0.3
    result_patch[high_mask] = 1.0

    expanded_weight = result_patch.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    if target_hw8 is not None:
        h8, w8 = target_hw8
    else:
        h8, w8 = h // 8, w // 8
    expanded_weight = expanded_weight[:, :, :h8, :w8]
    return expanded_weight.to(device=device, dtype=dtype)


def build_ragp_weight(lq_image_m11, target_hw8=None, sobel_layer=None):
    """Build CODSR RAGP latent noise scale from an LQ image in [-1, 1]."""
    if sobel_layer is None:
        sobel_layer = ChannelwiseSobel(mode="fast").to(lq_image_m11.device, dtype=lq_image_m11.dtype)
    gray_map = rgb_to_gray(lq_image_m11 * 0.5 + 0.5)
    gradient_result = sobel_layer(gray_map)
    return graygrad_to_weight_patchwise_sobel(gradient_result, target_hw8=target_hw8)
