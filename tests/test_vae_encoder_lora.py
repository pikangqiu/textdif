from pathlib import Path

import torch
from torch import nn

from models.vae_encoder_lora import (
    adapters_disabled,
    collect_vae_encoder_lora_targets,
    get_vae_lora_state_dict,
    resolve_vae_lora_path,
)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv2d(3, 4, 3)
        self.encoder.norm_out = nn.GroupNorm(1, 4)
        self.encoder.mid_block = nn.Module()
        self.encoder.mid_block.to_q = nn.Linear(4, 4)
        self.decoder = nn.Module()
        self.decoder.conv_in = nn.Conv2d(4, 4, 3)
        self.quant_conv = nn.Conv2d(4, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.adapters_are_disabled = False

    def disable_adapters(self):
        self.adapters_are_disabled = True

    def enable_adapters(self):
        self.adapters_are_disabled = False


def test_collect_targets_matches_codsr_encoder_rules():
    vae = TinyVAE()

    targets = collect_vae_encoder_lora_targets(vae)

    assert "encoder.conv_in" in targets
    assert "encoder.mid_block.to_q" in targets
    assert "quant_conv" in targets
    assert "encoder.norm_out" not in targets
    assert "decoder.conv_in" not in targets
    assert "post_quant_conv" not in targets
    assert len(targets) == len(set(targets))


def test_get_state_dict_only_returns_lora_parameters():
    module = nn.Module()
    module.base = nn.Linear(2, 2)
    module.lora_A = nn.Linear(2, 1, bias=False)
    module.lora_B = nn.Linear(1, 2, bias=False)

    state = get_vae_lora_state_dict(module)

    assert state
    assert all("lora_" in name for name in state)
    assert all(tensor.device.type == "cpu" for tensor in state.values())


def test_adapters_disabled_restores_previous_enabled_state():
    vae = TinyVAE()

    with adapters_disabled(vae):
        assert vae.adapters_are_disabled

    assert not vae.adapters_are_disabled


def test_resolve_lora_path_finds_clean_weight_next_to_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "checkpoint-00020000"
    clean_weights = checkpoint / "clean_weights"
    clean_weights.mkdir(parents=True)
    lora_path = clean_weights / "vae_encoder_lora.safetensors"
    lora_path.touch()

    assert resolve_vae_lora_path(checkpoint) == lora_path
    assert resolve_vae_lora_path(clean_weights / "ema_model.safetensors") == lora_path


def test_resolve_lora_path_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "custom.safetensors"
    explicit.touch()

    assert resolve_vae_lora_path(tmp_path / "missing-checkpoint", explicit) == explicit


def test_resolve_lora_path_falls_back_when_saved_explicit_path_is_stale(tmp_path):
    checkpoint = tmp_path / "checkpoint-00020000"
    clean_weights = checkpoint / "clean_weights"
    clean_weights.mkdir(parents=True)
    local_lora = clean_weights / "vae_encoder_lora.safetensors"
    local_lora.touch()

    assert (
        resolve_vae_lora_path(checkpoint, "/old/server/vae_encoder_lora.safetensors")
        == local_lora
    )
