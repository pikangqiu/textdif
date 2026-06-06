from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file


VAE_LORA_ADAPTER_NAME = "default_encoder"
VAE_LORA_FILENAME = "vae_encoder_lora.safetensors"
_CODSR_VAE_TARGET_PATTERNS = (
    "conv1",
    "conv2",
    "conv_in",
    "conv_shortcut",
    "conv",
    "conv_out",
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
)


def normalize_target_modules(target_modules):
    normalized = []
    seen = set()
    for module_name in target_modules or []:
        name = str(module_name).strip()
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        elif name.endswith(".bias"):
            name = name[: -len(".bias")]
        if name.endswith(".base_layer"):
            name = name[: -len(".base_layer")]
        lora_marker = name.find(".lora_")
        if lora_marker != -1:
            name = name[:lora_marker]
        name = name.rstrip(".")
        if name and name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def collect_vae_encoder_lora_targets(vae):
    """Match the VAE encoder target selection used by CODSR."""
    target_modules = []
    for name, _ in vae.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        if "quant_conv" in name and "post_quant_conv" not in name:
            target_modules.append(name.replace(".weight", ""))
            continue
        if "encoder" not in name:
            continue
        if any(pattern in name for pattern in _CODSR_VAE_TARGET_PATTERNS):
            target_modules.append(name.replace(".weight", ""))
    return normalize_target_modules(target_modules)


def get_vae_lora_state_dict(vae):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in vae.state_dict().items()
        if "lora_" in name
    }


def save_vae_lora_weights(vae, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = get_vae_lora_state_dict(vae)
    if not state_dict:
        raise RuntimeError("No VAE LoRA parameters were found to save.")
    save_file(state_dict, str(output_path))
    return output_path


def load_vae_lora_weights(vae, weight_path):
    weight_path = Path(weight_path)
    state_dict = load_file(str(weight_path))
    model_keys = set(vae.state_dict())
    missing_from_model = sorted(set(state_dict) - model_keys)
    if missing_from_model:
        raise RuntimeError(
            f"VAE LoRA checkpoint contains {len(missing_from_model)} unknown keys; "
            f"first key: {missing_from_model[0]}"
        )
    vae.load_state_dict(state_dict, strict=False)
    return len(state_dict)


def configure_vae_encoder_lora(
    vae,
    rank=4,
    weight_path=None,
    trainable=False,
    adapter_name=VAE_LORA_ADAPTER_NAME,
):
    if not hasattr(vae, "add_adapter"):
        raise TypeError("The selected VAE does not expose Diffusers PEFT adapter APIs.")
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "VAE encoder LoRA requires PEFT. Install the repository requirements "
            "or run `pip install peft==0.17.0`."
        ) from exc

    target_modules = collect_vae_encoder_lora_targets(vae)
    if not target_modules:
        raise RuntimeError("No CODSR-compatible VAE encoder LoRA targets were found.")

    vae.add_adapter(
        LoraConfig(
            r=int(rank),
            init_lora_weights="gaussian",
            target_modules=target_modules,
        ),
        adapter_name=adapter_name,
    )
    vae.set_adapter([adapter_name])
    vae.requires_grad_(False)
    if weight_path is not None:
        load_vae_lora_weights(vae, weight_path)
    if trainable:
        for name, parameter in vae.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
    return target_modules


def vae_lora_parameters(vae):
    return [
        parameter
        for name, parameter in vae.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]


@contextmanager
def adapters_disabled(vae):
    if not hasattr(vae, "disable_adapters") or not hasattr(vae, "enable_adapters"):
        yield
        return
    vae.disable_adapters()
    try:
        yield
    finally:
        vae.enable_adapters()


def _encode_latent(model_ae, image, ae_type, latents_mean=None, latents_std=None):
    if ae_type == "qwen":
        return (model_ae.encode(image).latent_dist.sample() - latents_mean) * latents_std
    if ae_type == "sd2":
        return (
            model_ae.encode(image.to(model_ae.dtype)).latent_dist.sample()
            * model_ae.config.scaling_factor
        )
    raise ValueError(f"Unsupported ae_type: {ae_type}")


def encode_lq_hq_latents(
    model_ae,
    lq,
    hq,
    ae_type,
    *,
    use_vae_encoder_lora=False,
    train_lq_adapter=False,
    latents_mean=None,
    latents_std=None,
):
    if not use_vae_encoder_lora:
        with torch.no_grad():
            combined = torch.cat([lq, hq], dim=0)
            combined_latent = _encode_latent(
                model_ae,
                combined,
                ae_type,
                latents_mean=latents_mean,
                latents_std=latents_std,
            )
        return combined_latent.chunk(2, dim=0)

    if ae_type != "sd2":
        raise ValueError("VAE encoder LoRA is currently supported only for ae_type=sd2.")

    lq_grad_context = nullcontext() if train_lq_adapter else torch.no_grad()
    with lq_grad_context:
        lq_latent = _encode_latent(model_ae, lq, ae_type)
    with adapters_disabled(model_ae), torch.no_grad():
        hq_latent = _encode_latent(model_ae, hq, ae_type)
    return lq_latent, hq_latent


class VAEEncoderLoRAWrapper(nn.Module):
    """DDP-compatible forward wrapper for the trainable VAE encoder adapter."""

    def __init__(self, vae, ae_type):
        super().__init__()
        self.vae = vae
        self.ae_type = ae_type

    def forward(self, lq, hq):
        return encode_lq_hq_latents(
            self.vae,
            lq,
            hq,
            self.ae_type,
            use_vae_encoder_lora=True,
            train_lq_adapter=True,
        )


def resolve_vae_lora_path(checkpoint_path, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return path

    checkpoint_path = Path(checkpoint_path)
    starts = [checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent]
    starts.extend(starts[0].parents)
    candidates = []
    for start in starts:
        candidates.extend(
            [
                start / VAE_LORA_FILENAME,
                start / "clean_weights" / VAE_LORA_FILENAME,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
