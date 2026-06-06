"""FD-Loss components adapted from Jiawei-Yang/FD-Loss.

This module keeps the original FD-Loss training recipe:
- frozen representation model judges,
- reference real-image mu/sigma loaded from .npz,
- generated feature queue,
- full-covariance differentiable Frechet distance,
- normalized loss: fid / (fid.detach() + eps).

The VOSR training loop supplies generated SR images in [-1, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("FD_loss")


class _DiffAllGather(torch.autograd.Function):
    """All-gather that preserves gradients for the local chunk."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        ctx.rank = torch.distributed.get_rank()
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = ctx.batch_size
        return grad_output[ctx.rank * chunk : (ctx.rank + 1) * chunk].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return tensor
    return _DiffAllGather.apply(tensor)


class FeatureQueue(nn.Module):
    """Circular feature buffer used by FD-Loss."""

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = int(size)
        self.feat_dim = int(feat_dim)
        self.online_accum = bool(online_accum)
        self.ema_beta = float(ema_beta)
        self.ema_stats = self.ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        if not hasattr(self, "ptr"):
            return 0
        return int(self.ptr.item())

    @torch.no_grad()
    def _init_accumulators(self):
        feats_d = self.feats.double()
        self.feat_sum_old.copy_(feats_d.sum(0))
        self.feat_outer_old.copy_(feats_d.T @ feats_d)

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        feats_d = feats.detach().float().double()
        self.mu_ema.add_(feats_d.sum(0))
        self.m2_ema.addmm_(feats_d.T, feats_d)
        self._ema_count += feats_d.shape[0]

    @torch.no_grad()
    def _finalize_streaming_init(self):
        count = self._ema_count.item()
        if count == 0:
            logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
            return
        self.mu_ema.div_(count)
        self.m2_ema.div_(count)

    def build_feats_stats(self, new_feats: torch.Tensor):
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        bsz = new_d.shape[0]
        n = self.size
        if n <= 1:
            raise ValueError("FeatureQueue stats mode requires queue size > 1.")

        evicted = self._get_evicted_feats(bsz).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / n
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / n) / (n - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        beta = self.ema_beta
        new_d = new_feats.double()
        bsz = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / bsz
        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return new
        n = new.shape[0]
        if n >= self.size:
            return new[-self.size :]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr : ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr : self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        return self._snapshot(self.feats, new_feats)

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        if self.size == 0:
            return

        n = new_feats.shape[0]
        new_det = new_feats.detach().float()
        if n >= self.size:
            new_det = new_det[-self.size :]
            n = new_det.shape[0]

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            return

        ptr = self.pointer
        if self.online_accum:
            evicted = self._get_evicted_feats(n).double()
            new_d = new_det.double()
            self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
            self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)

        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr : self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]
        self.ptr[0] = (ptr + n) % self.size

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        if n >= self.size:
            return self.feats
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr : ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr : self.size], self.feats[: n - first]], dim=0)


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if sigma_ref_sqrt is not None:
        mat = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        mat = 0.5 * (mat + mat.T)
        evals = torch.linalg.eigvalsh(mat)
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return None
        eigvals = torch.linalg.eigvals(product).real
        eigvals = torch.clamp(eigvals, min=0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))
    return torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * tr_covmean


def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor:
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            return torch.tensor(1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True)
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma")

    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)
    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        device = all_feats.device if all_feats is not None else mu.device
        return torch.tensor(1e6, device=device, dtype=torch.float32)
    return (mean_term + trace_term).float()


def load_mu_and_sigma_reference(stats_path: str, pool_type: str = "cls", device="cuda"):
    ref = np.load(stats_path)
    if pool_type == "avg":
        if "avg_mu" not in ref:
            raise KeyError(
                f"pool_type='avg' but {stats_path} has no avg_mu. Available keys: {list(ref.keys())}"
            )
        mu_ref = torch.tensor(ref["avg_mu"], device=device, dtype=torch.float64)
        sigma_ref = torch.tensor(ref["avg_sigma"], device=device, dtype=torch.float64)
    else:
        mu_ref = torch.tensor(ref["mu"], device=device, dtype=torch.float64)
        sigma_ref = torch.tensor(ref["sigma"], device=device, dtype=torch.float64)
    return mu_ref, sigma_ref


def _preprocess(x, mean, std, target_size=None):
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(
            x,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return (x - mean.to(x)) / std.to(x)


class TimmReprModel(nn.Module):
    """Frozen timm feature extractor following FD-Loss repr_models.py."""

    def __init__(self, model_name: str, device="cuda", target_size: int | None = None):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        self.feat_dim = self.model.num_features

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]
        if "naflex" in model_name.lower():
            native_size = 256
        self.target_size = int(target_size) if target_size is not None else native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image_01: torch.Tensor):
        x = _preprocess(image_01, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        patch_tokens = feats[:, self.num_prefix_tokens :]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token


def load_repr_model(name: str, device="cuda", target_size: int | None = None):
    if name == "convnext":
        net = TimmReprModel(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            device=device,
            target_size=224 if target_size is None else target_size,
        )
        return net, net.feat_dim, False, net.target_size
    if name == "inception":
        # The original repo uses a custom Inception loader. For VOSR, use timm
        # Inception consistently for both stats computation and training loss.
        net = TimmReprModel("inception_v3", device=device, target_size=299 if target_size is None else target_size)
        return net, net.feat_dim, False, net.target_size
    net = TimmReprModel(name, device=device, target_size=target_size)
    return net, net.feat_dim, False, net.target_size


def model_short_name(name: str) -> str:
    if name in ("inception", "convnext"):
        return name
    low = name.lower()
    if "naflex" in low:
        return "naflex_siglip"
    for keyword in ("dinov2", "dino", "mae", "clip", "siglip"):
        if keyword in low:
            return keyword
    return name.split(".")[0].replace("_", "-")


def extract_judge_features(judge, image_01):
    primary, secondary = judge["model"](image_01)
    if judge.get("pool_type") == "avg":
        if secondary is None:
            raise ValueError(f"Judge {judge['name']} does not provide avg pooled features.")
        return secondary
    return primary


@dataclass
class FDLossConfig:
    repr_models: list[str]
    repr_stats_paths: list[str]
    repr_weights: list[float]
    repr_pool_types: list[str]
    target_sizes: list[int | None]
    queue_size: int = 50000
    fid_norm_eps: float = 0.01
    online_accum: bool = False
    eigvalsh: bool = False
    ema_beta: float = 0.0


class FDLossSystem(nn.Module):
    """Multi-judge FD-Loss system with queue state."""

    def __init__(self, config: FDLossConfig, device="cuda"):
        super().__init__()
        self.config = config
        self.device_name = str(device)
        self.models = nn.ModuleList()
        self.queues = nn.ModuleList()
        self.judges = []

        for idx, (name, stats_path, weight, pool_type, target_size) in enumerate(
            zip(
                config.repr_models,
                config.repr_stats_paths,
                config.repr_weights,
                config.repr_pool_types,
                config.target_sizes,
            )
        ):
            if not Path(stats_path).is_file():
                raise FileNotFoundError(
                    f"Missing FD reference stats: {stats_path}. "
                    "Run scripts/compute_vosr_fd_ref_stats.py first."
                )
            repr_model, feat_dim, _, _ = load_repr_model(name, device=device, target_size=target_size)
            mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type, device=device)
            queue = FeatureQueue(
                size=config.queue_size,
                feat_dim=feat_dim,
                online_accum=config.online_accum,
                ema_beta=config.ema_beta,
            ).to(device)

            self.models.append(repr_model)
            self.queues.append(queue)
            self.register_buffer(f"mu_ref_{idx}", mu_ref)
            self.register_buffer(f"sigma_ref_{idx}", sigma_ref)
            sigma_ref_sqrt = None
            if config.eigvalsh:
                sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
                self.register_buffer(f"sigma_ref_sqrt_{idx}", sigma_ref_sqrt)

            self.judges.append(
                {
                    "name": model_short_name(name),
                    "model_idx": idx,
                    "queue_idx": idx,
                    "feat_dim": feat_dim,
                    "pool_type": pool_type,
                    "weight": float(weight),
                    "mu_ref_name": f"mu_ref_{idx}",
                    "sigma_ref_name": f"sigma_ref_{idx}",
                    "sigma_ref_sqrt_name": f"sigma_ref_sqrt_{idx}" if sigma_ref_sqrt is not None else None,
                }
            )

    def _image_m11_to_01(self, image_m11):
        return (image_m11.float() + 1.0) * 0.5

    def _extract(self, judge, image_01):
        model = self.models[judge["model_idx"]]
        primary, secondary = model(image_01)
        if judge.get("pool_type") == "avg":
            if secondary is None:
                raise ValueError(f"Judge {judge['name']} does not provide avg pooled features.")
            return secondary
        return primary

    @torch.no_grad()
    def fill_queues_from_images(self, image_m11):
        image_01 = self._image_m11_to_01(image_m11)
        for judge in self.judges:
            feats = diff_all_gather(self._extract(judge, image_01))
            self.queues[judge["queue_idx"]].enqueue(feats)

    @torch.no_grad()
    def init_queues_from_images(self, image_m11, start: int, max_count: int) -> int:
        """Fill the initial generated-feature queues exactly like FD-Loss.

        The original FD-Loss repository initializes the feature buffer with
        generated samples before training and resets the queue pointer to zero.
        This helper writes gathered features into the same index range for all
        judges, then returns the number of global features written.
        """
        if self.config.queue_size == 0:
            return 0

        image_01 = self._image_m11_to_01(image_m11)
        written = None
        for judge in self.judges:
            feats = diff_all_gather(self._extract(judge, image_01))
            count = min(feats.shape[0], max_count)
            queue = self.queues[judge["queue_idx"]]
            if queue.ema_stats:
                queue.accumulate_batch(feats[:count])
            else:
                queue.feats[start : start + count] = feats[:count].detach().float()
            written = count if written is None else min(written, count)
        return int(written or 0)

    @torch.no_grad()
    def init_online_accumulators(self):
        for queue in self.queues:
            if queue.ema_stats:
                queue._finalize_streaming_init()
            elif queue.online_accum and queue.size > 0:
                queue._init_accumulators()

    @torch.no_grad()
    def finalize_queue_init(self):
        for queue in self.queues:
            if queue.ema_stats:
                queue._finalize_streaming_init()
            elif queue.size > 0:
                queue.ptr.zero_()
                if queue.online_accum:
                    queue._init_accumulators()

    def compute_loss(self, image_m11):
        image_01 = self._image_m11_to_01(image_m11)
        loss = torch.tensor(0.0, device=image_m11.device)
        logs = {}
        new_feats_by_judge = []

        for judge in self.judges:
            local_feats = self._extract(judge, image_01)
            new_feats = diff_all_gather(local_feats)
            new_feats_by_judge.append(new_feats)

        for judge, new_feats in zip(self.judges, new_feats_by_judge):
            queue = self.queues[judge["queue_idx"]]
            mu_ref = getattr(self, judge["mu_ref_name"])
            sigma_ref = getattr(self, judge["sigma_ref_name"])
            sigma_ref_sqrt = (
                getattr(self, judge["sigma_ref_sqrt_name"])
                if judge["sigma_ref_sqrt_name"] is not None
                else None
            )

            if queue.online_accum or queue.ema_stats:
                mu, sigma = queue.build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(
                    mu_ref,
                    sigma_ref,
                    mu=mu,
                    sigma=sigma,
                    sigma_ref_sqrt=sigma_ref_sqrt,
                )
            else:
                all_feats = queue.build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(
                    mu_ref,
                    sigma_ref,
                    all_feats=all_feats,
                    sigma_ref_sqrt=sigma_ref_sqrt,
                )

            fid_loss = fid / (fid.detach() + self.config.fid_norm_eps)
            loss = loss + judge["weight"] * fid_loss
            logs[f"fid_{judge['name']}"] = float(fid.detach())

        return loss, logs, new_feats_by_judge

    @torch.no_grad()
    def enqueue_features(self, new_feats_by_judge):
        for judge, feats in zip(self.judges, new_feats_by_judge):
            self.queues[judge["queue_idx"]].enqueue(feats.detach())

    def export_queue_states(self):
        states = []
        for judge in self.judges:
            queue = self.queues[judge["queue_idx"]]
            states.append(
                {
                    "name": judge["name"],
                    "queue": {k: v.detach().cpu() for k, v in queue.state_dict().items()},
                }
            )
        return states

    def load_queue_states(self, states) -> bool:
        name_to_state = {item["name"]: item["queue"] for item in states}
        loaded = 0
        for judge in self.judges:
            state = name_to_state.get(judge["name"])
            if state is None:
                logger.warning(f"[FD] No saved queue state for '{judge['name']}'")
                continue
            queue = self.queues[judge["queue_idx"]]
            state = {k: v.to(next(queue.buffers()).device) for k, v in state.items()}
            queue.load_state_dict(state)
            loaded += 1
        return loaded == len(self.judges)


def _as_list(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_fd_loss_system(args, device="cuda"):
    repr_models = _as_list(getattr(args, "fd_repr_models", None))
    if not repr_models:
        legacy = getattr(args, "fd_repr_model", None)
        repr_models = [legacy or "convnext"]

    stats_paths = _as_list(getattr(args, "fd_repr_stats_paths", None))
    if not stats_paths:
        legacy_stats = getattr(args, "fd_repr_stats_path", None)
        if legacy_stats is None:
            raise ValueError("FD-Loss requires fd_repr_stats_paths in the YAML config.")
        stats_paths = [legacy_stats]
    if len(stats_paths) == 1 and len(repr_models) > 1:
        stats_paths *= len(repr_models)

    weights = _as_list(getattr(args, "fd_repr_weights", None), default=None)
    if weights is None:
        weights = [1.0] * len(repr_models)
    if len(weights) == 1 and len(repr_models) > 1:
        weights *= len(repr_models)
    weights = [float(x) for x in weights]

    pool_types = _as_list(getattr(args, "fd_repr_pool_types", None), default=None)
    if pool_types is None:
        pool_types = ["cls"] * len(repr_models)
    if len(pool_types) == 1 and len(repr_models) > 1:
        pool_types *= len(repr_models)

    target_sizes = _as_list(getattr(args, "fd_target_sizes", None), default=None)
    if target_sizes is None:
        legacy_size = getattr(args, "fd_target_size", None)
        target_sizes = [legacy_size] * len(repr_models)
    if len(target_sizes) == 1 and len(repr_models) > 1:
        target_sizes *= len(repr_models)
    target_sizes = [None if x is None else int(x) for x in target_sizes]

    if not (
        len(repr_models)
        == len(stats_paths)
        == len(weights)
        == len(pool_types)
        == len(target_sizes)
    ):
        raise ValueError("FD repr model/stats/weight/pool/target_size lists must have matching lengths.")

    cfg = FDLossConfig(
        repr_models=repr_models,
        repr_stats_paths=stats_paths,
        repr_weights=weights,
        repr_pool_types=pool_types,
        target_sizes=target_sizes,
        queue_size=int(getattr(args, "fd_queue_size", getattr(args, "queue_size", 50000))),
        fid_norm_eps=float(getattr(args, "fd_fid_norm_eps", 0.01)),
        online_accum=bool(getattr(args, "fd_online_accum", False)),
        eigvalsh=bool(getattr(args, "fd_eigvalsh", False)),
        ema_beta=float(getattr(args, "fd_ema_beta", 0.0)),
    )
    return FDLossSystem(cfg, device=device)
