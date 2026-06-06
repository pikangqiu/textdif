import torch
import torch.nn as nn
import torch.nn.functional as F


class RepaProjector(nn.Module):
    """Project DiT hidden tokens into an external representation space."""

    def __init__(self, input_dim, target_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        return self.net(x)


def _pool_tokens(x):
    if x.ndim == 4:
        x = x.flatten(2).transpose(1, 2)
    if x.ndim != 3:
        raise ValueError(f"Expected token tensor with 3 or 4 dims, got shape={tuple(x.shape)}")
    return x.mean(dim=1)


def repa_cosine_loss(student_tokens, target_tokens, projector, mode="global"):
    """REPA cosine loss.

    mode="token" follows the original REPA token-wise alignment when token counts
    match. mode="global" is a more permissive variant for OCR/segmentation
    encoders whose token layouts differ from DiT tokens.
    """
    student = projector(student_tokens)
    if target_tokens.ndim == 4:
        target_tokens = target_tokens.flatten(2).transpose(1, 2)
    if target_tokens.ndim != 3:
        raise ValueError(f"Expected target token tensor with 3 or 4 dims, got shape={tuple(target_tokens.shape)}")

    if mode == "token":
        if student.ndim != 3:
            raise ValueError(f"Expected student token tensor with 3 dims, got shape={tuple(student.shape)}")
        if student.shape[1] != target_tokens.shape[1]:
            raise ValueError(
                "Token-wise REPA requires matching token counts: "
                f"student={student.shape[1]}, target={target_tokens.shape[1]}"
            )
        student = F.normalize(student.float(), dim=-1)
        target = F.normalize(target_tokens.detach().float(), dim=-1)
        return 1.0 - (student * target).sum(dim=-1).mean()

    if mode != "global":
        raise ValueError(f"Invalid REPA alignment mode: {mode}")

    student = _pool_tokens(student.float())
    target = _pool_tokens(target_tokens.detach().float())
    student = F.normalize(student, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (student * target).sum(dim=-1).mean()
