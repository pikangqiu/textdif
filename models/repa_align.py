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


def repa_cosine_loss(student_tokens, target_tokens, projector):
    """Global pooled REPA loss, robust to different student/target token counts."""
    student = projector(student_tokens)
    student = _pool_tokens(student.float())
    target = _pool_tokens(target_tokens.detach().float())
    student = F.normalize(student, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (student * target).sum(dim=-1).mean()
