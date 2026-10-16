import torch
import torch.nn.functional as F

def quat_loss(q_pred: torch.Tensor, 
              q_true: torch.Tensor, 
              eps: float = 1e-8) -> torch.Tensor:
    q_pred = F.normalize(q_pred, p=2, dim=-1, eps=eps)
    q_true = F.normalize(q_true, p=2, dim=-1, eps=eps)
    dots = torch.sum(q_pred * q_true, dim=-1).abs()

    return 2.0 * (1.0 - dots.clamp(0.0, 1.0))