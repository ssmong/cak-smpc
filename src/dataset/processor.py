import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

log = logging.getLogger(__name__)


class DataProcessor(nn.Module):
    """Handles normalization/denormalization of state and control data."""
    
    def __init__(self, state_dim: int, control_dim: int, quat_indices: List[int] = [0, 1, 2, 3], eps: float = 1e-6):
        super().__init__()
        self.x_dim = state_dim
        self.u_dim = control_dim
        self.eps = eps
        
        self.register_buffer('mean', torch.zeros(state_dim))
        self.register_buffer('std', torch.ones(state_dim))
        self.register_buffer('min', torch.zeros(state_dim))
        self.register_buffer('max', torch.zeros(state_dim))
        self.register_buffer('u_mean', torch.zeros(control_dim))
        self.register_buffer('u_std', torch.ones(control_dim))
        self.register_buffer('u_min', torch.zeros(control_dim))
        self.register_buffer('u_max', torch.zeros(control_dim))
        self.register_buffer('skip_norm_mask', torch.tensor([i in quat_indices for i in range(state_dim)], dtype=torch.bool))

    @property
    def state_dim(self) -> int:
        return self.x_dim
    
    @property
    def control_dim(self) -> int:
        return self.u_dim

    def set_stats(self, stats: Dict[str, torch.Tensor]):
        for k in ['mean', 'std', 'min', 'max', 'u_mean', 'u_std', 'u_min', 'u_max']:
            getattr(self, k).copy_(stats[k])
        self.std.clamp_(min=self.eps)
        self.u_std.clamp_(min=self.eps)
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in ['mean', 'std', 'min', 'max', 'u_mean', 'u_std', 'u_min', 'u_max']}

    def load_stats(self, stats_path: Union[str, Path]):
        with open(stats_path, 'r') as f:
            data = json.load(f)
        self.set_stats({k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()})
        log.info(f"Loaded stats from {stats_path}")
    
    def save_stats(self, stats_path: Union[str, Path]):
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({k: v.cpu().tolist() for k, v in self.get_stats().items()}, f, indent=2)
        log.info(f"Saved stats to {stats_path}")

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.skip_norm_mask, x, (x - self.mean) / self.std)

    def denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.skip_norm_mask, x, x * self.std + self.mean)

    def normalize_control(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean) / self.u_std

    def denormalize_control(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.u_std + self.u_mean
