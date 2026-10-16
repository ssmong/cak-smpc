import torch
import numpy as np
from typing import Dict, List, Optional


class WindowCollator:
    """Slices history/future windows from pre-indexed window starts."""
    
    def __init__(
        self,
        log_dt: float,
        hist_dt: float,
        hist_len: int,
        pred_dt: float,
        rollout_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.hist_len = hist_len
        
        self.hist_stride = int(round(hist_dt / log_dt))
        self.pred_stride = int(round(pred_dt / log_dt))
        
        if not np.isclose(self.hist_stride * log_dt, hist_dt, atol=1e-5):
            raise ValueError(f"hist_dt ({hist_dt}) must be multiple of log_dt ({log_dt})")
        if not np.isclose(self.pred_stride * log_dt, pred_dt, atol=1e-5):
            raise ValueError(f"pred_dt ({pred_dt}) must be multiple of log_dt ({log_dt})")
        
        # Rollout mode
        if rollout_steps is not None:
            self.rollout_steps, self.min_steps, self.max_steps = int(rollout_steps), None, None
        elif min_steps is not None and max_steps is not None:
            self.rollout_steps, self.min_steps, self.max_steps = None, int(min_steps), int(max_steps)
        else:
            raise ValueError("Provide rollout_steps or (min_steps, max_steps)")
        
        # Pre-compute history offsets
        self.hist_x_offsets = torch.arange(-hist_len * self.hist_stride, 1, self.hist_stride, dtype=torch.long)
        self.hist_u_offsets = torch.arange(-hist_len * self.hist_stride, 0, self.hist_stride, dtype=torch.long)

    def _get_rollout_steps(self) -> int:
        if self.rollout_steps is not None:
            return self.rollout_steps
        return int(torch.randint(self.min_steps, self.max_steps + 1, (1,)).item())

    def _ensure_offsets_device(self, device: torch.device) -> None:
        """Move pre-computed offset tensors to match data device (lazy, once)."""
        if self.hist_x_offsets.device != device:
            self.hist_x_offsets = self.hist_x_offsets.to(device)
            self.hist_u_offsets = self.hist_u_offsets.to(device)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        N = self._get_rollout_steps()
        
        # Detect data device from first sample
        data_device = batch[0]['x_traj'].device
        self._ensure_offsets_device(data_device)
        x_offsets = torch.arange(0, N * self.pred_stride + 1, self.pred_stride, dtype=torch.long, device=data_device)
        u_offsets = torch.arange(0, N * self.pred_stride, self.pred_stride, dtype=torch.long, device=data_device)
        
        hist_x_list, hist_u_list, x_seq_list, u_seq_list = [], [], [], []
        
        for s in batch:
            if 'start_idx' not in s:
                raise KeyError("Missing required key 'start_idx' in sample")
            x_traj, u_traj = s['x_traj'], s['u_traj']
            k = int(s['start_idx'])
            max_idx = x_traj.shape[0] - 1 - N * self.pred_stride
            if k > max_idx:
                raise ValueError(
                    f"Window start index {k} is invalid for rollout={N} with trajectory length {x_traj.shape[0]}"
                )

            hist_x_list.append(x_traj[k + self.hist_x_offsets])
            hist_u_list.append(u_traj[k + self.hist_u_offsets])
            x_seq_list.append(x_traj[k + x_offsets])
            u_seq_list.append(u_traj[k + u_offsets])
        
        return {
            'hist_x': torch.stack(hist_x_list),   # (B, H+1, x_dim)
            'hist_u': torch.stack(hist_u_list),   # (B, H, u_dim)
            'x_seq': torch.stack(x_seq_list),     # (B, N+1, x_dim) - rollout GT
            'u_seq': torch.stack(u_seq_list),     # (B, N, u_dim) - rollout control
            'rollout_steps': N,
        }
