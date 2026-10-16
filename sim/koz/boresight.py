import torch
import torch.nn as nn
import numpy as np
from torch.func import jacrev, vmap
from typing import List, Tuple
from omegaconf import DictConfig

class BoresightHandler:
    def __init__(self, boresight_cfg: List[DictConfig], model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        vecs = []
        for bs in boresight_cfg:
            v = np.array(bs.vec, dtype=float)
            v = v / np.linalg.norm(v)
            vecs.append(v)

        # self.bs_B: (L, 3) - Constant tensor on device
        self.bs_B = torch.tensor(np.array(vecs), dtype=self.dtype, device=self.device)
        self.num_bs = len(vecs)

        self._setup_jacobian_func()

    def _setup_jacobian_func(self):
        def single_step_logic(z, bs_vec):
            # z: (D,), bs_vec: (3,)
            x = self.model.decode(z.unsqueeze(0)).squeeze(0)
            bs_N = self._rotate_vector(x[:4], bs_vec)
            return bs_N

        self._compute_val_and_jac = vmap(
            lambda z, bs: (single_step_logic(z, bs), jacrev(single_step_logic, argnums=0)(z, bs)),
            in_dims=(0, 0)
        )

    def _rotate_vector(self, q, v):
        q_w = q[0]
        q_xyz = q[1:4]
        t = 2.0 * torch.linalg.cross(q_xyz, v)
        return v + q_w * t + torch.linalg.cross(q_xyz, t)

    def compute(self, z_traj: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            z_traj: (H, D) Latent trajectory
        Returns:
            bs_N: (H, L, 3) Inertial vectors (Numpy)
            J_bs: (H, L, 3, D) Jacobian (Numpy)
        """
        if z_traj.device != self.device:
            z_traj = z_traj.to(self.device)

        H, D = z_traj.shape
        L = self.num_bs

        # z: (H, D) -> (H*L, D)
        z_in = z_traj.unsqueeze(1).expand(H, L, D).reshape(-1, D)
        # bs: (L, 3) -> (H*L, 3)
        bs_in = self.bs_B.unsqueeze(0).expand(H, L, 3).reshape(-1, 3)

        bs_N_flat, J_bs_flat = self._compute_val_and_jac(z_in, bs_in)

        bs_N = bs_N_flat.view(H, L, 3).detach().cpu().numpy()
        J_bs = J_bs_flat.view(H, L, 3, D).detach().cpu().numpy()

        return bs_N, J_bs
