import torch
import torch.nn as nn
import numpy as np

try:
    from torch.func import jacrev, vmap
    _HAS_TORCH_FUNC = True
except ImportError:
    _HAS_TORCH_FUNC = False


def _decode_jacobian(model: nn.Module, z_vec: torch.Tensor) -> torch.Tensor:
    """Compute 7×d Jacobian of normalized decode(z) w.r.t. z (single forward pass)."""
    z = z_vec.detach().clone().requires_grad_(True)
    x = model.decode(z.unsqueeze(0))[0]  # decoder already normalizes quaternion
    rows = []
    for i in range(7):
        gi = torch.autograd.grad(x[i], z, retain_graph=True, allow_unused=False)[0]
        rows.append(gi)
    return torch.stack(rows, dim=0)  # (7, d)


def _decode_jacobian_batch(model: nn.Module, z_batch: torch.Tensor) -> torch.Tensor:
    """Batch Jacobian of decode(z) w.r.t. z for multiple z vectors.

    Args:
        model: decoder model with .decode(z) method
        z_batch: (B, d) tensor

    Returns:
        (B, 7, d) Jacobian tensor
    """
    if _HAS_TORCH_FUNC:
        def _decode_single(z):
            return model.decode(z.unsqueeze(0)).squeeze(0)
        try:
            return vmap(jacrev(_decode_single))(z_batch)
        except Exception:
            pass  # fall through to loop
    # Fallback: sequential loop (still on GPU)
    return torch.stack([_decode_jacobian(model, z_batch[i])
                        for i in range(z_batch.shape[0])])


def _project_psd(Qz: np.ndarray, eig_floor_ratio: float) -> np.ndarray:
    """Project onto PSD cone with relative eigenvalue floor."""
    Qz = 0.5 * (Qz + Qz.T)
    eigvals, V = np.linalg.eigh(Qz)
    floor = max(eigvals.max() * eig_floor_ratio, eig_floor_ratio)
    eigvals = np.maximum(eigvals, floor)
    return V @ np.diag(eigvals) @ V.T


def get_latent_Q(model: nn.Module, z_vec: torch.Tensor, Q_diag: list[float],
                 eig_floor_ratio: float = 1e-8) -> np.ndarray:
    """
    Compute the latent weight matrix from the state weight matrix using Jacobian.
    Args:
        model: nn.Module (PartyInterface or Party)
        z_vec: torch.Tensor (d,) - latent vector
        Q_diag: list[float] - diagonal elements of the state weight matrix
    Returns:
        Qz: np.ndarray (d,d) - latent weight matrix
    """
    J = _decode_jacobian(model, z_vec)
    Qx = torch.diag(torch.tensor(Q_diag, device=z_vec.device, dtype=z_vec.dtype))
    Qz = (J.T @ Qx @ J).detach().cpu().numpy().astype(np.float64)
    return _project_psd(Qz, eig_floor_ratio)


def get_latent_QF(model: nn.Module, z_vec: torch.Tensor,
                  Q_diag: list[float], F_diag: list[float],
                  eig_floor_ratio: float = 1e-8
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute both Q_latent and F_latent from a single Jacobian evaluation."""
    J = _decode_jacobian(model, z_vec)
    dev, dt = z_vec.device, z_vec.dtype
    Qx = torch.diag(torch.tensor(Q_diag, device=dev, dtype=dt))
    Fx = torch.diag(torch.tensor(F_diag, device=dev, dtype=dt))
    JT = J.T
    Qz = (JT @ Qx @ J).detach().cpu().numpy().astype(np.float64)
    Fz = (JT @ Fx @ J).detach().cpu().numpy().astype(np.float64)
    return _project_psd(Qz, eig_floor_ratio), _project_psd(Fz, eig_floor_ratio)


def get_latent_QF_batch(model: nn.Module, z_batch: torch.Tensor,
                        Q_diag: list[float], F_diag: list[float],
                        eig_floor_ratio: float = 1e-8
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Stage-varying Q and terminal F via batch Jacobians along a trajectory.

    Args:
        model: decoder model (PartyInterface or Party)
        z_batch: (N+1, d) — z_batch[0..N-1] for stage costs,
                 z_batch[N] for terminal cost.
        Q_diag: diagonal of state-space stage cost (7,)
        F_diag: diagonal of state-space terminal cost (7,)
        eig_floor_ratio: eigenvalue floor for PSD projection

    Returns:
        Q_seq: (N, d, d) numpy — stage-varying Q in latent space
        F_lat: (d, d) numpy — terminal cost in latent space
    """
    N = z_batch.shape[0] - 1
    dev, dt = z_batch.device, z_batch.dtype
    Qx = torch.diag(torch.tensor(Q_diag, device=dev, dtype=dt))
    Fx = torch.diag(torch.tensor(F_diag, device=dev, dtype=dt))

    J_all = _decode_jacobian_batch(model, z_batch)  # (N+1, 7, d)
    JT = J_all.transpose(-2, -1)                    # (N+1, d, 7)

    # Stage costs: first N Jacobians → Q_k = J_k^T Q_x J_k
    Qz_all = (JT[:N] @ Qx @ J_all[:N]).detach().cpu().numpy().astype(np.float64)
    # Terminal cost: last Jacobian → F = J_N^T F_x J_N
    Fz = (JT[N:] @ Fx @ J_all[N:]).squeeze(0).detach().cpu().numpy().astype(np.float64)

    Q_seq = np.stack([_project_psd(Qz_all[i], eig_floor_ratio) for i in range(N)])
    F_lat = _project_psd(Fz, eig_floor_ratio)

    return Q_seq, F_lat
