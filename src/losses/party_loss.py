import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.losses.quat_loss import quat_loss


# ==================== Private helpers ====================

def _svd_loss(A: torch.Tensor, rho_max_A: float) -> torch.Tensor:
    """L2-hinge spectral radius loss using exact SVD (deterministic, no noise).

    For l_dim=32, exact SVD is fast and avoids power-iteration noise near A ≈ I.
    """
    sigma_max = torch.linalg.svdvals(A)[:, 0]  # (b,) largest singular value
    excess = torch.clamp(sigma_max - rho_max_A, min=0.0)
    return excess.pow(2).mean()


def _nll_loss(
    z_seq_hat: torch.Tensor, z_seq_gt_mu: torch.Tensor,
    A: torch.Tensor, z_logvar_0: torch.Tensor,
    N: int, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagated Gaussian NLL (stop-gradient on dynamics, only z_logvar_0 learns).

    NLL is self-balancing: 0.5*(error²/P_k + log(P_k)) has minimum at P_k = error².
    No logvar clamp needed — P_k min/max provide numerical safety only.

    Returns:
        loss_nll: scalar NLL loss.
        loss_P0: scalar initial variance (for monitoring).
    """
    P_K_MIN = 1e-6
    A_sq_sg = A.detach().pow(2)
    P_0 = torch.exp(z_logvar_0).unsqueeze(-1)
    P_k = P_0.clone()
    log_P_min = torch.tensor(P_K_MIN, device=device, dtype=dtype).log()

    nll_sum = torch.tensor(0.0, device=device, dtype=dtype)
    for k in range(N):
        P_k = torch.clamp(torch.bmm(A_sq_sg, P_k), min=P_K_MIN, max=1e2)
        P_k_flat = P_k.squeeze(-1)  # (b, l_dim)
        error_sq = (z_seq_hat[:, k + 1].detach() - z_seq_gt_mu[:, k]).pow(2)
        nll_sum = nll_sum + (0.5 * (error_sq / P_k_flat) + 0.5 * (P_k_flat.log() - log_P_min)).mean()

    loss_P0 = torch.exp(z_logvar_0).mean()

    return nll_sum / N, loss_P0


def _metric_loss(
    x_seq_gt: torch.Tensor,
    z_mu: torch.Tensor,
    z_gt_mu_raw: torch.Tensor,
    weight_cfg: DictConfig,
) -> torch.Tensor:
    """Metric-preserving loss: latent Euclidean distance ≈ physical geodesic distance.

    Direct comparison without learnable scale — prevents pred_z/metric tug-of-war.
    z scale converges to physical distance scale; MPC absorbs absolute scale via Q matrix.

    Args:
        x_seq_gt: (b, N+1, 7) GT physical states.
        z_mu: (b, l_dim) latent encoding of initial state.
        z_gt_mu_raw: (b, N, l_dim) latent encodings of GT states t=1..N (with grad).
        weight_cfg: loss weight config.
    """
    sample_size = int(weight_cfg.metric_samples)

    z_all = torch.cat([z_mu.unsqueeze(1), z_gt_mu_raw], dim=1)  # (b, N+1, l_dim)
    x_all = x_seq_gt  # (b, N+1, 7) — unwrapped, quat_loss is sign-invariant

    D_z = z_all.shape[-1]
    flat_z = z_all.reshape(-1, D_z)
    flat_x = x_all.reshape(-1, 7)
    n = min(flat_z.size(0), sample_size)

    idx = torch.randperm(flat_z.size(0), device=flat_z.device)[:n]
    sub_z = flat_z[idx]
    sub_q = flat_x[idx, :4]
    sub_w = flat_x[idx, 4:]

    dist_z = torch.cdist(sub_z, sub_z, p=2).pow(2)
    dist_q = quat_loss(sub_q.unsqueeze(1), sub_q.unsqueeze(0))  # squared chordal distance
    dist_omega = torch.cdist(sub_w, sub_w, p=2).pow(2)
    dist_x = weight_cfg.quat * dist_q + weight_cfg.omega * dist_omega

    # Scale target to keep z compact (relative structure preserved, absolute scale shrunk)
    metric_scale = float(weight_cfg.metric_scale)
    dist_x = metric_scale * dist_x

    return F.smooth_l1_loss(dist_x, dist_z)



def _geodesic_path_loss(
    model: nn.Module,
    device: torch.device, dtype: torch.dtype,
    num_pairs: int = 128,
    omega_max: float = 0.1,
) -> torch.Tensor:
    """Path linearity: chordal interpolation on S³ maps to straight line in latent space.

    Samples random quaternion pairs (training-data independent) and enforces:
        enc(lerp(x_a, x_b, t)) ≈ (1−t)·enc(x_a) + t·enc(x_b)

    Uses chordal (normalize-after-lerp) instead of SLERP — simpler, no arccos
    numerical issues, and double-cover invariance is not needed.
    """
    # Random unit quaternions (uniform on S³ → uniform on SO(3))
    q_a = F.normalize(torch.randn(num_pairs, 4, device=device, dtype=dtype), dim=-1)
    q_b = F.normalize(torch.randn(num_pairs, 4, device=device, dtype=dtype), dim=-1)

    # Random angular velocities
    w_a = omega_max * (2 * torch.rand(num_pairs, 3, device=device, dtype=dtype) - 1)
    w_b = omega_max * (2 * torch.rand(num_pairs, 3, device=device, dtype=dtype) - 1)

    x_a = torch.cat([q_a, w_a], dim=-1)  # (K, 7)
    x_b = torch.cat([q_b, w_b], dim=-1)

    # Encode endpoints
    z_a, _ = model.encode(x_a)
    z_b, _ = model.encode(x_b)

    # Random interpolation parameter per pair
    t = torch.rand(num_pairs, 1, device=device, dtype=dtype)

    # Chordal interpolation for quaternion (lerp + renormalize), lerp for omega
    q_interp = F.normalize((1 - t) * q_a + t * q_b, dim=-1)
    w_interp = (1 - t) * w_a + t * w_b
    x_interp = torch.cat([q_interp, w_interp], dim=-1)

    z_interp, _ = model.encode(x_interp)
    z_expected = (1 - t) * z_a + t * z_b

    return (z_interp - z_expected).pow(2).mean()


def _ctx_smooth_loss(
    model: nn.Module,
    hist_x: torch.Tensor,
    hist_u: torch.Tensor,
    A: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Context smoothness: ||A(ctx_t) - A(ctx_{t+shift})||_F^2.

    Penalizes A matrix change when the history window shifts by `shift` steps
    (typically pred_dt / hist_dt = 5, matching MPC re-solve interval).
    """
    ctx_shifted = model.get_ctx(hist_x[:, shift:], hist_u[:, shift:])
    A_shifted, _ = model.get_AB(ctx_shifted)
    return (A - A_shifted).pow(2).mean()


# ==================== Public API ====================

def party_loss(
    model: nn.Module,
    x_seq_hat: torch.Tensor,
    z_seq_hat: torch.Tensor,
    x_seq_gt: torch.Tensor,
    A: torch.Tensor,
    z_mu_logvar: tuple[torch.Tensor, torch.Tensor],
    weight_cfg: DictConfig,
    raw_qnorms: torch.Tensor | None = None,
    hist_x: torch.Tensor | None = None,
    hist_u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total PARTY training loss.

    Args:
        model: Party model.
        x_seq_hat: (b, N+1, x_dim) predicted states.
        z_seq_hat: (b, N+1, l_dim) latent rollout.
        x_seq_gt:  (b, N+1, x_dim) ground truth states.
        A: (b, l_dim, l_dim) Koopman A matrix.
        z_mu_logvar: (z_mu, z_logvar) from VAE encoder (initial state).
        weight_cfg: loss weight config.
        raw_qnorms: (b, N) raw quaternion norms from decoder (optional).
        hist_x: (b, H+1, x_dim) history states (for ctx_smooth loss).
        hist_u: (b, H, u_dim) history controls (for ctx_smooth loss).
    """
    device, dtype = x_seq_hat.device, x_seq_hat.dtype
    z_mu_0, z_logvar_0 = z_mu_logvar
    N = x_seq_gt.shape[1] - 1

    w_pred_x = weight_cfg.pred_x
    w_pred_z = weight_cfg.pred_z
    w_nll = weight_cfg.nll
    w_recon_z = weight_cfg.recon_z
    w_svd_rho = weight_cfg.svd_rho
    w_qnorm = weight_cfg.qnorm
    w_metric = weight_cfg.metric
    w_geodesic = weight_cfg.geodesic
    w_P0 = weight_cfg.P0
    w_ctx_smooth = weight_cfg.ctx_smooth

    # --- pred_x: physical-space prediction loss ---
    q_err = quat_loss(x_seq_hat[:, 1:, :4], x_seq_gt[:, 1:, :4])
    w_err = (x_seq_hat[:, 1:, 4:7] - x_seq_gt[:, 1:, 4:7]).pow(2).sum(-1)
    loss_pred_x = (weight_cfg.quat * q_err + weight_cfg.omega * w_err).mean()

    # --- pred_z: latent-space prediction loss ---
    z_seq_gt_mu_raw, _ = model.encode(x_seq_gt[:, 1:])  # (b, N, l_dim) — with grad
    z_seq_gt_mu = z_seq_gt_mu_raw.detach()
    loss_pred_z = (z_seq_hat[:, 1:] - z_seq_gt_mu).pow(2).mean()

    # --- nll: propagated Gaussian NLL ---
    loss_nll, loss_P0 = _nll_loss(z_seq_hat, z_seq_gt_mu, A, z_logvar_0, N, device, dtype)

    # --- recon_z: round-trip consistency (z -> x -> z) ---
    z_for_recon = z_seq_hat.detach()
    x_decoded = model.decode(z_for_recon)
    z_reencoded_mu, _ = model.encode(x_decoded)
    loss_recon_z = (z_reencoded_mu - z_for_recon).pow(2).mean()

    # --- svd: spectral radius constraint ---
    rho_max_A = getattr(model.koopman, 'rho_max_A', 1.03) if hasattr(model, 'koopman') else 1.03
    loss_svd = _svd_loss(A, rho_max_A)

    # --- qnorm: decoder raw quaternion norm ---
    if raw_qnorms is not None and w_qnorm > 0:
        loss_qnorm = (raw_qnorms - 1.0).pow(2).mean()
    else:
        loss_qnorm = torch.tensor(0.0, device=device, dtype=dtype)

    # --- metric: latent isometry ---
    if w_metric > 0:
        loss_metric = _metric_loss(x_seq_gt, z_mu_0, z_seq_gt_mu_raw, weight_cfg)
    else:
        loss_metric = torch.tensor(0.0, device=device, dtype=dtype)

    # --- geodesic: SLERP path ≈ straight line in latent space ---
    if w_geodesic > 0:
        geo_samples = int(weight_cfg.geodesic_samples)
        geo_omega_max = float(weight_cfg.geodesic_omega_max)
        loss_geodesic = _geodesic_path_loss(model, device, dtype,
                                            num_pairs=geo_samples,
                                            omega_max=geo_omega_max)
    else:
        loss_geodesic = torch.tensor(0.0, device=device, dtype=dtype)

    # --- ctx_smooth: A matrix consistency across shifted history windows ---
    if w_ctx_smooth > 0 and hist_x is not None and hist_u is not None:
        shift = int(weight_cfg.ctx_smooth_shift)
        loss_ctx_smooth = _ctx_smooth_loss(model, hist_x, hist_u, A, shift)
    else:
        loss_ctx_smooth = torch.tensor(0.0, device=device, dtype=dtype)

    loss_total = (
        w_pred_x * loss_pred_x +
        w_pred_z * loss_pred_z +
        w_nll * loss_nll +
        w_recon_z * loss_recon_z +
        w_svd_rho * loss_svd +
        w_qnorm * loss_qnorm +
        w_metric * loss_metric +
        w_geodesic * loss_geodesic +
        w_P0 * loss_P0 +
        w_ctx_smooth * loss_ctx_smooth
    )

    loss_dict = {
        'total': loss_total,
        'pred_x': loss_pred_x,
        'pred_z': loss_pred_z,
        'nll': loss_nll,
        'recon_z': loss_recon_z,
        'P0': loss_P0,
        'svd': loss_svd,
        'qnorm': loss_qnorm,
        'metric': loss_metric,
        'geodesic': loss_geodesic,
        'ctx_smooth': loss_ctx_smooth,
    }

    return loss_total, loss_dict
