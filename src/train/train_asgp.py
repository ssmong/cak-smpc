"""Offline training utilities for the Adaptive Sparse GP (ASGP).

Pipeline (3D dω output, per-sequence HP fitting):
  1. collect_mismatch_data_omega — per-sequence dω residuals (J_nom)
  2. prepare_gp_inputs           — global z-score normalize (pooled)
  3. fit_hyperparams_per_sequence — exact GP per sequence → median HPs
  4. kmeans_inducing_points      — k-means++ for initial Xu
  5. train_asgp                  — Pyro ELBO with median HP init
  6. calibrate_variance          — held-out variance calibration
  7. save_asgp_model             — export .npz + meta.json
"""

import os
import json
import logging

import numpy as np
import torch

from src.utils.math_numpy import attitude_dyn_dt_batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  1. Residual collection
# ---------------------------------------------------------------------------


def _jvec_to_mat(j_vec: np.ndarray) -> np.ndarray:
    """Convert 6D symmetric inertia vector to 3x3 matrix."""
    return np.array([
        [j_vec[0], j_vec[1], j_vec[2]],
        [j_vec[1], j_vec[3], j_vec[4]],
        [j_vec[2], j_vec[4], j_vec[5]],
    ], dtype=np.float64)


def collect_mismatch_data_omega(
    dataset,
    pred_stride: int,
    pred_dt: float,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
    rk4_steps: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-sequence dω residuals using nominal inertia only.

    Always uses ``inertia`` (nominal J) — never per-sequence actual J —
    so residuals include inertia mismatch, which the GP should learn.

    GP input: [ω₃, u₃] = 6D.  GP target: dω₃ = ω_actual − ω_nominal.

    Returns:
        z_per_seq: list of (T_i, 6) arrays — raw GP inputs per sequence.
        y_per_seq: list of (T_i, 3) arrays — raw GP targets per sequence.
    """
    z_per_seq: list[np.ndarray] = []
    y_per_seq: list[np.ndarray] = []

    for seq_idx in dataset.split_indices["train"]:
        x_traj = dataset.x_traj_list[seq_idx].cpu().numpy().astype(np.float64)
        u_traj = dataset.u_traj_list[seq_idx].cpu().numpy().astype(np.float64)
        max_t = min(len(u_traj), len(x_traj) - pred_stride)
        if max_t <= 0:
            continue

        t_indices = np.arange(max_t)
        x_batch = x_traj[t_indices]                     # (M, 7)
        u_batch = u_traj[t_indices]                      # (M, 3)
        x_actual = x_traj[t_indices + pred_stride]       # (M, 7)

        x_nom = attitude_dyn_dt_batch(
            x_batch, u_batch, pred_dt, inertia, inertia_inv,
            steps=rk4_steps)

        domega = x_actual[:, 4:] - x_nom[:, 4:]         # (M, 3)

        # GP input: [omega, u] only (no quaternion)
        z_raw = np.concatenate([x_batch[:, 4:], u_batch], axis=1)  # (M, 6)
        z_per_seq.append(z_raw.astype(np.float64))
        y_per_seq.append(domega.astype(np.float64))

    if not z_per_seq:
        raise RuntimeError("No ASGP training samples were created.")
    total = sum(z.shape[0] for z in z_per_seq)
    log.info(f"Collected {total} dω samples from {len(z_per_seq)} sequences")
    return z_per_seq, y_per_seq


# ---------------------------------------------------------------------------
#  1b. Per-sequence exact GP fitting + median aggregation
# ---------------------------------------------------------------------------


def _batched_exact_gp_fit(
    Z_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 200,
    lr: float = 0.05,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit independent exact GPs in a single GPU batch.

    Args:
        Z_batch: (B, N, Din) padded inputs.
        Y_batch: (B, N) padded targets for ONE output dimension.
        mask:    (B, N) bool — True for valid points, False for padding.
        num_steps: optimization steps.
        lr: learning rate.
        jitter: diagonal regularization.

    Returns:
        lengthscales: (B, Din) ARD lengthscales.
        outputscales: (B,) output scales.
        noise_vars:   (B,) noise variances.
    """
    B, N, Din = Z_batch.shape
    device = Z_batch.device

    # Unconstrained parameters (softplus transform: param = log(exp(raw) + 1))
    raw_ls = torch.zeros(B, Din, device=device, requires_grad=True)
    raw_os = torch.zeros(B, device=device, requires_grad=True)
    raw_nv = torch.zeros(B, device=device, requires_grad=True)  # init softplus(0)≈0.69

    optimizer = torch.optim.Adam([raw_ls, raw_os, raw_nv], lr=lr)
    softplus = torch.nn.functional.softplus

    # Count valid points per sequence for normalization
    n_valid = mask.sum(dim=1).float()  # (B,)

    for step in range(num_steps):
        optimizer.zero_grad()

        ls = softplus(raw_ls)        # (B, Din)
        os_ = softplus(raw_os)       # (B,)
        nv = softplus(raw_nv)        # (B,)

        # Scaled inputs: Z / ls
        Z_scaled = Z_batch / ls.unsqueeze(1)  # (B, N, Din)

        # RBF kernel: K_ij = os * exp(-0.5 * ||z_i - z_j||^2)
        # Pairwise squared distances
        diff = Z_scaled.unsqueeze(2) - Z_scaled.unsqueeze(1)  # (B, N, N, Din)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, N, N)
        K = os_.view(B, 1, 1) * torch.exp(-0.5 * sq_dist)  # (B, N, N)

        # Add noise + jitter on diagonal
        diag_add = nv.view(B, 1) + jitter
        K = K + torch.diag_embed(diag_add.expand(B, N))  # (B, N, N)

        # Mask: zero out rows/cols for padded points, set diagonal to 1
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
        K = K * mask_2d.float()
        # For padded indices, set diagonal to 1 so Cholesky doesn't fail
        pad_diag = (~mask).float()
        K = K + torch.diag_embed(pad_diag)

        # Batched Cholesky
        L = torch.linalg.cholesky(K)  # (B, N, N)

        # Solve L alpha = y  →  alpha = K^{-1} y
        Y_masked = Y_batch * mask.float()  # zero out padded targets
        alpha = torch.cholesky_solve(Y_masked.unsqueeze(-1), L).squeeze(-1)  # (B, N)

        # Neg log marginal likelihood (per sequence):
        # 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5 * n * log(2π)
        data_fit = 0.5 * (Y_masked * alpha).sum(dim=1)  # (B,)
        log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=1)  # (B,) — only valid entries matter
        complexity = log_det  # sum of log(diag(L)), padded entries contribute log(1)=0
        const = 0.5 * n_valid * np.log(2 * np.pi)

        nll = (data_fit + complexity + const).sum()
        nll.backward()
        optimizer.step()

    with torch.no_grad():
        ls_final = softplus(raw_ls).cpu().numpy()
        os_final = softplus(raw_os).cpu().numpy()
        nv_final = softplus(raw_nv).cpu().numpy()

    return ls_final, os_final, nv_final


def fit_hyperparams_per_sequence(
    z_per_seq: list[np.ndarray],
    y_per_seq: list[np.ndarray],
    z_mean: np.ndarray,
    z_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_steps: int = 200,
    lr: float = 0.05,
    max_pts_per_seq: int = 400,
    max_seqs: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit exact GP hyperparameters per sequence in global-normalized space.

    Uses batched GPU computation: all sequences for a given output dimension
    are optimized simultaneously via batched Cholesky / log-MLL.

    Returns:
        median_ls:  (Dout, Din) median ARD lengthscales.
        median_os:  (Dout,) median output-scales.
        median_nv:  (Dout,) median noise variances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_seq = min(len(z_per_seq), max_seqs)
    din = z_per_seq[0].shape[1]
    dout = y_per_seq[0].shape[1]
    rng = np.random.RandomState(42)

    # Normalize z globally (shared lengthscales), y per-sequence locally
    # (so GP fitting reliably separates signal from noise regardless of
    # per-sequence residual scale).  HP are converted to global y-scale after.
    z_list, y_list, y_local_std_list, lengths = [], [], [], []
    for s in range(n_seq):
        z_s = (z_per_seq[s] - z_mean) / z_std
        # Per-sequence local y normalization
        y_raw = y_per_seq[s]
        y_loc_mean = y_raw.mean(axis=0)
        y_loc_std = y_raw.std(axis=0)
        y_loc_std = np.where(y_loc_std < 1e-12, 1e-12, y_loc_std)
        y_s = (y_raw - y_loc_mean) / y_loc_std
        if len(z_s) > max_pts_per_seq:
            idx = rng.choice(len(z_s), max_pts_per_seq, replace=False)
            z_s, y_s = z_s[idx], y_s[idx]
        z_list.append(z_s)
        y_list.append(y_s)
        y_local_std_list.append(y_loc_std)
        lengths.append(len(z_s))

    max_len = max(lengths)
    y_local_stds = np.array(y_local_std_list)  # (n_seq, dout)

    # Build padded tensors
    Z_pad = np.zeros((n_seq, max_len, din), dtype=np.float64)
    Y_pad = np.zeros((n_seq, max_len, dout), dtype=np.float64)
    mask_np = np.zeros((n_seq, max_len), dtype=bool)
    for s in range(n_seq):
        n = lengths[s]
        Z_pad[s, :n] = z_list[s]
        Y_pad[s, :n] = y_list[s]
        mask_np[s, :n] = True

    Z_t = torch.tensor(Z_pad, dtype=torch.float64, device=device)
    Y_t = torch.tensor(Y_pad, dtype=torch.float64, device=device)
    mask_t = torch.tensor(mask_np, device=device)

    all_ls = np.zeros((n_seq, dout, din), dtype=np.float64)
    all_os = np.zeros((n_seq, dout), dtype=np.float64)
    all_nv = np.zeros((n_seq, dout), dtype=np.float64)

    # Process in mini-batches to avoid GPU OOM
    gpu_batch = 50  # sequences per GPU batch
    for j in range(dout):
        log.info(f"  Batched GP fitting: output dim {j}/{dout} ({n_seq} sequences on {device}, batch={gpu_batch})")
        for b_start in range(0, n_seq, gpu_batch):
            b_end = min(b_start + gpu_batch, n_seq)
            ls, os_, nv = _batched_exact_gp_fit(
                Z_t[b_start:b_end], Y_t[b_start:b_end, :, j], mask_t[b_start:b_end],
                num_steps=num_steps, lr=lr,
            )
            all_ls[b_start:b_end, j, :] = ls
            all_os[b_start:b_end, j] = os_
            all_nv[b_start:b_end, j] = nv
            torch.cuda.empty_cache()

    # Convert local-scale HP to global y-scale:
    #   os_global = os_local × (y_local_std / y_std)²
    #   nv_global = nv_local × (y_local_std / y_std)²
    scale_sq = (y_local_stds / y_std[np.newaxis, :]) ** 2  # (n_seq, dout)
    all_os_global = all_os * scale_sq
    all_nv_global = all_nv * scale_sq

    median_ls = np.median(all_ls, axis=0)
    median_os = np.median(all_os_global, axis=0)
    median_nv = np.median(all_nv_global, axis=0)

    log.info(f"Per-sequence HP fitting done ({n_seq} sequences, batched on {device})")
    log.info(f"  Median lengthscale: {median_ls}")
    log.info(f"  Median outputscale: {median_os}")
    noise_frac = all_nv / (all_nv + all_os + 1e-12)
    log.info(f"  Median noise_frac (local): {np.median(noise_frac, axis=0)}")
    log.info(f"  Median noise_var:   {median_nv}")

    return median_ls, median_os, median_nv


# ---------------------------------------------------------------------------
#  2. Input preparation
# ---------------------------------------------------------------------------


def prepare_gp_inputs(
    z_raw: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample (if needed) and z-score normalize GP inputs and targets.

    Returns:
        z_s:    (N', Din) normalized inputs.
        z_mean: (Din,) mean used for input normalization.
        z_std:  (Din,) std used for input normalization.
        y_s:    (N', Dout) normalized targets.
        y_mean: (Dout,) mean used for target normalization.
        y_std:  (Dout,) std used for target normalization.
    """
    if max_samples > 0 and len(z_raw) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(z_raw), size=max_samples, replace=False)
        z_raw = z_raw[idx]
        y = y[idx]

    z_mean = np.mean(z_raw, axis=0)
    z_std = np.maximum(np.std(z_raw, axis=0), 1e-6)
    z_s = (z_raw - z_mean) / z_std

    y_mean = np.mean(y, axis=0)
    y_std = np.maximum(np.std(y, axis=0), 1e-12)
    y_s = (y - y_mean) / y_std

    return z_s, z_mean, z_std, y_s, y_mean, y_std


# ---------------------------------------------------------------------------
#  3. k-means++ inducing points
# ---------------------------------------------------------------------------


def kmeans_inducing_points(
    z: np.ndarray,
    num_clusters: int,
    seed: int,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> np.ndarray:
    """k-means++ initialization for inducing-point locations."""
    n, d = z.shape
    k = int(min(max(1, num_clusters), n))
    if k >= n:
        return z.copy()

    rng = np.random.RandomState(seed)
    centers = np.empty((k, d), dtype=np.float64)
    centers[0] = z[int(rng.randint(0, n))]
    min_dist2 = np.sum((z - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        prob = min_dist2 / np.maximum(np.sum(min_dist2), 1e-12)
        centers[i] = z[int(rng.choice(n, p=prob))]
        min_dist2 = np.minimum(min_dist2, np.sum((z - centers[i]) ** 2, axis=1))

    for _ in range(max_iter):
        dist2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            members = z[labels == ci]
            if len(members) == 0:
                far_idx = int(np.argmax(np.min(dist2, axis=1)))
                new_centers[ci] = z[far_idx]
            else:
                new_centers[ci] = np.mean(members, axis=0)
        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < tol:
            centers = new_centers
            break
        centers = new_centers
    return centers


# ---------------------------------------------------------------------------
#  4. Training via Pyro ELBO
# ---------------------------------------------------------------------------


def train_asgp(
    z_s: np.ndarray,
    y: np.ndarray,
    u_s: np.ndarray,
    lengthscale_ard: np.ndarray,
    outputscale: np.ndarray,
    noise_var: np.ndarray,
    lamb: float,
    jitter: float,
    num_steps: int = 200,
    lr: float = 0.05,
    fixed_Xu: bool = True,
    fixed_noise: bool = False,
):
    """Create MultiOutputASGP and train via batch_update (Pyro ELBO).

    Returns the trained ``MultiOutputASGP`` instance.
    """
    from src.model.adaptive_sgp import MultiOutputASGP

    X_t = torch.tensor(z_s, dtype=torch.float64)
    Y_t = torch.tensor(y, dtype=torch.float64)
    Xu_t = torch.tensor(u_s, dtype=torch.float64)

    model = MultiOutputASGP(
        X=X_t,
        Y=Y_t,
        Xu=Xu_t,
        lengthscale_ard=lengthscale_ard,
        outputscale=outputscale,
        noise_var=noise_var,
        lamb=lamb,
        jitter=jitter,
    )

    log.info(f"Starting ELBO training ({num_steps} steps, lr={lr}, fixed_noise={fixed_noise}) ...")
    losses = model.batch_update(
        num_steps=num_steps, lr=lr, fixed_Xu=fixed_Xu,
        fixed_noise=fixed_noise, verbose=True,
    )
    log.info(f"Training done.  Final losses: {[f'{l:.4f}' for l in losses]}")

    return model


# ---------------------------------------------------------------------------
#  5. Variance calibration
# ---------------------------------------------------------------------------


def calibrate_variance(
    model,
    z_s: np.ndarray,
    y: np.ndarray,
    calib_samples: int,
    seed: int,
    z_mean: np.ndarray | None = None,
    z_std: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    sparse_dir: str | None = None,
    exp_name: str | None = None,
    lambda_forget: float = 0.99,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Calibrate variance via online GP simulation on long sequences.

    Instead of using the offline sparse GP's (broken) posterior variance,
    saves a temporary model, loads it via ``from_pretrained`` (which
    correctly initializes the online sliding-window state), and runs
    predict → observe → update on calibration data.

    Returns:
        var_calib: (Dout,) calibration multipliers.
    """
    from src.model.adaptive_sgp import MultiOutputASGP

    dout = model.output_dim

    # Save a temporary model file so we can use from_pretrained
    import tempfile
    params = model.export_params()
    tmp_npz = os.path.join(tempfile.gettempdir(), f"_calib_tmp_{exp_name}.npz")
    np.savez(
        tmp_npz,
        **params,
        z_mean=z_mean, z_std=z_std,
        y_mean=y_mean, y_std=y_std,
        lambda_forget=lambda_forget,
        jitter=jitter,
        state_dim=6, control_dim=3,
        output_dim=dout,
        residual_mode="domega_only",
        var_calib=np.ones(dout),  # placeholder
    )

    n_calib_seqs = 3
    seq_len = min(200, len(z_s) // n_calib_seqs)
    warmup = 10

    all_err2 = []
    all_var = []

    for seq_i in range(n_calib_seqs):
        # Fresh online GP per sequence via from_pretrained
        online = MultiOutputASGP.from_pretrained(
            model_file=tmp_npz,
            window_size=100,
            num_inducing=30,
            perc_th=0.01,
            lambda_forget=lambda_forget,
            warmup_no_prune=warmup,
        )

        start = seq_i * seq_len
        # Denormalize back to raw space for update() interface
        z_raw_seq = z_s[start:start + seq_len] * z_std + z_mean
        y_raw_seq = y[start:start + seq_len] * y_std + y_mean

        for t in range(seq_len):
            z_raw_t = z_raw_seq[t]
            y_raw_t = y_raw_seq[t]

            if t >= warmup:
                # Predict BEFORE seeing this point
                mu_pred, var_pred = online.predict_np(z_raw_t)
                err2_t = (y_raw_t - mu_pred) ** 2
                all_err2.append(err2_t)
                all_var.append(np.maximum(var_pred, 1e-20))

            # Feed data to online GP (raw space — update normalizes internally)
            online.update(z_raw_t, y_raw_t)

    os.remove(tmp_npz)

    all_err2 = np.array(all_err2)  # (N, Dout)
    all_var = np.array(all_var)    # (N, Dout)

    # var_calib=1 means no correction needed;
    # predict_np already applies var_calib, but we passed var_calib=1 placeholder,
    # so all_var is the raw GP variance. The ratio tells us the multiplier.
    ratios = all_err2 / all_var
    raw_calib = np.mean(ratios, axis=0)
    median_calib = np.median(ratios, axis=0)

    log.info(f"  Online calibration ({n_calib_seqs} seqs × {seq_len} steps, warmup={warmup}):")
    log.info(f"    Mean err²:       {np.mean(all_err2, axis=0)}")
    log.info(f"    Mean pred var:   {np.mean(all_var, axis=0)}")
    log.info(f"    Raw var_calib:   {raw_calib}")
    log.info(f"    Median ratio:    {median_calib}")

    var_calib = np.clip(raw_calib, 0.1, 50.0).astype(np.float64)
    return var_calib


def calibrate_variance_online(
    lengthscale_ard: np.ndarray,
    outputscale: np.ndarray,
    noise_var: np.ndarray,
    u_s: np.ndarray,
    z_per_seq: list[np.ndarray],
    y_per_seq: list[np.ndarray],
    z_mean: np.ndarray,
    z_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    lambda_forget: float = 0.99,
    jitter: float = 1e-8,
    n_calib_seqs: int = 20,
    warmup: int = 10,
) -> np.ndarray:
    """Calibrate variance by simulating online GP on original temporal sequences.

    For each calibration sequence:
      1. Create a fresh online GP via ``from_pretrained``
      2. Run predict → update in temporal order (as in actual deployment)
      3. Collect (err², pred_var) pairs after warmup

    This preserves temporal correlation and gives realistic var_calib
    values, unlike pooled/shuffled data which underestimates variance.

    Returns:
        var_calib: (Dout,) calibration multipliers (clipped to [0.01, 50]).
    """
    import tempfile
    from src.model.adaptive_sgp import MultiOutputASGP

    dout = len(outputscale)
    din = z_per_seq[0].shape[1]
    tmp_npz = os.path.join(tempfile.gettempdir(), "_calib_tmp.npz")
    np.savez(
        tmp_npz,
        U_scaled=u_s,
        lengthscale_ard=lengthscale_ard,
        outputscale=outputscale,
        noise_var=noise_var,
        z_mean=z_mean, z_std=z_std,
        y_mean=y_mean, y_std=y_std,
        lambda_forget=lambda_forget,
        jitter=jitter,
        state_dim=din - 3,
        control_dim=3,
        output_dim=dout,
        residual_mode="domega_only",
        var_calib=np.ones(dout),  # placeholder (multiplicative identity)
    )

    # Select calibration sequences (evenly spaced across dataset)
    n_avail = len(z_per_seq)
    n_use = min(n_calib_seqs, n_avail)
    indices = np.linspace(0, n_avail - 1, n_use, dtype=int)

    all_err2 = []
    all_var = []
    total_pts = 0

    for seq_i in indices:
        z_raw = z_per_seq[seq_i]  # (T, Din) already raw space
        y_raw = y_per_seq[seq_i]  # (T, Dout) already raw space
        seq_len = len(z_raw)
        if seq_len <= warmup:
            continue

        # Fresh online GP per sequence
        online = MultiOutputASGP.from_pretrained(
            model_file=tmp_npz,
            window_size=100,
            num_inducing=len(u_s),
            perc_th=0.0001,
            lambda_forget=lambda_forget,
            warmup_no_prune=warmup,
        )

        for t in range(seq_len):
            if t >= warmup:
                mu_pred, var_pred = online.predict_np(z_raw[t])
                all_err2.append((y_raw[t] - mu_pred) ** 2)
                all_var.append(np.maximum(var_pred, 1e-20))
                total_pts += 1
            online.update(z_raw[t], y_raw[t])

    os.remove(tmp_npz)

    all_err2 = np.array(all_err2)  # (N, Dout)
    all_var = np.array(all_var)    # (N, Dout)
    ratios = all_err2 / all_var
    median_calib = np.median(ratios, axis=0)
    mean_calib = np.mean(ratios, axis=0)

    log.info(f"  Online calibration ({n_use} seqs, warmup={warmup}, {total_pts} points):")
    log.info(f"    Mean err²:       {np.mean(all_err2, axis=0)}")
    log.info(f"    Mean pred var:   {np.mean(all_var, axis=0)}")
    log.info(f"    Mean  ratio:     {mean_calib}")
    log.info(f"    Median ratio:    {median_calib}")

    # Use median: robust to outliers from near-zero variance moments
    return np.clip(median_calib, 0.01, 50.0).astype(np.float64)


# ---------------------------------------------------------------------------
#  6. Save
# ---------------------------------------------------------------------------


def save_asgp_model_direct(
    sparse_dir: str,
    exp_name: str,
    u_s: np.ndarray,
    lengthscale_ard: np.ndarray,
    outputscale: np.ndarray,
    noise_var: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    lambda_forget: float,
    jitter: float,
    state_dim: int,
    control_dim: int,
    pred_dt: float,
    num_samples: int,
    num_inducing: int,
) -> None:
    """Save model directly from HP arrays (no ELBO model object needed)."""
    os.makedirs(sparse_dir, exist_ok=True)
    out_dim = len(outputscale)

    np.savez(
        os.path.join(sparse_dir, f"model_{exp_name}.npz"),
        z_mean=z_mean, z_std=z_std,
        y_mean=y_mean, y_std=y_std,
        lambda_forget=np.array(lambda_forget, dtype=np.float64),
        jitter=np.array(jitter, dtype=np.float64),
        state_dim=np.array(state_dim, dtype=np.int32),
        control_dim=np.array(control_dim, dtype=np.int32),
        output_dim=np.array(out_dim, dtype=np.int32),
        U_scaled=u_s,
        lengthscale_ard=lengthscale_ard,
        outputscale=outputscale,
        noise_var=noise_var,
        residual_mode="domega_only",
    )

    meta = {
        "num_samples": num_samples,
        "num_inducing": num_inducing,
        "pred_dt": pred_dt,
        "state_dim": state_dim,
        "control_dim": control_dim,
        "output_dim": out_dim,
        "model_file": f"sparse_gp/model_{exp_name}.npz",
    }
    with open(os.path.join(sparse_dir, f"meta_{exp_name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"ASGP artifacts saved to {sparse_dir}")


def save_asgp_model(
    model,
    var_calib: np.ndarray,
    sparse_dir: str,
    exp_name: str,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    lambda_forget: float,
    jitter: float,
    state_dim: int,
    control_dim: int,
    pred_dt: float,
    num_samples: int,
    num_inducing: int,
) -> None:
    """Save model_{exp_name}.npz and meta_{exp_name}.json to *sparse_dir*."""
    os.makedirs(sparse_dir, exist_ok=True)

    params = model.export_params()
    out_dim = model.output_dim

    np.savez(
        os.path.join(sparse_dir, f"model_{exp_name}.npz"),
        z_mean=z_mean,
        z_std=z_std,
        y_mean=y_mean,
        y_std=y_std,
        lambda_forget=np.array(lambda_forget, dtype=np.float64),
        jitter=np.array(jitter, dtype=np.float64),
        state_dim=np.array(state_dim, dtype=np.int32),
        control_dim=np.array(control_dim, dtype=np.int32),
        output_dim=np.array(out_dim, dtype=np.int32),
        var_calib=var_calib,
        **params,
    )

    meta = {
        "num_samples": num_samples,
        "num_inducing": num_inducing,
        "pred_dt": pred_dt,
        "state_dim": state_dim,
        "control_dim": control_dim,
        "output_dim": out_dim,
        "model_file": f"sparse_gp/model_{exp_name}.npz",
    }
    with open(os.path.join(sparse_dir, f"meta_{exp_name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"ASGP artifacts saved to {sparse_dir}")
