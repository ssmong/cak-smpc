"""Offline model diagnostics — no Basilisk required.

Analyzes a trained PARTY model for:
  1. z_logvar distribution (encoder uncertainty calibration)
  2. Decoder Jacobian norms (quaternion vs omega amplification)
  3. W_offline eigenvalue spectrum (latent prediction error covariance)
  4. A spectral radius distribution (Koopman stability)
  5. ctx sensitivity (context change → A change correlation)
  6. Gaussianity check for latent process noise (--gaussianity flag)
     Generates gaussianity_data.mat, gaussianity_summary.yaml,
     and latent_residuals.npy for MATLAB plotting.

Usage::

    python -m scripts.90_party_diagnostics --model_dir 2026-02-25/09-22-57
    python -m scripts.90_party_diagnostics --model_dir 2026-02-25/09-22-57 --num_batches 5
    python -m scripts.90_party_diagnostics --model_dir 2026-02-25/09-22-57 --gaussianity
    python -m scripts.90_party_diagnostics --model_dir 2026-02-25/09-22-57 --gaussianity --num_dims 8
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import PartyDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Model & data loading
# ---------------------------------------------------------------------------

def load_model_and_cfg(model_dir: str, device: str = "cuda"):
    """Load trained model and its config (no PartyInterface)."""
    import hydra

    base_dir = os.path.join("outputs/learning_party", model_dir)
    cfg_path = os.path.join(base_dir, ".hydra", "config.yaml")
    weights_path = os.path.join(base_dir, "best_model.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    cfg = OmegaConf.load(cfg_path)

    for key in ["context_conditioning", "branch_decoder"]:
        if key in cfg.model:
            del cfg.model[key]

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd = {k.replace('log_var_head', 'logvar_head'): v
          for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg


def make_test_loader(cfg, batch_size: int = 256):
    """Create a test DataLoader from the training config."""
    stats_dir = os.path.join("outputs/learning_party",
                             cfg._metadata_model_dir if hasattr(cfg, '_metadata_model_dir')
                             else "tmp_stats", "stats")
    if not os.path.exists(stats_dir):
        stats_dir = "data/stats"

    ds = PartyDataset(
        name=cfg.data.name,
        data_root=cfg.data.data_root,
        state_dim=cfg.data.state_dim,
        control_dim=cfg.data.control_dim,
        state_key=cfg.data.state_key,
        control_key=cfg.data.control_key,
        num_seqs=cfg.data.num_seqs,
        sim_time=cfg.data.sim_time,
        log_dt=cfg.data.log_dt,
        hist_dt=cfg.data.hist_dt,
        pred_dt=cfg.data.pred_dt,
        hist_len=cfg.data.hist_len,
        sample_stride=cfg.data.get('sample_stride', 1),
        quat_indices=list(cfg.data.quat_indices),
        split="test",
        seed=cfg.data.seed,
        use_norm=False,
        stats_dir=stats_dir,
        storage_device="cpu",
        load_inertia=False,
    )
    collate_fn = ds.get_collate_fn(rollout_steps=cfg.train.test_steps)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0)


def make_test_dataset(cfg):
    """Create raw test dataset (no DataLoader — for manual iteration)."""
    base_dir = os.path.join("outputs/learning_party",
                            cfg._metadata_model_dir, "stats")
    stats_dir = base_dir if os.path.exists(base_dir) else "data/stats"

    return PartyDataset(
        name=cfg.data.name,
        data_root=cfg.data.data_root,
        state_dim=cfg.data.state_dim,
        control_dim=cfg.data.control_dim,
        state_key=cfg.data.state_key,
        control_key=cfg.data.control_key,
        num_seqs=cfg.data.num_seqs,
        sim_time=cfg.data.sim_time,
        log_dt=cfg.data.log_dt,
        hist_dt=cfg.data.hist_dt,
        pred_dt=cfg.data.pred_dt,
        hist_len=cfg.data.hist_len,
        sample_stride=cfg.data.get('sample_stride', 1),
        quat_indices=list(cfg.data.quat_indices),
        split="test",
        seed=cfg.data.seed,
        use_norm=False,
        stats_dir=stats_dir,
        storage_device="cpu",
        load_inertia=False,
    )


# ---------------------------------------------------------------------------
#  Diagnostic 1: z_logvar distribution
# ---------------------------------------------------------------------------

def diag_z_logvar(model, loader, device, num_batches, out_dir):
    print("\n=== 1. z_logvar Distribution ===")
    all_logvar = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            x = batch['x_seq'].to(device)
            x_flat = x.reshape(-1, x.shape[-1])
            _, logvar = model.encode(x_flat)
            all_logvar.append(logvar.cpu())

    all_logvar = torch.cat(all_logvar, dim=0).numpy()
    variance = np.exp(all_logvar)

    print(f"  z_logvar: mean={all_logvar.mean():.2f}, std={all_logvar.std():.2f}, "
          f"min={all_logvar.min():.2f}, max={all_logvar.max():.2f}")
    print(f"  variance: mean={variance.mean():.2e}, median={np.median(variance):.2e}, "
          f"max={variance.max():.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_logvar.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("z_logvar")
    axes[0].set_ylabel("Count")
    axes[0].set_title("z_logvar Distribution")
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5, label='logvar=0 (var=1)')
    axes[0].legend()

    per_dim_mean = all_logvar.mean(axis=0)
    axes[1].bar(range(len(per_dim_mean)), per_dim_mean)
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Mean z_logvar")
    axes[1].set_title("Per-Dimension Mean z_logvar")
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_z_logvar.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Diagnostic 2: Decoder Jacobian
# ---------------------------------------------------------------------------

def diag_jacobian(model, loader, device, num_samples, out_dir):
    print("\n=== 2. Decoder Jacobian ===")
    with torch.no_grad():
        batch = next(iter(loader))
        x = batch['x_seq'][:num_samples, 0].to(device)
        z_mu, _ = model.encode(x)

    z_mu = z_mu.detach().requires_grad_(True)
    jac_list = []
    for i in range(z_mu.shape[0]):
        z_i = z_mu[i:i+1]

        def decode_fn(z):
            return model.decode(z)

        J = torch.autograd.functional.jacobian(decode_fn, z_i, vectorize=True)
        J = J.squeeze(0).squeeze(1)
        jac_list.append(J.detach().cpu().numpy())

    jac = np.stack(jac_list)
    q_jac = jac[:, :4, :]
    w_jac = jac[:, 4:, :]
    q_norms = np.linalg.norm(q_jac, axis=(1, 2))
    w_norms = np.linalg.norm(w_jac, axis=(1, 2))

    print(f"  ||J_quat||: mean={q_norms.mean():.3f}, std={q_norms.std():.3f}")
    print(f"  ||J_omega||: mean={w_norms.mean():.3f}, std={w_norms.std():.3f}")
    print(f"  Ratio ||J_q||/||J_w||: {(q_norms / (w_norms + 1e-8)).mean():.1f}x")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['||J_quat||', '||J_omega||'], [q_norms.mean(), w_norms.mean()],
           yerr=[q_norms.std(), w_norms.std()], capsize=5, alpha=0.7)
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Decoder Jacobian: Quaternion vs Omega")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_jacobian.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Diagnostic 3: W_offline eigenvalue spectrum
# ---------------------------------------------------------------------------

def diag_W_offline(model, loader, device, num_batches, out_dir):
    print("\n=== 3. W_offline (1-step Latent Error Covariance) ===")
    errors = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            hist_x = batch['hist_x'].to(device)
            hist_u = batch['hist_u'].to(device)
            x_seq = batch['x_seq'].to(device)
            u_seq = batch['u_seq'].to(device)

            ctx = model.get_ctx(hist_x, hist_u)
            z_mu, _ = model.encode(x_seq[:, 0])
            z_next_true, _ = model.encode(x_seq[:, 1])
            A, B = model.get_AB(ctx)
            z_next_pred = model.step(z_mu, u_seq[:, 0], A, B)
            e = (z_next_true - z_next_pred).cpu().numpy()
            errors.append(e)

    errors = np.vstack(errors)
    W = np.cov(errors, rowvar=False)
    eigvals = np.linalg.eigvalsh(W)[::-1]

    print(f"  Error norm: mean={np.linalg.norm(errors, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(errors, axis=1).std():.4f}")
    print(f"  W trace={np.trace(W):.4f}, cond={np.linalg.cond(W):.1f}")
    print(f"  Top-5 eigenvalues: {eigvals[:5]}")
    print(f"  Bottom-5 eigenvalues: {eigvals[-5:]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(eigvals, 'o-', markersize=3)
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("W_offline Eigenvalue Spectrum")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.linalg.norm(errors, axis=1), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("||e|| (1-step error norm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("1-step Latent Prediction Error Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_W_offline.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Diagnostic 4: A spectral radius
# ---------------------------------------------------------------------------

def diag_A_stability(model, loader, device, num_batches, out_dir):
    print("\n=== 4. A Spectral Radius ===")
    rho_list = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            hist_x = batch['hist_x'].to(device)
            hist_u = batch['hist_u'].to(device)
            ctx = model.get_ctx(hist_x, hist_u)
            A, _ = model.get_AB(ctx)
            eigvals = torch.linalg.eigvals(A).abs().max(dim=1).values
            rho_list.append(eigvals.cpu().numpy())

    rho_all = np.concatenate(rho_list)
    pct_gt1 = (rho_all > 1.0).mean() * 100

    print(f"  rho(A): mean={rho_all.mean():.4f}, std={rho_all.std():.4f}, "
          f"min={rho_all.min():.4f}, max={rho_all.max():.4f}")
    print(f"  Pct rho > 1.0: {pct_gt1:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rho_all, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='r', linestyle='--', label='rho=1.0')
    ax.set_xlabel("Spectral Radius rho(A)")
    ax.set_ylabel("Count")
    ax.set_title(f"A Spectral Radius Distribution (>{1.0}: {pct_gt1:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_A_stability.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Diagnostic 5: ctx sensitivity
# ---------------------------------------------------------------------------

def diag_ctx_sensitivity(model, loader, device, num_batches, out_dir):
    print("\n=== 5. Context Sensitivity (ctx change → A change) ===")
    ctx_diffs = []
    A_diffs = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            hist_x = batch['hist_x'].to(device)
            hist_u = batch['hist_u'].to(device)
            b = hist_x.shape[0]
            H = hist_u.shape[1]

            if H < 2:
                continue

            ctx_full = model.get_ctx(hist_x, hist_u)
            A_full, _ = model.get_AB(ctx_full)

            ctx_shift = model.get_ctx(hist_x[:, 1:], hist_u[:, 1:])
            A_shift, _ = model.get_AB(ctx_shift)

            cd = torch.norm(ctx_full - ctx_shift, dim=-1).cpu().numpy()
            ad = torch.norm((A_full - A_shift).reshape(b, -1), dim=-1).cpu().numpy()
            ctx_diffs.append(cd)
            A_diffs.append(ad)

    ctx_diffs = np.concatenate(ctx_diffs)
    A_diffs = np.concatenate(A_diffs)
    corr = np.corrcoef(ctx_diffs, A_diffs)[0, 1]

    print(f"  ||delta_ctx||: mean={ctx_diffs.mean():.4f}, std={ctx_diffs.std():.4f}")
    print(f"  ||delta_A||_F: mean={A_diffs.mean():.4f}, std={A_diffs.std():.4f}")
    print(f"  Correlation(ctx_diff, A_diff): {corr:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ctx_diffs, A_diffs, alpha=0.3, s=5)
    ax.set_xlabel("||delta_ctx||")
    ax.set_ylabel("||delta_A||_F")
    ax.set_title(f"Context → A Sensitivity (corr={corr:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_ctx_sensitivity.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Diagnostic 6: Gaussianity check (latent process noise)
# ---------------------------------------------------------------------------

def collect_residuals(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    max_samples: int = 100_000,
) -> np.ndarray:
    """Collect one-step latent prediction residuals from the test set.

    w_k = encode(x_{k+1}) - (A(ctx) @ encode(x_k) + B(ctx) @ u_k)

    Returns:
        residuals: (N_samples, latent_dim) array.
    """
    hist_len = dataset.hist_len
    hist_stride = int(round(dataset.hist_dt / dataset.log_dt))
    pred_stride = int(round(dataset.pred_dt / dataset.log_dt))

    hist_x_offsets = list(range(-hist_len * hist_stride, 1, hist_stride))
    hist_u_offsets = list(range(-hist_len * hist_stride, 0, hist_stride))

    all_errors = []
    n_collected = 0

    model.eval()
    with torch.no_grad():
        for traj_idx in dataset.sequence_indices:
            if n_collected >= max_samples:
                break

            x_traj = dataset.x_traj_list[traj_idx]
            u_traj = dataset.u_traj_list[traj_idx]
            T = x_traj.shape[0]
            min_k = hist_len * hist_stride
            max_k = T - 1 - pred_stride

            if max_k < min_k:
                continue

            k_indices = list(range(min_k, max_k + 1, pred_stride))
            batch_size = 256

            for b_start in range(0, len(k_indices), batch_size):
                if n_collected >= max_samples:
                    break
                ks = k_indices[b_start:b_start + batch_size]

                hist_x = torch.stack([
                    x_traj[[k + o for o in hist_x_offsets]] for k in ks
                ])
                hist_u = torch.stack([
                    u_traj[[k + o for o in hist_u_offsets]] for k in ks
                ])
                x_curr = torch.stack([x_traj[k] for k in ks])
                x_next = torch.stack([x_traj[k + pred_stride] for k in ks])
                u_curr = torch.stack([u_traj[k] for k in ks])

                hist_x = hist_x.to(device, non_blocking=True)
                hist_u = hist_u.to(device, non_blocking=True)
                x_curr = x_curr.to(device, non_blocking=True)
                x_next = x_next.to(device, non_blocking=True)
                u_curr = u_curr.to(device, non_blocking=True)

                ctx = model.get_ctx(hist_x, hist_u)
                z_mu, _ = model.encode(x_curr)
                z_next_true, _ = model.encode(x_next)
                A, B = model.get_AB(ctx)
                z_next_pred = model.step(z_mu, u_curr, A, B)

                e = (z_next_true - z_next_pred).cpu().numpy()
                all_errors.append(e)
                n_collected += e.shape[0]

    residuals = np.vstack(all_errors)
    if residuals.shape[0] > max_samples:
        residuals = residuals[:max_samples]

    log.info(f"Collected {residuals.shape[0]} residual samples "
             f"(latent_dim={residuals.shape[1]})")
    return residuals


def generate_qq_data(
    residuals: np.ndarray,
    out_dir: str,
    num_dims: int = 5,
    selection: str = "largest_variance",
) -> tuple[np.ndarray, list[int]]:
    """Compute Q-Q data for selected latent dimensions and save as .mat."""
    import scipy.io as sio
    from scipy import stats as sp_stats

    N, D = residuals.shape
    num_dims = min(num_dims, D)

    means = residuals.mean(axis=0)
    stds = np.maximum(residuals.std(axis=0), 1e-12)
    normalized = (residuals - means) / stds

    if selection == "largest_variance":
        var = np.var(residuals, axis=0)
        selected_dims = np.argsort(var)[-num_dims:][::-1].tolist()
    else:
        selected_dims = np.linspace(0, D - 1, num_dims, dtype=int).tolist()

    QQ_RANGE = 3.0
    qq_theoretical, qq_empirical = [], []
    qq_dim_labels, qq_skewness, qq_kurtosis = [], [], []

    for dim in selected_dims:
        data = normalized[:, dim]
        osm, osr = sp_stats.probplot(data, dist="norm", fit=False)
        mask = (np.abs(osm) <= QQ_RANGE) & (np.abs(osr) <= QQ_RANGE)
        qq_theoretical.append(osm[mask].astype(np.float64))
        qq_empirical.append(osr[mask].astype(np.float64))
        qq_dim_labels.append(dim + 1)
        qq_skewness.append(float(sp_stats.skew(data)))
        qq_kurtosis.append(float(sp_stats.kurtosis(data, fisher=False)))

    all_skewness = np.array([
        float(sp_stats.skew(normalized[:, d])) for d in range(D)
    ])
    all_kurtosis = np.array([
        float(sp_stats.kurtosis(normalized[:, d], fisher=False)) for d in range(D)
    ])

    qq_theo_cell = np.empty(num_dims, dtype=object)
    qq_emp_cell = np.empty(num_dims, dtype=object)
    for i in range(num_dims):
        qq_theo_cell[i] = qq_theoretical[i]
        qq_emp_cell[i] = qq_empirical[i]

    mat_dict = {
        "num_samples": np.array(N, dtype=np.int64),
        "latent_dim": np.array(D, dtype=np.int64),
        "num_selected": np.array(num_dims, dtype=np.int64),
        "qq_range": np.array(QQ_RANGE),
        "selected_dims": np.array(qq_dim_labels, dtype=np.int64),
        "selected_skewness": np.array(qq_skewness),
        "selected_kurtosis": np.array(qq_kurtosis),
        "all_skewness": all_skewness,
        "all_kurtosis": all_kurtosis,
        "skewness_mean": np.array(float(np.mean(all_skewness))),
        "kurtosis_mean": np.array(float(np.mean(all_kurtosis))),
        "skewness_std": np.array(float(np.std(all_skewness))),
        "kurtosis_std": np.array(float(np.std(all_kurtosis))),
        "qq_theoretical": qq_theo_cell,
        "qq_empirical": qq_emp_cell,
    }

    mat_path = os.path.join(out_dir, "gaussianity_data.mat")
    sio.savemat(mat_path, mat_dict)
    log.info(f"Q-Q data saved: {mat_path}")

    return normalized, selected_dims


def compute_normality_stats(
    residuals: np.ndarray,
    selected_dims: list[int] | None = None,
    shapiro_cap: int = 5000,
) -> list[dict]:
    """Per-dimension normality statistics (skewness, kurtosis, Shapiro-Wilk)."""
    from scipy import stats as sp_stats

    N, D = residuals.shape
    means = residuals.mean(axis=0)
    stds = np.maximum(residuals.std(axis=0), 1e-12)
    normalized = (residuals - means) / stds

    dims = selected_dims if selected_dims is not None else list(range(D))
    results = []

    for dim in dims:
        data = normalized[:, dim]
        skew = float(sp_stats.skew(data))
        kurt = float(sp_stats.kurtosis(data, fisher=False))

        n_shapiro = min(N, shapiro_cap)
        if n_shapiro < N:
            rng = np.random.RandomState(dim)
            idx = rng.choice(N, size=n_shapiro, replace=False)
            data_shapiro = data[idx]
        else:
            data_shapiro = data
        sw_stat, sw_p = sp_stats.shapiro(data_shapiro)

        results.append({
            "dim": int(dim),
            "mean": float(means[dim]),
            "std": float(stds[dim]),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "shapiro_stat": round(float(sw_stat), 6),
            "shapiro_p": float(sw_p),
        })

    return results


def print_stats_table(stats_list: list[dict]) -> None:
    log.info("")
    log.info("Per-dimension normality statistics (normalized residuals)")
    log.info("  Ideal: skewness=0.00, kurtosis=3.00")
    log.info("-" * 80)
    log.info(f"  {'Dim':>4s}  {'Mean':>10s}  {'Std':>10s}  {'Skew':>8s}  "
             f"{'Kurt':>8s}  {'SW stat':>10s}  {'SW p':>10s}")
    log.info(f"  {'':>4s}  {'':>10s}  {'':>10s}  {'(→0)':>8s}  "
             f"{'(→3)':>8s}  {'(→1)':>10s}  {'(>0.05)':>10s}")
    log.info("-" * 80)
    for s in stats_list:
        log.info(
            f"  {s['dim']:4d}  {s['mean']:10.2e}  {s['std']:10.2e}  "
            f"{s['skewness']:8.4f}  {s['kurtosis']:8.4f}  "
            f"{s['shapiro_stat']:10.6f}  {s['shapiro_p']:10.4e}"
        )
    skews = [s['skewness'] for s in stats_list]
    kurts = [s['kurtosis'] for s in stats_list]
    log.info("-" * 80)
    log.info(f"  {'Avg':>4s}  {'':>10s}  {'':>10s}  "
             f"{np.mean(skews):8.4f}  {np.mean(kurts):8.4f}")
    log.info(f"  {'Std':>4s}  {'':>10s}  {'':>10s}  "
             f"{np.std(skews):8.4f}  {np.std(kurts):8.4f}")


def diag_gaussianity(model, cfg, device, num_dims, max_samples, out_dir):
    """Gaussianity check for latent process noise (Appendix figure)."""
    import yaml

    print("\n=== 6. Gaussianity Check (Latent Process Noise) ===")
    os.makedirs(out_dir, exist_ok=True)

    log.info("Loading test dataset...")
    dataset = make_test_dataset(cfg)
    log.info(f"Test sequences: {len(dataset.sequence_indices)}")

    log.info("Collecting one-step latent prediction residuals...")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    residuals = collect_residuals(model, dataset, dev, max_samples=max_samples)

    residual_path = os.path.join(out_dir, "latent_residuals.npy")
    np.save(residual_path, residuals)
    log.info(f"Raw residuals saved: {residual_path}")

    log.info("Computing Q-Q data...")
    normalized, selected_dims = generate_qq_data(residuals, out_dir, num_dims=num_dims)

    latent_dim = model.l_dim
    log.info("Computing normality statistics (all dimensions)...")
    stats_all = compute_normality_stats(residuals, selected_dims=list(range(latent_dim)))

    stats_selected = compute_normality_stats(residuals, selected_dims=selected_dims)
    print_stats_table(stats_selected)

    skews_all = [s['skewness'] for s in stats_all]
    kurts_all = [s['kurtosis'] for s in stats_all]

    summary = {
        "model_dir": cfg._metadata_model_dir,
        "num_samples": int(residuals.shape[0]),
        "latent_dim": int(latent_dim),
        "num_test_sequences": len(dataset.sequence_indices),
        "all_dimensions": {
            "skewness_mean": round(float(np.mean(skews_all)), 4),
            "skewness_std": round(float(np.std(skews_all)), 4),
            "skewness_max_abs": round(float(np.max(np.abs(skews_all))), 4),
            "kurtosis_mean": round(float(np.mean(kurts_all)), 4),
            "kurtosis_std": round(float(np.std(kurts_all)), 4),
        },
        "selected_dimensions": [
            {
                "dim": s["dim"],
                "skewness": s["skewness"],
                "kurtosis": s["kurtosis"],
                "shapiro_p": round(s["shapiro_p"], 6),
            }
            for s in stats_selected
        ],
    }

    yaml_path = os.path.join(out_dir, "gaussianity_summary.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    log.info(f"Summary YAML saved: {yaml_path}")

    log.info("")
    log.info("=" * 60)
    log.info("  Gaussianity Check Summary")
    log.info("=" * 60)
    log.info(f"  Samples:    {residuals.shape[0]:,}")
    log.info(f"  Latent dim: {latent_dim}")
    log.info(f"  Skewness:   {np.mean(skews_all):.4f} ± {np.std(skews_all):.4f}  (ideal: 0)")
    log.info(f"  Kurtosis:   {np.mean(kurts_all):.4f} ± {np.std(kurts_all):.4f}  (ideal: 3)")
    log.info(f"  Output:     {out_dir}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PARTY Model Diagnostics")
    parser.add_argument("--model_dir", type=str, required=True,
                        help='Model directory relative to outputs/learning_party/ '
                             '(e.g. "2026-02-25/09-22-57")')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of test batches for diagnostics 1-5")
    parser.add_argument("--jac_samples", type=int, default=32,
                        help="Number of samples for Jacobian analysis")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for standard diagnostic plots "
                             "(default: outputs/learning_party/{model_dir}/diagnostics)")
    # Gaussianity options
    parser.add_argument("--gaussianity", action="store_true",
                        help="Also run gaussianity check (diagnostic 6)")
    parser.add_argument("--num_dims", type=int, default=5,
                        help="Number of latent dims for Q-Q plot (gaussianity)")
    parser.add_argument("--max_samples", type=int, default=100_000,
                        help="Maximum residual samples for gaussianity check")
    parser.add_argument("--gauss_out_dir", type=str, default=None,
                        help="Output directory for gaussianity results "
                             "(default: outputs/learning_party/{model_dir}/gaussianity)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(
            "outputs/learning_party", args.model_dir, "diagnostics"
        )
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gauss_out_dir is None:
        args.gauss_out_dir = os.path.join(
            "outputs/learning_party", args.model_dir, "gaussianity"
        )

    print(f"Loading model from: {args.model_dir}")
    model, cfg = load_model_and_cfg(args.model_dir, args.device)
    cfg._metadata_model_dir = args.model_dir

    print("Loading test data...")
    loader = make_test_loader(cfg, batch_size=256)
    print(f"  Test samples: {len(loader.dataset)}, batches: {len(loader)}")

    diag_z_logvar(model, loader, args.device, args.num_batches, args.out_dir)
    diag_jacobian(model, loader, args.device, args.jac_samples, args.out_dir)
    diag_W_offline(model, loader, args.device, args.num_batches, args.out_dir)
    diag_A_stability(model, loader, args.device, args.num_batches, args.out_dir)
    diag_ctx_sensitivity(model, loader, args.device, args.num_batches, args.out_dir)

    if args.gaussianity:
        diag_gaussianity(
            model, cfg, args.device,
            num_dims=args.num_dims,
            max_samples=args.max_samples,
            out_dir=args.gauss_out_dir,
        )

    print(f"\nAll diagnostics saved to: {args.out_dir}")
    if args.gaussianity:
        print(f"Gaussianity results saved to: {args.gauss_out_dir}")


if __name__ == "__main__":
    main()
