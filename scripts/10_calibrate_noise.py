"""
Offline calibration of process noise covariance (Sigma_w) for L-SMPC and UKF-SMPC.

From a single perturbed-inertia calibration dataset, computes:

  1. L-SMPC  Sigma_w:  prediction errors using NOMINAL inertia
     → captures model mismatch + disturbances (the only uncertainty source)
  2. UKF-SMPC Sigma_w: prediction errors using each sequence's ACTUAL inertia
     → captures non-parametric disturbances only (parametric handled by estimator)
  3. P_J_0_diag:       covariance of inertia deviations (j_vec 6D) from nominal
     → initial inertia uncertainty for the UKF-SMPC augmented estimator

Output files (Hydra noise config group):
    configs/noise/lsmpc_exp1a.yaml   (Sigma_w_diag)
    configs/noise/ukf_exp1a.yaml     (Sigma_w_diag + P_J_0_diag)

Usage:
    python -m scripts.10_calibrate_noise sc=exp1a data.name=calib_perturb_exp1a
"""

import os
import logging

import numpy as np
import h5py
import yaml

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.utils.math_numpy import attitude_dyn_dt_batch, j_mat_to_vec

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#   Core computation
# ---------------------------------------------------------------------------

def collect_prediction_errors(
    h5_path: str,
    state_key: str,
    control_key: str,
    pred_dt: float,
    log_dt: float,
    j_nom: np.ndarray,
    j_nom_inv: np.ndarray,
    integrator_steps: int = 5,
    use_actual_inertia: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect one-step prediction errors across all sequences.

    When use_actual_inertia=False (L-SMPC mode):
        Uses nominal J for all predictions → residuals include model mismatch.

    When use_actual_inertia=True (UKF-SMPC mode):
        Uses each sequence's actual J → residuals are non-parametric only.
    """
    pred_stride = int(round(pred_dt / log_dt))
    assert pred_stride >= 1, (
        f"Invalid pred_stride={pred_stride} (pred_dt={pred_dt}, log_dt={log_dt})"
    )

    errors_7d_list: list[np.ndarray] = []

    with h5py.File(h5_path, 'r') as hf:
        ts = hf['timeseries']
        seq_keys = sorted(ts.keys(), key=lambda name: int(name.split("_")[-1]))

        for seq_key in seq_keys:
            grp = ts[seq_key]
            states = np.array(grp[state_key], dtype=np.float64)      # (T, 7)
            controls = np.array(grp[control_key], dtype=np.float64)  # (T, 3)

            # Per-sequence inertia (for UKF-SMPC mode)
            if use_actual_inertia:
                j_vec = np.array(grp['inertia'][0], dtype=np.float64)  # (6,)
                j_seq = np.array([
                    [j_vec[0], j_vec[1], j_vec[2]],
                    [j_vec[1], j_vec[3], j_vec[4]],
                    [j_vec[2], j_vec[4], j_vec[5]],
                ], dtype=np.float64)
                j_seq_inv = np.linalg.inv(j_seq)
            else:
                j_seq = j_nom
                j_seq_inv = j_nom_inv

            max_t = min(len(controls), len(states) - pred_stride)
            t_indices = np.arange(0, max_t, pred_stride)
            if len(t_indices) == 0:
                continue

            x_batch = states[t_indices]                    # (M, 7)
            u_batch = controls[t_indices]                  # (M, 3)
            x_actual_batch = states[t_indices + pred_stride]  # (M, 7)

            x_pred_batch = attitude_dyn_dt_batch(
                x_batch, u_batch, pred_dt, j_seq, j_seq_inv,
                steps=integrator_steps)                    # (M, 7)

            # Fix quaternion sign ambiguity (batch)
            dots = np.sum(x_pred_batch[:, :4] * x_actual_batch[:, :4], axis=1)
            sign = np.where(dots < 0, -1.0, 1.0)[:, None]
            q_pred = x_pred_batch[:, :4] * sign

            errors_7d_list.append(
                np.concatenate([q_pred - x_actual_batch[:, :4],
                                x_pred_batch[:, 4:] - x_actual_batch[:, 4:]], axis=1)
            )

    errors_7d = np.concatenate(errors_7d_list, axis=0) if errors_7d_list else np.empty((0, 7))
    mode_str = "actual-J" if use_actual_inertia else "nominal-J"
    log.info(
        f"Collected {len(errors_7d)} prediction error samples ({mode_str}) "
        f"from {len(seq_keys)} sequences (pred_stride={pred_stride})"
    )
    return errors_7d


def collect_inertia_deviations(
    h5_path: str,
    j_nom: np.ndarray,
) -> np.ndarray:
    """Collect j_vec deviations (actual - nominal) across all sequences.

    Returns:
        deviations: (N_seq, 6) array of j_vec deviations.
    """
    j_nom_vec = j_mat_to_vec(j_nom)
    devs = []
    with h5py.File(h5_path, 'r') as hf:
        ts = hf['timeseries']
        for seq_key in sorted(ts.keys(), key=lambda name: int(name.split("_")[-1])):
            j_vec = np.array(ts[seq_key]['inertia'][0], dtype=np.float64)
            devs.append(j_vec - j_nom_vec)
    return np.array(devs, dtype=np.float64)


# ---------------------------------------------------------------------------
#   Statistics
# ---------------------------------------------------------------------------

def compute_statistics(errors: np.ndarray) -> dict:
    mean = np.mean(errors, axis=0)
    std = np.std(errors, axis=0)
    var = np.var(errors, axis=0)
    cov = np.cov(errors.T)

    percentiles = {}
    for p in [1, 5, 25, 50, 75, 95, 99]:
        percentiles[p] = np.percentile(errors, p, axis=0)

    return {
        "N": len(errors),
        "mean": mean,
        "std": std,
        "var": var,
        "cov": cov,
        "percentiles": percentiles,
    }


def print_diagnostics(stats_7d: dict, label: str = "") -> None:
    dim_names_7d = ["dq0", "dq1", "dq2", "dq3", "dw1", "dw2", "dw3"]

    log.info(f"\n{'='*70}")
    log.info(f"  {label} Prediction Error Diagnostics  (N = {stats_7d['N']})")
    log.info(f"{'='*70}")

    log.info("\n--- 7D errors (quaternion + omega) ---")
    for i, name in enumerate(dim_names_7d):
        log.info(
            f"  {name:>8s}:  mean={stats_7d['mean'][i]:+.2e}  "
            f"std={stats_7d['std'][i]:.2e}  "
            f"p5={stats_7d['percentiles'][5][i]:+.2e}  "
            f"p95={stats_7d['percentiles'][95][i]:+.2e}"
        )

    log.info(f"\n  Sigma_w_diag (7D variance): {_fmt_list(stats_7d['var'])}")

    # Bias check
    bias_ratio = np.abs(stats_7d['mean']) / np.maximum(stats_7d['std'], 1e-12)
    if np.any(bias_ratio > 0.1):
        log.warning(
            f"  Non-zero mean bias (|mean/std| > 0.1): {_fmt_list(bias_ratio)}"
        )

    # Off-diagonal correlation check
    cov_7d = stats_7d['cov']
    std_outer = np.outer(stats_7d['std'], stats_7d['std'])
    corr_7d = np.where(std_outer > 1e-16, cov_7d / std_outer, 0.0)
    np.fill_diagonal(corr_7d, 0.0)
    max_corr = np.max(np.abs(corr_7d))
    log.info(f"\n  Max off-diagonal |correlation| (7D): {max_corr:.4f}")
    if max_corr > 0.3:
        log.warning(
            f"  Significant off-diagonal coupling (|corr|={max_corr:.4f} > 0.3). "
            f"Consider full covariance."
        )


def _fmt_list(arr: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.2e}" for v in arr) + "]"


# ---------------------------------------------------------------------------
#   Output — Hydra noise config yaml
# ---------------------------------------------------------------------------

def save_lsmpc_config(
    output_path: str,
    stats_7d: dict,
    pred_dt: float,
    j_nom_diag: list[float],
    h5_meta: dict,
) -> None:
    """Save L-SMPC noise config (Sigma_w only, includes model mismatch)."""
    metadata = {
        "calibration_type": "disturbance_and_model_mismatch",
        "controller": "lsmpc",
        "num_samples": int(stats_7d['N']),
        "pred_dt": float(pred_dt),
        "nominal_inertia_diag": j_nom_diag,
        "error_mean_7d": [float(v) for v in stats_7d['mean']],
        "error_std_7d": [float(v) for v in stats_7d['std']],
        "perturb_pct": float(h5_meta.get("perturb_pct", 0)),
        "off_diag_pct": float(h5_meta.get("off_diag_pct", h5_meta.get("perturb_pct", 0))),
    }

    output = {
        "Sigma_w_diag": [float(v) for v in stats_7d['var']],
        "_metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    log.info(f"Saved L-SMPC noise config to {output_path}")


def save_ukf_config(
    output_path: str,
    stats_7d: dict,
    p_j_0_diag: list[float],
    pred_dt: float,
    j_nom_diag: list[float],
    h5_meta: dict,
) -> None:
    """Save UKF-SMPC noise config (non-parametric Sigma_w + P_J_0_diag)."""
    metadata = {
        "calibration_type": "disturbance_only",
        "controller": "ukf_smpc",
        "num_samples": int(stats_7d['N']),
        "pred_dt": float(pred_dt),
        "nominal_inertia_diag": j_nom_diag,
        "error_mean_7d": [float(v) for v in stats_7d['mean']],
        "error_std_7d": [float(v) for v in stats_7d['std']],
        "perturb_pct": float(h5_meta.get("perturb_pct", 0)),
        "off_diag_pct": float(h5_meta.get("off_diag_pct", h5_meta.get("perturb_pct", 0))),
    }

    output = {
        "Sigma_w_diag": [float(v) for v in stats_7d['var']],
        "P_J_0_diag": p_j_0_diag,
        "_metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    log.info(f"Saved UKF-SMPC noise config to {output_path}")


# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="calibrate_noise", version_base=None)
def main(cfg: DictConfig):
    log.info("Noise Calibration (L-SMPC + UKF-SMPC)")

    # --- Nominal inertia ---
    inertia_arr = np.array(cfg.sc.inertia, dtype=np.float64)
    if inertia_arr.size != 9:
        raise ValueError(f"Inertia must be 9 elements (3x3 row-major), got {inertia_arr.size}")
    j_nom = inertia_arr.reshape(3, 3)
    j_nom_inv = np.linalg.inv(j_nom)
    j_nom_diag = np.diag(j_nom).tolist()
    log.info(f"Nominal inertia (diagonal): {j_nom_diag}")

    # --- H5 data path ---
    data_root = to_absolute_path(cfg.data.data_root)
    h5_filename = (
        f"{cfg.data.name}_{cfg.data.num_seqs}_{cfg.data.sim_time}_{cfg.data.log_dt}.h5"
    )
    h5_path = os.path.join(data_root, h5_filename)
    log.info(f"Loading data from {h5_path}")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Calibration data not found: {h5_path}")

    # --- Read H5 metadata ---
    h5_meta = {}
    with h5py.File(h5_path, 'r') as hf:
        if 'metadata' in hf:
            for k, v in hf['metadata'].attrs.items():
                h5_meta[k] = v if not hasattr(v, 'tolist') else v.tolist()
    log.info(f"Data inertia_mode: {h5_meta.get('inertia_mode', 'unknown')}")

    integrator_steps = int(cfg.get('integrator_steps', 5))
    pred_dt = float(cfg.data.pred_dt)
    log_dt = float(cfg.data.log_dt)
    common_kw = dict(
        h5_path=h5_path,
        state_key=cfg.data.state_key,
        control_key=cfg.data.control_key,
        pred_dt=pred_dt,
        log_dt=log_dt,
        j_nom=j_nom,
        j_nom_inv=j_nom_inv,
        integrator_steps=integrator_steps,
    )

    # =====================================================================
    #  1. L-SMPC: prediction errors with NOMINAL inertia
    # =====================================================================
    log.info("\n>>> L-SMPC: collecting errors with nominal inertia <<<")
    lsmpc_7d = collect_prediction_errors(**common_kw, use_actual_inertia=False)
    lsmpc_stats_7d = compute_statistics(lsmpc_7d)
    print_diagnostics(lsmpc_stats_7d, label="L-SMPC (nominal J)")

    # =====================================================================
    #  2. UKF-SMPC: prediction errors with per-sequence ACTUAL inertia
    # =====================================================================
    log.info("\n>>> UKF-SMPC: collecting errors with actual inertia <<<")
    ukf_7d = collect_prediction_errors(**common_kw, use_actual_inertia=True)
    ukf_stats_7d = compute_statistics(ukf_7d)
    print_diagnostics(ukf_stats_7d, label="UKF-SMPC (actual J)")

    # =====================================================================
    #  3. P_J_0_diag: inertia deviation covariance (j_vec 6D)
    # =====================================================================
    log.info("\n>>> Computing P_J_0_diag from inertia deviations <<<")
    j_devs = collect_inertia_deviations(h5_path, j_nom)
    p_j_0_diag = np.var(j_devs, axis=0).tolist()
    p_j_0_diag_fmt = [float(v) for v in p_j_0_diag]
    log.info(f"  P_J_0_diag (variance of j_vec deviations): {_fmt_list(np.array(p_j_0_diag))}")
    log.info(f"  J deviation mean: {_fmt_list(np.mean(j_devs, axis=0))}")
    log.info(f"  J deviation std:  {_fmt_list(np.std(j_devs, axis=0))}")

    # =====================================================================
    #  4. Save noise configs
    # =====================================================================
    output_dir = to_absolute_path(cfg.output_dir)
    exp_name = cfg.exp_name

    lsmpc_path = os.path.join(output_dir, f"lsmpc_{exp_name}.yaml")
    ukf_path = os.path.join(output_dir, f"ukf_{exp_name}.yaml")

    save_lsmpc_config(lsmpc_path, lsmpc_stats_7d, pred_dt, j_nom_diag, h5_meta)
    save_ukf_config(ukf_path, ukf_stats_7d, p_j_0_diag_fmt, pred_dt, j_nom_diag, h5_meta)

    # --- Summary ---
    log.info(f"\n{'='*70}")
    log.info(f"  L-SMPC  Sigma_w_diag: {_fmt_list(lsmpc_stats_7d['var'])}")
    log.info(f"  UKF     Sigma_w_diag: {_fmt_list(ukf_stats_7d['var'])}")
    log.info(f"  P_J_0_diag:           {_fmt_list(np.array(p_j_0_diag))}")
    log.info(f"  Ratio (L-SMPC / UKF): {_fmt_list(lsmpc_stats_7d['var'] / np.maximum(ukf_stats_7d['var'], 1e-30))}")
    log.info(f"\n  Hydra usage:")
    log.info(f"    L-SMPC:    noise=lsmpc_{exp_name}")
    log.info(f"    UKF-SMPC:  noise=ukf_{exp_name}")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
