"""Offline Monte-Carlo simulation to compute empirical linearization error margins.

Runs N closed-loop episodes per experiment (exp1a, 1b, 3, 2) with perturbed
initial conditions using PartySMPC, records the linearization error
|g(z) - g_tilde(z)| at each MPC solve, and outputs a per-horizon-step margin
array suitable for constraint tightening in ``LinearKOZ``.

Each experiment uses the same (q_init, q_goal) and inertia as ``3_run_mc.py``,
with q_init perturbed by a small random rotation and omega_init sampled randomly.

Supports automatic two-pass SCP calibration (``mc.two_pass_scp=true``):
  Pass 1 (SCP off): measure latent deviation distribution → pick threshold
  Pass 2 (SCP on):  re-run with SCP at that threshold → tighter margin

Usage::

    python -m scripts.90_calibrate_party
    python -m scripts.90_calibrate_party mc.num_episodes=50 mc.num_workers=4
    python -m scripts.90_calibrate_party mc.two_pass_scp=true
"""

import copy
import os
# Pin BLAS/OpenMP threads BEFORE importing numpy/torch (fork inherits parent's thread pool)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import multiprocessing as mp
import time

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from Basilisk.architecture import bskLogging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from sim.koz.koz import KeepoutZone
from sim.koz.koz_utils import perturb_quaternion
from sim.montecarlo.mc_utils import random_omega
from sim.montecarlo.single_episode import goal_diagnostics, run_episode
from sim.utils.plot_utils import (plot_sim_results, plot_control_decomposition,
                                  plot_AB_diff_diagnostics, plot_v_plan_diagnostics,
                                  plot_uncertainty_state_space, plot_latent_diagnostics)

logger = logging.getLogger(__name__)

_QUIET_LOGGERS = [
    "sim.koz.koz_utils", "sim.utils.bsk_setup", "sim.montecarlo.single_episode",
    "src.utils.party_utils", "src.model.party_interface", "src.dataset.processor",
    "sim.utils.plot_utils",
    "PartySMPC",
]

# ---------------------------------------------------------------------------
#  Helper: load experiment-specific configs (same logic as 3_run_mc.py)
# ---------------------------------------------------------------------------

def _load_sc_config(cfg: DictConfig, sc_name: str) -> DictConfig:
    """Load an experiment-specific sc config and merge into *cfg*."""
    sc_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", "sc", f"{sc_name}.yaml"
    )
    sc_override = OmegaConf.load(sc_path)
    if "defaults" in sc_override:
        sc_override = OmegaConf.create(
            {k: v for k, v in sc_override.items() if k != "defaults"}
        )
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.sc = OmegaConf.merge(cfg_copy.sc, sc_override)
    return cfg_copy


def _load_constraint_config(cfg: DictConfig, constraint_name: str) -> DictConfig:
    """Load an experiment-specific constraint config and merge into *cfg*."""
    cst_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", "constraint", f"{constraint_name}.yaml"
    )
    cst_override = OmegaConf.load(cst_path)
    if "defaults" in cst_override:
        cst_override = OmegaConf.create(
            {k: v for k, v in cst_override.items() if k != "defaults"}
        )
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.constraint = OmegaConf.merge(cfg_copy.constraint, cst_override)
    return cfg_copy


# ---------------------------------------------------------------------------
#  Worker process (multiprocessing)
# ---------------------------------------------------------------------------
_WORKER_CFG = None


def _init_worker(cfg_dict):
    """Initializer for pool workers — restore OmegaConf and suppress logs."""
    global _WORKER_CFG
    _WORKER_CFG = OmegaConf.create(cfg_dict)
    import torch
    torch.set_num_threads(1)  # env vars already set at module top; this handles PyTorch's own pool
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


def _run_episode(args):
    """Run a single MC episode.  Returns a dict with all needed data."""
    ep_idx, q_init, q_goal, omega_init, inertia = args
    cfg = _WORKER_CFG

    t0 = time.perf_counter()
    try:
        state_traj, ctrl_traj, controller = run_episode(
            cfg, q_init=q_init, q_goal=q_goal,
            omega_init=omega_init, inertia=inertia,
            controller_overrides={"record_linearization_error": True},
        )
    except Exception as e:
        return {"ep_idx": ep_idx, "error": str(e)}

    wall = time.perf_counter() - t0
    _, att_err, omega_m = goal_diagnostics(state_traj, q_goal)
    qp_mean_ms = (float(np.mean(controller.wall_times)) * 1000
                  if hasattr(controller, 'wall_times') and controller.wall_times else 0.0)

    return {
        "ep_idx": ep_idx,
        "state_traj": state_traj,
        "ctrl_traj": ctrl_traj,
        "q_init": q_init,
        "q_goal": q_goal,
        "omega_init": omega_init,
        "lin_errors": controller.get_lin_errors(),
        "lin_devs": controller.get_lin_devs(),
        "att_err_deg": att_err,
        "omega_mag": omega_m,
        "mpc_failures": controller.solve_failures,
        "qp_mean_ms": qp_mean_ms,
        "wall_time": wall,
        "v_history": (controller.get_v_history()
                      if hasattr(controller, 'get_v_history') else None),
        "z_err_norm_history": (controller.get_z_err_norm_history()
                               if hasattr(controller, 'get_z_err_norm_history') else None),
        "ctx_diff_history": (controller.get_ctx_diff_history()
                             if hasattr(controller, 'get_ctx_diff_history') else None),
        "A_diff_history": (controller.get_A_diff_history()
                           if hasattr(controller, 'get_A_diff_history') else None),
        "A_rel_diff_history": (controller.get_A_rel_diff_history()
                               if hasattr(controller, 'get_A_rel_diff_history') else None),
        "A_norm_history": (controller.get_A_norm_history()
                           if hasattr(controller, 'get_A_norm_history') else None),
        "B_diff_history": (controller.get_B_diff_history()
                           if hasattr(controller, 'get_B_diff_history') else None),
        "u_diff_history": (controller.get_u_diff_history()
                           if hasattr(controller, 'get_u_diff_history') else None),
        "B_sv_history": (controller.get_B_sv_history()
                         if hasattr(controller, 'get_B_sv_history') else None),
        "z_pred_err_history": (controller.get_z_pred_err_history()
                               if hasattr(controller, 'get_z_pred_err_history') else None),
        "q_goal_err_deg_history": (controller.get_q_goal_err_deg_history()
                                   if hasattr(controller, 'get_q_goal_err_deg_history') else None),
        "v_plan_history": (controller.get_v_plan_history()
                           if hasattr(controller, 'get_v_plan_history') else None),
        "W_history": (controller.get_W_history()
                      if hasattr(controller, 'get_W_history') else None),
        "z_history": (controller.get_z_history()
                      if hasattr(controller, 'get_z_history') else None),
        "z_goal_history": (controller.get_z_goal_history()
                           if hasattr(controller, 'get_z_goal_history') else None),
        "z_cost_Q_history": (controller.get_z_cost_Q_history()
                             if hasattr(controller, 'get_z_cost_Q_history') else None),
        "ctx_history": (controller.get_ctx_history()
                        if hasattr(controller, 'get_ctx_history') else None),
        "tighten_history": (controller.get_tighten_history()
                            if hasattr(controller, 'get_tighten_history') else None),
        "model": controller.model if hasattr(controller, 'model') else None,
        "mpc_dt": controller.mpc_dt if hasattr(controller, 'mpc_dt') else None,
    }


# ---------------------------------------------------------------------------
#  Reusable helpers for episode execution & margin computation
# ---------------------------------------------------------------------------

def _run_experiments_pass(
    cfg: DictConfig,
    experiments: list,
    num_episodes: int,
    perturb_deg: float,
    num_workers: int,
    total_episodes: int,
    exp_meta: dict,
    seed: int,
    scp_override: dict | None = None,
    label: str = "",
) -> list[dict]:
    """Run all experiments and return sorted list of result dicts.

    Args:
        scp_override: If not None, override ``controller.scp_relinearize``
            in each experiment config before spawning workers.
            E.g. ``{"enabled": True, "threshold": 0.5}``.
        label: Prefix for progress logging (e.g. "Pass 1", "Pass 2 (SCP)").
    """
    all_results: list[dict] = []
    global_ep_idx = 0
    mc_start = time.perf_counter()
    np.random.seed(seed)

    for exp in experiments:
        sc_name = exp.sc if hasattr(exp, "sc") else exp["sc"]
        exp_desc = exp.desc if hasattr(exp, "desc") else exp["desc"]
        constraint_name = exp.constraint if hasattr(exp, "constraint") else exp.get("constraint")

        # Load experiment-specific config (constraint + sc)
        exp_cfg = cfg
        if constraint_name:
            exp_cfg = _load_constraint_config(exp_cfg, constraint_name)
        exp_cfg = _load_sc_config(exp_cfg, sc_name)

        # KOZ objects for this experiment (used in plotting)
        koz_objects = KeepoutZone.from_cfg(exp_cfg.constraint.koz_list)
        boresight_list = list(exp_cfg.constraint.boresight_list)

        # Base IC from experiment sc config
        q_init_base = np.array(exp_cfg.sc.attitude.quat_init, dtype=float)
        q_goal = np.array(exp_cfg.sc.attitude.quat_goal, dtype=float)

        tag = f"[{label}] " if label else ""
        logger.info(
            "%sExperiment %s (%s): q_init=%s, q_goal=%s, constraint=%s",
            tag, sc_name, exp_desc,
            [round(x, 4) for x in q_init_base],
            [round(x, 4) for x in q_goal],
            constraint_name or "default",
        )

        exp_meta[sc_name] = {
            "koz_objects": koz_objects,
            "boresight_list": boresight_list,
            "desc": exp_desc,
        }

        # Pre-generate ICs (perturb q_init, random omega, inertia from config)
        ic_list: list[tuple] = []
        for _ in range(num_episodes):
            if perturb_deg > 0.0:
                q_init = perturb_quaternion(q_init_base, perturb_deg)
            else:
                q_init = q_init_base.copy()

            omega_init = random_omega(cfg.mc.omega_max)
            ic_list.append((global_ep_idx, q_init, q_goal, omega_init, None))
            global_ep_idx += 1

        logger.info("  Generated %d ICs for %s.", len(ic_list), sc_name)

        # Run episodes with experiment-specific config
        exp_cfg_dict = OmegaConf.to_container(exp_cfg, resolve=True)
        if scp_override is not None:
            exp_cfg_dict["controller"]["scp_relinearize"] = scp_override
        exp_results: list[dict] = []

        progress_tag = f"{label} {sc_name}" if label else sc_name
        pbar = tqdm(total=num_episodes, desc=progress_tag, unit="ep")
        if num_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(num_workers, initializer=_init_worker,
                          initargs=(exp_cfg_dict,)) as pool:
                for res in pool.imap_unordered(_run_episode, ic_list):
                    if "error" in res:
                        tqdm.write(f"  Episode {res['ep_idx'] + 1} failed: {res['error']}")
                        pbar.update(1)
                        continue
                    res["experiment"] = sc_name
                    exp_results.append(res)
                    pbar.update(1)
                    _update_pbar(pbar, res, all_results + exp_results)
        else:
            _init_worker(exp_cfg_dict)
            for ic in ic_list:
                res = _run_episode(ic)
                if "error" in res:
                    tqdm.write(f"  Episode {res['ep_idx'] + 1} failed: {res['error']}")
                    pbar.update(1)
                    continue
                res["experiment"] = sc_name
                exp_results.append(res)
                pbar.update(1)
                _update_pbar(pbar, res, all_results + exp_results)

        pbar.close()
        all_results.extend(exp_results)

    all_results.sort(key=lambda r: r["ep_idx"])
    wall = time.perf_counter() - mc_start
    logger.info("%sTotal: %d episodes in %.0fs", f"[{label}] " if label else "", len(all_results), wall)
    return all_results


def _compute_margin_stats(all_results: list[dict], horizon: int,
                          skip_initial: int = 0) -> dict:
    """Compute margin statistics from episode results.

    Args:
        all_results: List of episode result dicts.
        horizon: MPC prediction horizon.
        skip_initial: Number of initial MPC solves to skip per episode
            (removes cold-start transient from statistics).

    Returns dict with keys: margin_max, margin_max_raw, margin_mean, margin_std,
    margin_p50/p90/p95/p99, all_devs, all_max_errors, num_samples, num_skipped.

    Note: The last horizon step (N-1) is excluded from statistics because
    the warm-start pads it by copying step N-2, making its linearization
    error artificially large.  Its margin is filled with the value from
    step N-2 (monotone non-decreasing).
    """
    running_max = np.zeros(horizon)
    running_sum = np.zeros(horizon)
    running_count = np.zeros(horizon, dtype=np.int64)
    all_samples: list[np.ndarray] = []
    all_devs: list[float] = []
    all_max_errors: list[float] = []
    all_tighten_samples: list[np.ndarray] = []
    num_skipped = 0

    for res in all_results:
        lin_errors = res["lin_errors"]
        lin_devs = res["lin_devs"]
        tighten_hist = res.get("tighten_history")
        for i, err_hlm in enumerate(lin_errors):
            if i < skip_initial:
                num_skipped += 1
                continue
            max_per_step = np.max(err_hlm, axis=(1, 2))  # (H,)
            # Replace padded last horizon step with second-to-last
            if horizon >= 2:
                max_per_step[-1] = max_per_step[-2]
            running_max = np.maximum(running_max, max_per_step)
            running_sum += max_per_step
            running_count += 1
            all_samples.append(max_per_step)
            if i < len(lin_devs):
                all_devs.append(lin_devs[i])
                all_max_errors.append(float(np.max(max_per_step)))
            # Tightening per solve
            if tighten_hist is not None and i < len(tighten_hist):
                all_tighten_samples.append(tighten_hist[i])

    margin_mean = np.where(running_count > 0, running_sum / running_count, 0.0)
    margin_max_raw = running_max.copy()
    sample_matrix = np.array(all_samples)
    margin_p50 = np.percentile(sample_matrix, 50, axis=0)
    margin_p90 = np.percentile(sample_matrix, 90, axis=0)
    margin_p95 = np.percentile(sample_matrix, 95, axis=0)
    margin_p99 = np.percentile(sample_matrix, 99, axis=0)
    margin_std = np.std(sample_matrix, axis=0)

    margin_max = margin_max_raw.copy()
    for j in range(1, horizon):
        margin_max[j] = max(margin_max[j], margin_max[j - 1])

    if num_skipped > 0:
        logger.info("Skipped %d cold-start solves (%d per episode x %d episodes)",
                     num_skipped, skip_initial, len(all_results))

    # CC tightening statistics
    tighten_stats = {}
    if all_tighten_samples:
        tighten_matrix = np.array(all_tighten_samples)  # (N_solves, horizon)
        tighten_stats = {
            "tighten_mean": np.mean(tighten_matrix, axis=0),
            "tighten_std": np.std(tighten_matrix, axis=0),
            "tighten_min": np.min(tighten_matrix, axis=0),
            "tighten_max": np.max(tighten_matrix, axis=0),
            "tighten_num_samples": len(all_tighten_samples),
        }

    return {
        "margin_max": margin_max,
        "margin_max_raw": margin_max_raw,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "margin_p50": margin_p50,
        "margin_p90": margin_p90,
        "margin_p95": margin_p95,
        "margin_p99": margin_p99,
        "num_samples": len(all_samples),
        "num_skipped": num_skipped,
        "all_devs": all_devs,
        "all_max_errors": all_max_errors,
        **tighten_stats,
    }


def _print_margin_table(stats: dict, horizon: int, label: str = "") -> None:
    """Print per-step margin table to logger."""
    tag = f" [{label}]" if label else ""
    logger.info("Linearization Margin Results%s (%d samples)", tag, stats["num_samples"])
    logger.info("-" * 90)
    logger.info("  %4s  %10s  %10s  %10s  %10s  %10s  %10s  %10s",
                "step", "max_raw", "max", "p99", "p95", "p90", "p50", "mean")
    logger.info("  " + "-" * 80)
    for j in range(horizon):
        logger.info("  %4d  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f",
                     j, stats["margin_max_raw"][j], stats["margin_max"][j],
                     stats["margin_p99"][j], stats["margin_p95"][j],
                     stats["margin_p90"][j], stats["margin_p50"][j], stats["margin_mean"][j])


def _print_tighten_table(stats: dict, horizon: int) -> None:
    """Print per-step CC tightening table to logger."""
    if "tighten_mean" not in stats:
        return
    logger.info("CC Tightening (cosine units) — %d solves", stats["tighten_num_samples"])
    logger.info("-" * 90)
    logger.info("  %4s  %10s  %10s  %10s  %10s  %12s  %12s",
                "step", "mean", "std", "min", "max", "Δθ@20°", "Δθ@30°")
    logger.info("  " + "-" * 78)
    cos20 = np.cos(np.radians(20.0))
    cos30 = np.cos(np.radians(30.0))
    for j in range(horizon):
        # Exact angular tightening: Δθ = arccos(cos(θ_KOZ) - δ_cos) - θ_KOZ
        # Tightening GROWS the KOZ: effective exclusion = θ_KOZ + Δθ
        mean_cos = stats["tighten_mean"][j]
        deg_at_20 = np.degrees(np.arccos(np.clip(cos20 - mean_cos, -1, 1))) - 20.0
        deg_at_30 = np.degrees(np.arccos(np.clip(cos30 - mean_cos, -1, 1))) - 30.0
        logger.info("  %4d  %10.6f  %10.6f  %10.6f  %10.6f  %10.4f°  %10.4f°",
                     j, stats["tighten_mean"][j], stats["tighten_std"][j],
                     stats["tighten_min"][j], stats["tighten_max"][j],
                     deg_at_20, deg_at_30)


def _print_deviation_stats(all_devs: list[float], all_max_errors: list[float]) -> float | None:
    """Print latent deviation diagnostics. Returns suggested SCP threshold."""
    if not all_devs:
        return None
    devs_arr = np.array(all_devs)
    errs_arr = np.array(all_max_errors)
    logger.info("-" * 90)
    logger.info("Latent deviation ||z_opt - z_nom||_inf statistics (%d MPC solves):",
                 len(devs_arr))
    for pct in [50, 75, 90, 95, 99]:
        d_pct = np.percentile(devs_arr, pct)
        mask = devs_arr >= d_pct
        e_cond = np.mean(errs_arr[mask]) if mask.any() else 0.0
        logger.info("  p%02d  dev=%.4f  ->  mean_error|dev>=p%02d = %.4f  (n=%d)",
                    pct, d_pct, pct, e_cond, mask.sum())
    logger.info("  max  dev=%.4f  error=%.4f", devs_arr.max(), errs_arr.max())
    return None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="calibrate_party", version_base=None)
def run(cfg: DictConfig) -> None:
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_episodes: int = cfg.mc.num_episodes
    horizon: int = cfg.controller.horizon
    log_dt: float = cfg.sim.log_dt
    num_workers: int = cfg.mc.get("num_workers", 1)
    max_safe = max(1, os.cpu_count())  # logical cores (incl. hyperthreading)
    if num_workers > max_safe:
        logger.warning("num_workers=%d > cpu_count=%d. Capping.", num_workers, max_safe)
        num_workers = max_safe

    experiments = list(cfg.experiments)
    perturb_deg: float = cfg.mc.perturb_q_init_deg
    skip_initial: int = cfg.mc.get("skip_initial_solves", 0)
    two_pass_scp: bool = cfg.mc.get("two_pass_scp", False)
    scp_pct: int = cfg.mc.get("scp_threshold_percentile", 90)

    total_episodes = num_episodes * len(experiments)
    logger.info(
        "MC margin: %d experiments x %d episodes = %d total. Workers=%d. "
        "skip_initial=%d. two_pass_scp=%s",
        len(experiments), num_episodes, total_episodes, num_workers,
        skip_initial, two_pass_scp,
    )

    exp_meta: dict[str, dict] = {}

    # ---------------------------------------------------------------
    #  Pass 1: Baseline (SCP off)
    # ---------------------------------------------------------------
    pass1_label = "Pass 1 (baseline)" if two_pass_scp else ""
    all_results = _run_experiments_pass(
        cfg, experiments, num_episodes, perturb_deg, num_workers,
        total_episodes, exp_meta, seed=cfg.seed, label=pass1_label,
    )

    if not all_results:
        logger.error("No episodes completed successfully. Cannot compute margin.")
        return

    stats = _compute_margin_stats(all_results, horizon, skip_initial=skip_initial)

    logger.info("=" * 90)
    _print_margin_table(stats, horizon, label=pass1_label)
    _print_tighten_table(stats, horizon)
    _print_deviation_stats(stats["all_devs"], stats["all_max_errors"])

    # ---------------------------------------------------------------
    #  Pass 2: SCP re-linearization (auto threshold from pass 1)
    # ---------------------------------------------------------------
    scp_threshold = None
    if two_pass_scp and stats["all_devs"]:
        scp_threshold = float(np.percentile(stats["all_devs"], scp_pct))
        logger.info("=" * 90)
        logger.info(
            "Two-pass SCP: threshold=%.4f (p%d of %d deviation samples)",
            scp_threshold, scp_pct, len(stats["all_devs"]),
        )

        all_results = _run_experiments_pass(
            cfg, experiments, num_episodes, perturb_deg, num_workers,
            total_episodes, exp_meta, seed=cfg.seed,
            scp_override={"enabled": True, "threshold": scp_threshold},
            label="Pass 2 (SCP)",
        )

        if all_results:
            stats = _compute_margin_stats(all_results, horizon, skip_initial=skip_initial)
            logger.info("=" * 90)
            _print_margin_table(stats, horizon, label="Pass 2 (SCP)")
            _print_tighten_table(stats, horizon)
            _print_deviation_stats(stats["all_devs"], stats["all_max_errors"])
        else:
            logger.error("Pass 2 failed. Using pass 1 margins.")

    completed = len(all_results)
    total_wall_all = sum(r["wall_time"] for r in all_results)

    # ---------------------------------------------------------------
    #  Save per-episode data & figures (final pass only)
    # ---------------------------------------------------------------
    episodes_dir = os.path.join(output_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)

    constraint_cfg = {
        "u_min": cfg.constraint.u_min,
        "u_max": cfg.constraint.u_max,
        "ctrl_dt": float(cfg.sim.ctrl_dt),
    }

    # Move CUDA tensors to CPU so they can be pickled for spawn workers
    import torch
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        return obj

    for res in all_results:
        if res.get("model") is not None:
            res["model"] = copy.deepcopy(res["model"]).cpu()
        for key in ("W_history", "z_history", "z_goal_history", "ctx_history",
                     "z_cost_Q_history"):
            if res.get(key) is not None:
                if isinstance(res[key], list):
                    res[key] = [_to_cpu(x) for x in res[key]]
                else:
                    res[key] = _to_cpu(res[key])

    save_args = [
        (res, episodes_dir, log_dt, constraint_cfg, exp_meta[res["experiment"]])
        for res in all_results
    ]

    save_workers = min(num_workers, completed)
    logger.info("Saving %d episode data & figures (%d workers)...", completed, save_workers)
    t_save = time.perf_counter()
    if save_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(save_workers) as pool:
            pool.map(_save_episode, save_args)
    else:
        for sa in save_args:
            _save_episode(sa)
    logger.info("Saving done in %.1fs", time.perf_counter() - t_save)

    # ---------------------------------------------------------------
    #  Console: episode-level summary
    # ---------------------------------------------------------------
    att_arr = np.array([r["att_err_deg"] for r in all_results])
    omega_arr = np.array([r["omega_mag"] for r in all_results])
    total_fails = sum(r["mpc_failures"] for r in all_results)
    goal_mask = (att_arr < 5.0) & (omega_arr < 0.01)
    goal_rate = goal_mask.sum() / completed * 100
    qp_times = [r["qp_mean_ms"] for r in all_results if r["qp_mean_ms"] > 0]
    qp_mean = np.mean(qp_times) if qp_times else 0.0
    wall_times = [r["wall_time"] for r in all_results]
    wall_mean = np.mean(wall_times)

    logger.info("-" * 90)
    logger.info("Episode diagnostics:")
    logger.info("  Final att err [deg]:  mean=%.2f  std=%.2f  med=%.2f  p95=%.2f  max=%.2f",
                att_arr.mean(), att_arr.std(), np.median(att_arr),
                np.percentile(att_arr, 95), att_arr.max())
    logger.info("  Final omega [rad/s]:  mean=%.4f  std=%.4f  med=%.4f  p95=%.4f  max=%.4f",
                omega_arr.mean(), omega_arr.std(), np.median(omega_arr),
                np.percentile(omega_arr, 95), omega_arr.max())
    logger.info("  Goal (<5\u00b0, <0.01):  %.0f%%  (%d/%d)",
                goal_rate, int(goal_mask.sum()), completed)
    logger.info("  MPC failures:    %d total  (%.1f/ep)", total_fails, total_fails / completed)
    logger.info("  QP solve time:   %.1f ms mean", qp_mean)
    logger.info("  Episode wall:    %.1fs mean  (%.0fs total, %d workers)",
                wall_mean, total_wall_all, num_workers)

    # Per-experiment breakdown
    logger.info("-" * 90)
    logger.info("Per-experiment breakdown:")
    for exp in experiments:
        sc_name = exp.sc if hasattr(exp, "sc") else exp["sc"]
        exp_results = [r for r in all_results if r["experiment"] == sc_name]
        if not exp_results:
            continue
        exp_att = np.array([r["att_err_deg"] for r in exp_results])
        exp_omega = np.array([r["omega_mag"] for r in exp_results])
        exp_goal = ((exp_att < 5.0) & (exp_omega < 0.01)).sum() / len(exp_results) * 100
        logger.info("  %-25s  %3d eps  att=%.2f\u00b1%.2f\u00b0  goal=%.0f%%",
                     sc_name, len(exp_results), exp_att.mean(), exp_att.std(), exp_goal)
    logger.info("=" * 90)

    # ---------------------------------------------------------------
    #  Save results YAML
    # ---------------------------------------------------------------
    exp_counts = {}
    for exp in experiments:
        sc_name = exp.sc if hasattr(exp, "sc") else exp["sc"]
        exp_counts[sc_name] = sum(1 for r in all_results if r["experiment"] == sc_name)

    result = {
        "num_episodes": completed,
        "num_episodes_requested": total_episodes,
        "num_experiments": len(experiments),
        "episodes_per_experiment": exp_counts,
        "num_samples": stats["num_samples"],
        "horizon": int(horizon),
        "margin_max": stats["margin_max"].tolist(),
        "margin_max_raw": stats["margin_max_raw"].tolist(),
        "margin_mean": stats["margin_mean"].tolist(),
        "margin_std": stats["margin_std"].tolist(),
        "margin_p50": stats["margin_p50"].tolist(),
        "margin_p90": stats["margin_p90"].tolist(),
        "margin_p95": stats["margin_p95"].tolist(),
        "margin_p99": stats["margin_p99"].tolist(),
        "skip_initial_solves": skip_initial,
        "num_skipped": stats["num_skipped"],
        "description": "Empirical linearization error margin per horizon step (cosine units).",
    }
    if "tighten_mean" in stats:
        result["tighten"] = {
            "description": "CC tightening per horizon step (cosine units). Max over active constraints per step.",
            "num_samples": stats["tighten_num_samples"],
            "mean": stats["tighten_mean"].tolist(),
            "std": stats["tighten_std"].tolist(),
            "min": stats["tighten_min"].tolist(),
            "max": stats["tighten_max"].tolist(),
        }
    result.update({
        "scp": {
            "two_pass": two_pass_scp,
            "threshold": round(scp_threshold, 6) if scp_threshold is not None else None,
            "threshold_percentile": scp_pct if two_pass_scp else None,
        },
        "diagnostics": {
            "total_mpc_failures": total_fails,
            "att_err_deg": {
                "mean": round(float(att_arr.mean()), 2),
                "std": round(float(att_arr.std()), 2),
                "median": round(float(np.median(att_arr)), 2),
                "p95": round(float(np.percentile(att_arr, 95)), 2),
                "max": round(float(att_arr.max()), 2),
            },
            "omega_mag": {
                "mean": round(float(omega_arr.mean()), 4),
                "std": round(float(omega_arr.std()), 4),
                "median": round(float(np.median(omega_arr)), 4),
                "p95": round(float(np.percentile(omega_arr, 95)), 4),
                "max": round(float(omega_arr.max()), 4),
            },
            "goal_rate": round(float(goal_rate / 100), 4),
            "qp_solve_time_mean_ms": round(float(qp_mean), 2),
            "episode_wall_mean_s": round(float(wall_mean), 1),
            "num_workers": num_workers,
            "model_dir": str(cfg.controller.model_dir),
            "qp_solver": str(cfg.controller.qp_solver),
            "eig_floor_ratio": float(cfg.controller.eig_floor_ratio),
        },
    })
    if stats["all_devs"]:
        devs_arr = np.array(stats["all_devs"])
        result["deviation"] = {
            "p50": round(float(np.percentile(devs_arr, 50)), 6),
            "p90": round(float(np.percentile(devs_arr, 90)), 6),
            "p95": round(float(np.percentile(devs_arr, 95)), 6),
            "p99": round(float(np.percentile(devs_arr, 99)), 6),
            "max": round(float(devs_arr.max()), 6),
        }

    result_path = os.path.join(output_dir, "linearization_margin.yaml")
    with open(result_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    logger.info("Margin YAML saved to %s", result_path)

    orig_cwd = hydra.utils.get_original_cwd()
    model_margin_path = os.path.join(
        orig_cwd, "outputs", "learning_party",
        cfg.controller.model_dir, "linearization_margin.yaml",
    )
    with open(model_margin_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    logger.info("Canonical margin YAML saved to %s", model_margin_path)


def _save_episode(args):
    """Save data + generate figures for one episode.  Designed for Pool.map."""
    res, episodes_dir, log_dt, constraint_cfg, exp_meta_entry = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep_idx = res["ep_idx"]
    sc_name = res["experiment"]
    ep_dir = os.path.join(episodes_dir, sc_name, f"episode_{ep_idx + 1:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    # NPZ
    time_vec = np.arange(res["state_traj"].shape[0]) * log_dt
    np.savez(os.path.join(ep_dir, "sim_data.npz"),
             time=time_vec, state=res["state_traj"], control=res["ctrl_traj"])

    # Info YAML
    ep_info = {
        "experiment": sc_name,
        "q_init": res["q_init"].tolist(),
        "q_goal": res["q_goal"].tolist(),
        "omega_init": res["omega_init"].tolist(),
        "att_err_deg": round(res["att_err_deg"], 4),
        "omega_mag": round(res["omega_mag"], 6),
        "num_solves": len(res["lin_errors"]),
        "mpc_failures": res["mpc_failures"],
        "wall_time_s": round(res["wall_time"], 1),
    }
    with open(os.path.join(ep_dir, "episode_info.yaml"), "w") as f:
        yaml.dump(ep_info, f, default_flow_style=False, sort_keys=False)

    # Figures
    meta = exp_meta_entry
    fig_dir = os.path.join(ep_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    prefix = f"ep{ep_idx + 1:03d}"
    x_goal_plot = np.concatenate([res["q_goal"], np.zeros(3)])
    try:
        plot_sim_results(
            x_traj=res["state_traj"], u_traj=res["ctrl_traj"],
            x_goal=x_goal_plot, koz_object=meta["koz_objects"],
            boresight_list=meta["boresight_list"],
            u_min=constraint_cfg["u_min"], u_max=constraint_cfg["u_max"],
            log_dt=log_dt, figure_dir=fig_dir, save_prefix=prefix,
            print_summary=False,
        )
        if res["v_history"] is not None and len(res["v_history"]) > 0:
            plot_control_decomposition(
                Kz_history=np.zeros_like(res["v_history"]),
                v_history=res["v_history"],
                dt=float(constraint_cfg["ctrl_dt"]),
                u_min=constraint_cfg["u_min"], u_max=constraint_cfg["u_max"],
                figure_dir=fig_dir, save_prefix=prefix,
            )
        if res.get("A_diff_history") is not None and len(res["A_diff_history"]) > 0:
            plot_AB_diff_diagnostics(
                A_diff_history=res["A_diff_history"],
                B_diff_history=res["B_diff_history"],
                u_diff_history=res["u_diff_history"],
                mpc_dt=float(res["mpc_dt"]),
                figure_dir=fig_dir, save_prefix=prefix,
                B_sv_history=res.get("B_sv_history"),
                ctx_diff_history=res.get("ctx_diff_history"),
                A_rel_diff_history=res.get("A_rel_diff_history"),
                A_norm_history=res.get("A_norm_history"),
            )
        if res.get("v_plan_history") is not None and len(res["v_plan_history"]) > 1:
            plot_v_plan_diagnostics(
                v_plan_history=res["v_plan_history"],
                mpc_dt=float(res["mpc_dt"]),
                figure_dir=fig_dir, save_prefix=prefix,
            )
        if (res.get("W_history") is not None and res.get("z_history") is not None
                and res.get("ctx_history") is not None and res.get("model") is not None
                and len(res["z_history"]) > 0):
            plot_uncertainty_state_space(
                W_history=res["W_history"],
                model=res["model"],
                z_history=res["z_history"],
                ctx_history=res["ctx_history"],
                mpc_dt=float(res["mpc_dt"]),
                figure_dir=fig_dir, save_prefix=prefix,
            )
        if (res.get("z_history") is not None and res.get("z_goal_history") is not None
                and res.get("ctx_history") is not None and res.get("model") is not None
                and len(res["z_history"]) > 0 and len(res["z_goal_history"]) > 0):
            plot_latent_diagnostics(
                z_history=res["z_history"],
                z_goal_history=res["z_goal_history"],
                model=res["model"],
                ctx_history=res["ctx_history"],
                mpc_dt=float(res["mpc_dt"]),
                figure_dir=fig_dir, save_prefix=prefix,
                z_cost_Q_history=res.get("z_cost_Q_history"),
            )
        plt.close("all")
    except Exception as e:
        logger.warning("Episode %d: plotting failed: %s", ep_idx + 1, e)


def _update_pbar(pbar, res, all_results):
    """Update tqdm postfix with last and average attitude error."""
    att_errs = [r["att_err_deg"] for r in all_results if "att_err_deg" in r]
    pbar.set_postfix(
        last=f"{res['att_err_deg']:.1f}\u00b0",
        avg=f"{np.mean(att_errs):.1f}\u00b0" if att_errs else "-",
    )


if __name__ == "__main__":
    run()
