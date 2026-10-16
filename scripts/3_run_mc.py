"""Monte-Carlo validation of KOZ constraint satisfaction across controllers.

Runs N experiments × (optional alpha sweep) × (optional margin settings) × M
episodes and reports per-step and per-episode KOZ constraint violation rates,
plus controller performance metrics (control effort, solver timing, etc.).

Each experiment specifies its own constraint config (KOZ geometry) and has
pre-generated base (q_init, q_goal) attitudes stored in its sc config.  The
goal attitude q_goal is kept fixed across all episodes; q_init is perturbed
by a small random rotation (configurable via ``mc.perturb_q_init_deg``).
Angular velocity omega_init is sampled independently each episode.

Supports any controller that follows the Basilisk SysModel interface
(NMPC, RampNet, ASGPSMPC, PartySMPC, …).  Chance-constraint alpha
sweeps and linearization margin comparisons are automatically enabled only
when the chosen controller supports them.

Supports multiprocessing (``mc.num_workers``).  When ``num_workers > 1``,
MPC step-level timing is *not* collected (thread contention makes it
unreliable).  Run with ``num_workers=1`` for accurate timing measurement.

Usage::

    python -m scripts.3_run_mc                         # default (party_smpc)
    python -m scripts.3_run_mc controller=nmpc         # NMPC
    python -m scripts.3_run_mc controller=asgp_smpc     # Adaptive SGP-MPC
    python -m scripts.3_run_mc mc.num_episodes=20
    python -m scripts.3_run_mc mc.num_workers=4
"""

import copy
import json
import logging
import os
# Pin BLAS/OpenMP threads BEFORE importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing as mp
import time

import hydra
import numpy as np
import scipy.io
from Basilisk.architecture import bskLogging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

from sim.koz.koz import KeepoutZone
from sim.koz.koz_utils import perturb_quaternion
from sim.montecarlo.mc_utils import random_omega
from sim.montecarlo.single_episode import (
    check_koz_violations,
    compute_control_effort,
    goal_diagnostics,
    run_episode,
)
from sim.utils.math_utils import quat2boresight_batch
from sim.utils.plot_utils import plot_mc_koz_separation, plot_mc_boresight_3d

logger = logging.getLogger(__name__)

# Suppress verbose per-episode logging from sub-modules
_QUIET_LOGGERS = [
    "sim.koz.koz_utils", "sim.utils.bsk_setup", "sim.montecarlo.single_episode",
    "src.utils.party_utils", "src.model.party_interface", "src.dataset.processor",
    "PartySMPC", "NMPC", "RampNet", "ASGP_SMPC", "UKF_SMPC", "L_SMPC",
    "ConvexPotential",
]


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


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


def _load_noise_config(cfg: DictConfig, noise_name: str) -> DictConfig:
    """Load an experiment-specific noise config and merge into *cfg*."""
    noise_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", "noise", f"{noise_name}.yaml"
    )
    noise_override = OmegaConf.load(noise_path)
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.noise = OmegaConf.merge(cfg_copy.get("noise", {}), noise_override)
    return cfg_copy


def _safe_stats(arr: list[float]) -> dict:
    """Compute summary stats from a list, returning NaN-safe dict."""
    if not arr:
        return {"mean": None, "std": None, "min": None, "max": None, "p95": None}
    a = np.array(arr)
    return {
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "min": round(float(np.min(a)), 4),
        "max": round(float(np.max(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


# ---------------------------------------------------------------------------
#  Worker process (multiprocessing)
# ---------------------------------------------------------------------------
_WORKER_CFG = None
_WORKER_KOZ = None
_WORKER_BS_B = None
_WORKER_BS_CFGS = None
_WORKER_LOG_DT = None
_WORKER_VIOL_MARGIN_DEG = 0.0


def _init_mc_worker(cfg_dict, boresight_B, boresight_cfgs, log_dt, viol_margin_deg=0.0):
    """Initializer for pool workers — restore OmegaConf and suppress logs."""
    global _WORKER_CFG, _WORKER_KOZ, _WORKER_BS_B, _WORKER_BS_CFGS, _WORKER_LOG_DT, _WORKER_VIOL_MARGIN_DEG
    _WORKER_CFG = OmegaConf.create(cfg_dict)
    _WORKER_KOZ = KeepoutZone.from_cfg(_WORKER_CFG.constraint.koz_list)
    _WORKER_BS_B = boresight_B
    _WORKER_BS_CFGS = boresight_cfgs
    _WORKER_LOG_DT = log_dt
    _WORKER_VIOL_MARGIN_DEG = viol_margin_deg
    import torch
    torch.set_num_threads(1)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)


def _run_mc_episode(args):
    """Run a single MC episode.  Returns a dict with all needed data."""
    ep_idx, q_init, q_goal, omega_init = args

    t0 = time.perf_counter()
    try:
        state_traj, ctrl_traj, controller = run_episode(
            _WORKER_CFG, q_init=q_init, q_goal=q_goal,
            omega_init=omega_init, inertia=None,
        )
    except Exception as e:
        return {"ep_idx": ep_idx, "error": str(e)}

    ep_wall = time.perf_counter() - t0

    # KOZ violations
    violated_steps, total_steps, min_gap, ep_koz_traces = check_koz_violations(
        state_traj, _WORKER_BS_B, _WORKER_KOZ, return_traces=True,
        viol_margin_deg=_WORKER_VIOL_MARGIN_DEG,
    )

    # Boresight trajectories (inertial frame)
    boresight_trajs = {}
    for bs_cfg in _WORKER_BS_CFGS:
        bs_vec = np.array(bs_cfg["vec"], dtype=float)
        bs_N = quat2boresight_batch(state_traj[:, :4], bs_vec)
        boresight_trajs[bs_cfg["name"]] = bs_N

    # Goal diagnostics
    reached, att_err_deg, omega_mag = goal_diagnostics(state_traj, q_goal)

    # Control effort
    effort = compute_control_effort(ctrl_traj, dt=_WORKER_LOG_DT)

    # MPC solver stats
    tt = list(getattr(controller, 'total_times', []))
    wt = list(getattr(controller, 'wall_times', []))

    # Profile breakdown (PartySMPC only)
    profile_accum = None
    raw = getattr(controller, '_profile_accum', None)
    if raw and raw.get('qp'):
        profile_accum = {k: list(v) for k, v in raw.items()}

    # Solver statuses (NMPC / CasADi-IPOPT controllers)
    solver_statuses = list(getattr(controller, 'solver_statuses', []))

    return {
        "ep_idx": ep_idx,
        "q_init": q_init,
        "q_goal": q_goal,
        "omega_init": omega_init,
        "violated": violated_steps > 0,
        "violated_steps": int(violated_steps),
        "total_steps": int(total_steps),
        "step_rate": violated_steps / total_steps if total_steps > 0 else 0.0,
        "min_gap": float(min_gap),
        "koz_traces": ep_koz_traces,
        "boresight_trajs": boresight_trajs,
        "q_traj": state_traj[:, :4].copy(),
        "omega_traj": state_traj[:, 4:7].copy(),
        "reached": bool(reached),
        "att_err_deg": att_err_deg,
        "omega_mag": omega_mag,
        "effort": effort,
        "solve_failures": int(getattr(controller, 'solve_failures', 0)),
        "total_times": tt,
        "wall_times": wt,
        "profile_accum": profile_accum,
        "solver_statuses": solver_statuses,
        "ep_wall": ep_wall,
    }


@hydra.main(config_path="../configs", config_name="mc_party", version_base=None)
def run(cfg: DictConfig) -> None:
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_episodes: int = cfg.mc.num_episodes
    log_dt: float = cfg.sim.log_dt
    num_workers: int = cfg.mc.get("num_workers", 1)
    max_safe = max(1, os.cpu_count())
    if num_workers > max_safe:
        logger.warning("num_workers=%d > cpu_count=%d. Capping.", num_workers, max_safe)
        num_workers = max_safe

    experiments = list(cfg.experiments)

    # Detect controller capabilities from config keys
    has_cc = "cc" in cfg.controller
    has_margin = "lin_margin_file" in cfg.controller

    # Alpha sweep: only when controller supports chance constraints (cc config)
    if has_cc and cfg.get("alpha_list"):
        alpha_list = list(cfg.alpha_list)
    else:
        alpha_list = [None]  # single pass, no alpha override

    # Margin sweep: only when controller supports linearization margin
    if has_margin and cfg.get("margin_files"):
        margin_files = list(cfg.margin_files)
    else:
        margin_files = [None]  # single pass, no margin override

    total_combos = len(experiments) * len(alpha_list) * len(margin_files)
    controller_name = cfg.controller.get("_target_", "unknown").rsplit(".", 1)[-1]
    parallel = num_workers > 1
    logger.info(
        "MC validation [%s]: %d experiments × %d alphas × %d margin_settings × %d episodes = %d total runs  (workers=%d)",
        controller_name,
        len(experiments), len(alpha_list), len(margin_files), num_episodes,
        total_combos * num_episodes,
        num_workers,
    )
    if parallel:
        logger.info("Parallel mode: MPC step-level timing will NOT be collected.")

    # Perturbation config
    perturb_deg: float = cfg.mc.get("perturb_q_init_deg", 0.0)
    viol_margin_deg: float = cfg.mc.viol_margin_deg

    # Config snapshot for reproducibility
    config_snapshot = {
        "controller": controller_name,
        "ctrl_dt": float(cfg.sim.ctrl_dt),
        "sim_time": float(cfg.sim.sim_time),
        "num_episodes": num_episodes,
        "seed": int(cfg.seed),
        "perturb_q_init_deg": perturb_deg,
        "num_workers": num_workers,
    }
    # Add controller-specific fields when available
    for key in ("model_dir", "qp_solver", "horizon"):
        if key in cfg.controller:
            config_snapshot[key] = str(cfg.controller[key])

    np.random.seed(cfg.seed)

    results: dict[str, dict] = {}
    combo_idx = 0
    global_mpc_step_times: list[float] = []  # only populated when num_workers == 1
    global_profile_accum: dict[str, list[float]] = {}  # breakdown (sequential only)
    global_solver_statuses: list[str] = []  # IPOPT return_status (sequential only)
    mc_start_time = time.perf_counter()

    for exp in experiments:
        sc_name = exp.sc if hasattr(exp, "sc") else exp["sc"]
        exp_desc = exp.desc if hasattr(exp, "desc") else exp["desc"]
        exp_tag = sc_name

        # ----------------------------------------------------------
        #  Load per-experiment constraint config
        # ----------------------------------------------------------
        constraint_name = exp.constraint if hasattr(exp, "constraint") else exp.get("constraint")
        if constraint_name:
            exp_base_cfg = _load_constraint_config(cfg, constraint_name)
        else:
            exp_base_cfg = cfg

        # ----------------------------------------------------------
        #  Load per-experiment noise config (for UKF-SMPC)
        # ----------------------------------------------------------
        noise_name = exp.noise if hasattr(exp, "noise") else exp.get("noise") if isinstance(exp, dict) else None
        if noise_name:
            exp_base_cfg = _load_noise_config(exp_base_cfg, noise_name)

        koz_objects = KeepoutZone.from_cfg(exp_base_cfg.constraint.koz_list)
        boresight_B = np.array(
            [np.array(bs.vec, dtype=float) for bs in exp_base_cfg.constraint.boresight_list]
        )
        boresight_B = boresight_B / np.linalg.norm(boresight_B, axis=1, keepdims=True)

        # Boresight configs as plain dicts (picklable for spawn)
        boresight_cfgs = [
            {"name": bs.name, "vec": list(bs.vec)}
            for bs in exp_base_cfg.constraint.boresight_list
        ]

        # ----------------------------------------------------------
        #  Load fixed base IC pair from sc config
        # ----------------------------------------------------------
        sc_cfg_tmp = _load_sc_config(exp_base_cfg, sc_name)
        q_init_base = np.array(sc_cfg_tmp.sc.attitude.quat_init, dtype=float)
        q_goal = np.array(sc_cfg_tmp.sc.attitude.quat_goal, dtype=float)

        logger.info(
            "Base ICs for %s: q_init=%s, q_goal=%s (constraint=%s)",
            sc_name,
            [round(x, 4) for x in q_init_base],
            [round(x, 4) for x in q_goal],
            constraint_name or "default",
        )

        for alpha in alpha_list:
            for margin_file in margin_files:
                combo_idx += 1
                margin_label = "with_margin" if margin_file else "no_margin"

                # Build combo key depending on which axes are active
                parts = [sc_name]
                if alpha is not None:
                    parts.append(f"alpha{alpha:.2f}")
                if has_margin:
                    parts.append(margin_label)
                combo_key = "/".join(parts)

                logger.info(
                    "[%d/%d] %s  (alpha=%s, margin=%s)",
                    combo_idx, total_combos, exp_desc,
                    f"{alpha:.2f}" if alpha is not None else "N/A",
                    margin_label,
                )

                # Build experiment-specific config (sc + constraint overrides)
                exp_cfg = _load_sc_config(exp_base_cfg, sc_name)

                # Per-experiment exp_name for model loading (defaults to sc_name)
                exp_name = exp.exp_name if hasattr(exp, "exp_name") else exp.get("exp_name", sc_name) if isinstance(exp, dict) else sc_name
                if "exp_name" in exp_cfg:
                    OmegaConf.update(exp_cfg, "exp_name", exp_name)
                if "exp_name" in exp_cfg.get("controller", {}):
                    OmegaConf.update(exp_cfg, "controller.exp_name", exp_name)

                if alpha is not None and has_cc:
                    OmegaConf.update(exp_cfg, "controller.cc.alpha", float(alpha))
                if has_margin:
                    OmegaConf.update(
                        exp_cfg, "controller.lin_margin_file",
                        str(margin_file) if margin_file else None,
                    )

                # Pre-generate ICs
                ic_list = []
                for ep in range(num_episodes):
                    if perturb_deg > 0.0:
                        q_init = perturb_quaternion(q_init_base, perturb_deg)
                    else:
                        q_init = q_init_base.copy()
                    omega_init = random_omega(cfg.mc.omega_max)
                    ic_list.append((ep, q_init, q_goal, omega_init))

                # Run episodes
                exp_cfg_dict = OmegaConf.to_container(exp_cfg, resolve=True)
                ep_results: list[dict] = []
                combo_start = time.perf_counter()
                pbar = tqdm(total=num_episodes, desc=exp_desc, unit="ep")

                if parallel:
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(
                        num_workers,
                        initializer=_init_mc_worker,
                        initargs=(exp_cfg_dict, boresight_B, boresight_cfgs, log_dt, viol_margin_deg),
                    ) as pool:
                        for res in pool.imap_unordered(_run_mc_episode, ic_list):
                            if "error" in res:
                                tqdm.write(f"  Episode {res['ep_idx'] + 1} failed: {res['error']}")
                                pbar.update(1)
                                continue
                            ep_results.append(res)
                            pbar.update(1)
                            pbar.set_postfix(
                                last=f"{res['att_err_deg']:.1f}\u00b0",
                                avg=f"{np.mean([r['att_err_deg'] for r in ep_results]):.1f}\u00b0",
                            )
                else:
                    _init_mc_worker(exp_cfg_dict, boresight_B, boresight_cfgs, log_dt, viol_margin_deg)
                    for ic in ic_list:
                        res = _run_mc_episode(ic)
                        if "error" in res:
                            tqdm.write(f"  Episode {res['ep_idx'] + 1} failed: {res['error']}")
                            pbar.update(1)
                            continue
                        ep_results.append(res)
                        # Collect MPC step times (sequential only)
                        global_mpc_step_times.extend(res["total_times"])
                        if res.get("profile_accum"):
                            for k, v in res["profile_accum"].items():
                                global_profile_accum.setdefault(k, []).extend(v)
                        global_solver_statuses.extend(res.get("solver_statuses", []))
                        pbar.update(1)
                        pbar.set_postfix(
                            last=f"{res['att_err_deg']:.1f}\u00b0",
                            avg=f"{np.mean([r['att_err_deg'] for r in ep_results]):.1f}\u00b0",
                        )

                pbar.close()
                ep_results.sort(key=lambda r: r["ep_idx"])
                combo_wall = time.perf_counter() - combo_start
                completed = len(ep_results)

                if completed == 0:
                    logger.warning("  No episodes completed for %s.", combo_key)
                    results[combo_key] = {
                        "experiment": sc_name,
                        "desc": exp_desc,
                        "alpha": float(alpha) if alpha is not None else None,
                        "margin": margin_label if has_margin else None,
                        "margin_file": str(margin_file) if margin_file else None,
                        "completed_episodes": 0,
                        "episode_violation_rate": None,
                        "step_violation_rate": None,
                        "target_violation_rate": round(1 - alpha, 4) if alpha is not None else None,
                    }
                    continue

                # --- Aggregate statistics from worker results ---
                episode_violated = [r["violated"] for r in ep_results]
                step_violation_rates = [r["step_rate"] for r in ep_results]
                min_koz_gaps = [r["min_gap"] for r in ep_results]
                goal_reached_list = [r["reached"] for r in ep_results]
                control_efforts = [r["effort"] for r in ep_results]
                att_errors = [r["att_err_deg"] for r in ep_results]
                omega_mags = [r["omega_mag"] for r in ep_results]
                episode_wall_times = [r["ep_wall"] for r in ep_results]
                mpc_fail_counts = [r["solve_failures"] for r in ep_results]

                # Collect per-episode MPC timing
                mpc_solve_times_all = []
                mpc_solve_times_mean = []
                for r in ep_results:
                    tt = r["total_times"]
                    if tt:
                        mpc_solve_times_mean.append(float(np.mean(tt)))
                        mpc_solve_times_all.extend(tt)

                # KOZ traces
                koz_traces: dict[str, list[np.ndarray]] = {k.name: [] for k in koz_objects}
                for r in ep_results:
                    for koz_name, trace in r["koz_traces"].items():
                        koz_traces[koz_name].append(trace)

                # Boresight trajectories
                boresight_trajs: dict[str, list[np.ndarray]] = {
                    bc["name"]: [] for bc in boresight_cfgs
                }
                q_inits_list = []
                q_trajs_list = []
                omega_trajs_list = []
                for r in ep_results:
                    q_inits_list.append(r["q_init"].copy())
                    q_trajs_list.append(r["q_traj"])
                    omega_trajs_list.append(r["omega_traj"])
                    for bs_name, bs_traj in r["boresight_trajs"].items():
                        boresight_trajs[bs_name].append(bs_traj)

                # Per-episode raw data for JSON
                episode_raw_data = []
                for r in ep_results:
                    tt = r["total_times"]
                    wt = r["wall_times"]
                    episode_raw_data.append({
                        "q_init": r["q_init"].tolist(),
                        "q_goal": r["q_goal"].tolist(),
                        "omega_init": r["omega_init"].tolist(),
                        "violated_steps": r["violated_steps"],
                        "total_steps": r["total_steps"],
                        "min_gap_deg": round(r["min_gap"], 2),
                        "goal_reached": r["reached"],
                        "att_err_deg": round(r["att_err_deg"], 4),
                        "omega_mag": round(r["omega_mag"], 6),
                        "control_effort": round(r["effort"], 4),
                        "mpc_failures": r["solve_failures"],
                        "qp_mean_ms": round(float(np.mean(wt)) * 1000, 2) if wt else None,
                        "mpc_mean_ms": round(float(np.mean(tt)) * 1000, 2) if tt else None,
                        "mpc_std_ms": round(float(np.std(tt)) * 1000, 2) if tt else None,
                        "mpc_max_ms": round(float(np.max(tt)) * 1000, 2) if tt else None,
                        "episode_wall_s": round(r["ep_wall"], 1),
                    })

                ep_viol_rate = float(np.mean(episode_violated))
                step_viol_rate = float(np.mean(step_violation_rates))
                n_violated = int(np.sum(episode_violated))
                ci_lo, ci_hi = _wilson_ci(n_violated, completed)
                goal_success_rate = float(np.mean(goal_reached_list))
                min_gap_overall = float(np.min(min_koz_gaps))
                mean_gap = float(np.mean(min_koz_gaps))
                total_mpc_fails = int(np.sum(mpc_fail_counts))

                results[combo_key] = {
                    "experiment": sc_name,
                    "desc": exp_desc,
                    "alpha": float(alpha) if alpha is not None else None,
                    "margin": margin_label if has_margin else None,
                    "margin_file": str(margin_file) if margin_file else None,
                    "completed_episodes": completed,
                    # Base initial conditions
                    "base_q_init": q_init_base.tolist(),
                    "base_q_goal": q_goal.tolist(),
                    # KOZ violation
                    "episode_violation_rate": round(ep_viol_rate, 4),
                    "episode_violation_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                    "step_violation_rate": round(step_viol_rate, 6),
                    "target_violation_rate": round(1 - alpha, 4) if alpha is not None else None,
                    "min_koz_gap_deg": round(min_gap_overall, 2),
                    "mean_min_koz_gap_deg": round(mean_gap, 2),
                    # Goal reaching
                    "goal_success_rate": round(goal_success_rate, 4),
                    "att_err_deg": _safe_stats(att_errors),
                    "att_err_rmse_deg": round(float(np.sqrt(np.mean(np.array(att_errors)**2))), 4),
                    "att_err_p95_deg": round(float(np.percentile(att_errors, 95)), 4),
                    "omega_mag": _safe_stats(omega_mags),
                    "omega_rmse": round(float(np.sqrt(np.mean(np.array(omega_mags)**2))), 6),
                    # Control performance
                    "control_effort": _safe_stats(control_efforts),
                    # Solver health
                    "mpc_solver_failures": total_mpc_fails,
                    "mpc_solve_time_ms": _safe_stats(
                        [t * 1000 for t in mpc_solve_times_all]
                    ),
                    "episode_wall_s": _safe_stats(episode_wall_times),
                    # Raw per-episode data
                    "episodes": episode_raw_data,
                }

                att_rmse = float(np.sqrt(np.mean(np.array(att_errors)**2)))
                att_p95 = float(np.percentile(att_errors, 95))
                target_pct = (1 - alpha) * 100 if alpha is not None else float("nan")
                logger.info(
                    "  -> %d/%d ep | ep_viol=%.1f%% [%.1f, %.1f] | step_v=%.4f%% | 1-a=%.1f%% | "
                    "gap_min=%.2f\u00b0 | goal=%.0f%% | err_rmse=%.2f\u00b0 p95=%.2f\u00b0 | impulse=%.2f | "
                    "mpc=%.1fms | fail=%d | %.0fs",
                    completed, num_episodes,
                    ep_viol_rate * 100, ci_lo * 100, ci_hi * 100,
                    step_viol_rate * 100, target_pct,
                    min_gap_overall, goal_success_rate * 100,
                    att_rmse, att_p95,
                    float(np.mean(control_efforts)),
                    float(np.mean(mpc_solve_times_mean)) * 1000 if mpc_solve_times_mean else 0.0,
                    total_mpc_fails,
                    combo_wall,
                )

                # --- MC KOZ separation overlay plot ---
                if koz_traces:
                    combo_fig_dir = os.path.join(output_dir, "figures")
                    plot_mc_koz_separation(
                        all_koz_traces=koz_traces,
                        koz_object=koz_objects,
                        log_dt=log_dt,
                        figure_dir=combo_fig_dir,
                        save_prefix=combo_key.replace("/", "_"),
                        title_suffix=f" [{exp_desc}]",
                    )

                # --- MC 3D boresight trajectory overlay plot ---
                if completed > 0 and boresight_trajs:
                    combo_fig_dir = os.path.join(output_dir, "figures")
                    plot_mc_boresight_3d(
                        boresight_trajs=boresight_trajs,
                        koz_objects=koz_objects,
                        q_goal=q_goal,
                        figure_dir=combo_fig_dir,
                        save_prefix=combo_key.replace("/", "_"),
                        title_suffix=f" [{exp_desc}]",
                    )

                # --- Save per-episode boresight data as .mat ---
                if completed > 0 and boresight_trajs:
                    data_dir = os.path.join(output_dir, "data")
                    os.makedirs(data_dir, exist_ok=True)
                    # Build short mat filename: {tag}_{alpha}_{margin}.mat
                    mat_parts = [exp_tag]
                    if alpha is not None:
                        mat_parts.append(f"{alpha:.2f}")
                    if has_margin:
                        mat_parts.append(margin_label)
                    mat_prefix = "_".join(mat_parts)

                    # Pad to max timestep length across episodes (NaN fill)
                    bs_names = list(boresight_trajs.keys())
                    max_T = max(
                        arr.shape[0]
                        for arrs in boresight_trajs.values()
                        for arr in arrs
                    )
                    time_vec = np.arange(max_T) * log_dt

                    mat_dict = {
                        "time_vec": time_vec,
                        "boresight_names": np.array(bs_names, dtype=object),
                        "boresight_vecs_B": boresight_B,
                        "koz_names": np.array([k.name for k in koz_objects], dtype=object),
                        "koz_directions": np.array([k.direction for k in koz_objects]),
                        "koz_angles_deg": np.array([k.angle_deg for k in koz_objects]),
                        "q_goal": q_goal,
                        "q_inits": np.array(q_inits_list),
                        "num_episodes": completed,
                    }
                    # Quaternion trajectories: (N, T, 4)
                    q_padded = np.full((completed, max_T, 4), np.nan)
                    for i, qt in enumerate(q_trajs_list):
                        q_padded[i, :qt.shape[0], :] = qt
                    mat_dict["q_traj"] = q_padded

                    # Angular velocity trajectories: (N, T, 3)
                    omega_padded = np.full((completed, max_T, 3), np.nan)
                    for i, wt in enumerate(omega_trajs_list):
                        omega_padded[i, :wt.shape[0], :] = wt
                    mat_dict["omega_traj"] = omega_padded

                    for bs_name in bs_names:
                        arrs = boresight_trajs[bs_name]
                        padded = np.full((completed, max_T, 3), np.nan)
                        for i, arr in enumerate(arrs):
                            padded[i, :arr.shape[0], :] = arr
                        mat_dict[bs_name] = padded

                    mat_path = os.path.join(data_dir, f"{mat_prefix}.mat")
                    scipy.io.savemat(mat_path, mat_dict)
                    logger.info("Boresight data saved to %s", mat_path)

    total_wall = time.perf_counter() - mc_start_time

    # ------------------------------------------------------------------
    #  Console summary table
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 165)
    logger.info(
        "%-24s  %5s  %12s  %10s  %18s  %8s  %8s  %7s  %9s  %9s  %8s  %7s  %5s",
        "Experiment", "Alpha", "Margin", "EpViol(%)", "CI95(%)",
        "StepV(%)", "1-a(%)", "Gap(\u00b0)", "RMSE(\u00b0)", "P95(\u00b0)",
        "Impulse", "MPC(ms)", "Fail",
    )
    logger.info("-" * 165)
    for key, r in results.items():
        alpha_str = f"{r['alpha']:.2f}" if r["alpha"] is not None else "  -  "
        margin_str = r["margin"] if r["margin"] is not None else "-"
        tgt_str = f"{r['target_violation_rate'] * 100:8.1f}" if r["target_violation_rate"] is not None else "     -  "
        if r["completed_episodes"] == 0:
            logger.info(
                "%-24s  %5s  %12s  %10s  %18s  %8s  %8s  %7s  %9s  %9s  %8s  %7s  %5s",
                r["experiment"], alpha_str, margin_str,
                "N/A", "N/A", "N/A", tgt_str,
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
            )
        else:
            ci = r["episode_violation_ci95"]
            effort_mean = r["control_effort"]["mean"] if r["control_effort"]["mean"] is not None else 0.0
            mpc_mean = r["mpc_solve_time_ms"]["mean"] if r["mpc_solve_time_ms"]["mean"] is not None else 0.0
            logger.info(
                "%-24s  %5s  %12s  %10.1f  [%6.1f, %6.1f]  %8.4f  %8s  %7.2f  %9.2f  %9.2f  %8.2f  %7.1f  %5d",
                r["experiment"], alpha_str, margin_str,
                r["episode_violation_rate"] * 100,
                ci[0] * 100, ci[1] * 100,
                r["step_violation_rate"] * 100,
                tgt_str,
                r["min_koz_gap_deg"],
                r["att_err_rmse_deg"],
                r["att_err_p95_deg"],
                effort_mean,
                mpc_mean,
                r["mpc_solver_failures"],
            )
    logger.info("=" * 165)
    logger.info("Total MC wall time: %.0fs  [%s, %d workers]", total_wall, controller_name, num_workers)

    # ------------------------------------------------------------------
    #  Aggregate step-level MPC solve times & save
    # ------------------------------------------------------------------
    if global_mpc_step_times:
        all_arr = np.array(global_mpc_step_times) * 1000  # s -> ms
        mpc_overall = {
            "num_steps": len(global_mpc_step_times),
            "mean_ms": round(float(all_arr.mean()), 4),
            "std_ms": round(float(all_arr.std()), 4),
            "p95_ms": round(float(np.percentile(all_arr, 95)), 4),
            "p99_ms": round(float(np.percentile(all_arr, 99)), 4),
            "wcet_ms": round(float(all_arr.max()), 4),
            "min_ms": round(float(all_arr.min()), 4),
        }
    else:
        mpc_overall = None

    # Profile breakdown (PartySMPC)
    mpc_profile_breakdown = None
    if global_profile_accum and global_profile_accum.get("qp"):
        warmup_skip = 1
        n_total = len(global_profile_accum["qp"])
        s = min(warmup_skip, n_total - 1)
        n = n_total - s
        mpc_keys = ["mpc_init", "QF_jac", "koz", "qp"]
        mpc_init_sub_keys = ["mpc_init/get_ctx", "mpc_init/get_AB",
                             "mpc_init/encode", "mpc_init/to_numpy"]

        breakdown = {}
        mpc_per_call = np.zeros(n)
        for key in mpc_keys:
            vals = np.array(global_profile_accum.get(key, [0.0]))[s:] * 1000
            breakdown[key] = {
                "mean_ms": round(float(np.mean(vals)), 4),
                "std_ms": round(float(np.std(vals)), 4),
                "wcet_ms": round(float(np.max(vals)), 4),
                "min_ms": round(float(np.min(vals)), 4),
            }
            mpc_per_call += vals
        for sub in mpc_init_sub_keys:
            sub_data = global_profile_accum.get(sub)
            if sub_data and len(sub_data) > s:
                sv = np.array(sub_data)[s:] * 1000
                breakdown[sub] = {
                    "mean_ms": round(float(np.mean(sv)), 4),
                    "std_ms": round(float(np.std(sv)), 4),
                    "wcet_ms": round(float(np.max(sv)), 4),
                    "min_ms": round(float(np.min(sv)), 4),
                }

        total_arr = np.array(global_mpc_step_times)[s:] * 1000
        other_arr = total_arr - mpc_per_call
        breakdown["other"] = {
            "mean_ms": round(float(np.mean(other_arr)), 4),
            "std_ms": round(float(np.std(other_arr)), 4),
            "wcet_ms": round(float(np.max(other_arr)), 4),
            "min_ms": round(float(np.min(other_arr)), 4),
        }
        breakdown["mpc_only"] = {
            "mean_ms": round(float(np.mean(mpc_per_call)), 4),
            "std_ms": round(float(np.std(mpc_per_call)), 4),
            "wcet_ms": round(float(np.max(mpc_per_call)), 4),
            "min_ms": round(float(np.min(mpc_per_call)), 4),
        }
        breakdown["total"] = {
            "mean_ms": round(float(np.mean(total_arr)), 4),
            "std_ms": round(float(np.std(total_arr)), 4),
            "wcet_ms": round(float(np.max(total_arr)), 4),
            "min_ms": round(float(np.min(total_arr)), 4),
        }
        breakdown["num_calls"] = n_total
        breakdown["warmup_skipped"] = s
        mpc_profile_breakdown = breakdown

        # ctx_step (incremental context encoding)
        if "ctx_step" in global_profile_accum and len(global_profile_accum["ctx_step"]) > 0:
            ctx_vals = np.array(global_profile_accum["ctx_step"]) * 1000
            breakdown["ctx_step"] = {
                "mean_ms": round(float(np.mean(ctx_vals)), 4),
                "std_ms": round(float(np.std(ctx_vals)), 4),
                "wcet_ms": round(float(np.max(ctx_vals)), 4),
                "min_ms": round(float(np.min(ctx_vals)), 4),
                "num_calls": len(ctx_vals),
            }

        # Print breakdown table
        logger.info("")
        logger.info(
            "_solve() profiling (%d calls, first %d skipped -> %d calls, ms):",
            n_total, s, n,
        )
        logger.info(
            "  %-15s  %7s  %7s  %7s  %7s", "", "mean", "std", "WCET", "min",
        )
        logger.info("  %s  %s  %s  %s  %s", "-" * 15, "-" * 7, "-" * 7, "-" * 7, "-" * 7)
        for key in mpc_keys:
            d = breakdown[key]
            logger.info(
                "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f",
                key, d["mean_ms"], d["std_ms"], d["wcet_ms"], d["min_ms"],
            )
            if key == "mpc_init":
                for sub in mpc_init_sub_keys:
                    if sub in breakdown:
                        sd = breakdown[sub]
                        label = "  " + sub.split("/")[-1]
                        logger.info(
                            "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f",
                            label, sd["mean_ms"], sd["std_ms"], sd["wcet_ms"], sd["min_ms"],
                        )
        if "ctx_step" in breakdown:
            d = breakdown["ctx_step"]
            logger.info(
                "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f  (%d calls)",
                "ctx_step", d["mean_ms"], d["std_ms"], d["wcet_ms"], d["min_ms"], d["num_calls"],
            )
        d = breakdown["other"]
        logger.info(
            "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f",
            "other", d["mean_ms"], d["std_ms"], d["wcet_ms"], d["min_ms"],
        )
        logger.info("  %s  %s  %s  %s  %s", "-" * 15, "-" * 7, "-" * 7, "-" * 7, "-" * 7)
        d = breakdown["mpc_only"]
        logger.info(
            "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f",
            "MPC only", d["mean_ms"], d["std_ms"], d["wcet_ms"], d["min_ms"],
        )
        d = breakdown["total"]
        logger.info(
            "  %-15s  %7.2f  %7.2f  %7.2f  %7.2f",
            "TOTAL", d["mean_ms"], d["std_ms"], d["wcet_ms"], d["min_ms"],
        )
        logger.info("  (first %d call(s) skipped for CUDA warmup)", s)

    # ------------------------------------------------------------------
    #  Solver status summary (IPOPT return_status breakdown)
    # ------------------------------------------------------------------
    solver_status_summary = None
    if global_solver_statuses:
        from collections import Counter
        status_counts = Counter(global_solver_statuses)
        total_calls = len(global_solver_statuses)
        solver_status_summary = {
            "total_calls": total_calls,
            "counts": dict(status_counts),
            "rates": {k: round(v / total_calls, 6) for k, v in status_counts.items()},
        }
        logger.info("")
        logger.info("Solver status breakdown (%d calls):", total_calls)
        for status, count in status_counts.most_common():
            logger.info(
                "  %-40s  %6d  (%5.1f%%)",
                status, count, count / total_calls * 100,
            )

    result_path = os.path.join(output_dir, "mc_validation_results.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "config": config_snapshot,
                "num_episodes_requested": num_episodes,
                "total_wall_time_s": round(total_wall, 1),
                "mpc_computation_time": mpc_overall,
                "mpc_profile_breakdown": mpc_profile_breakdown,
                "solver_status_summary": solver_status_summary,
                "results": results,
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    run()
