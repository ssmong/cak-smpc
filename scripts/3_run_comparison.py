"""Comparison runner: same experiment conditions, multiple controllers.

Runs each experiment × controller combination as a single deterministic
simulation and saves results side-by-side for direct comparison.

**Sweep mode** (``sweep.num_trials > 1``): runs *num_trials* trials per
experiment, scoring each trial by the *combined* KOZ gap across all
baseline controllers (sum of min_gaps — lower = harder for everyone).
Picks the collectively worst-case trial and runs ALL controllers under
those identical conditions, including excluded ones (e.g. party/asgp).
Supports ``sweep.perturb_q_init_deg`` for random q_init perturbation
per trial, and multiprocessing (``sweep.num_workers``).

Output structure::

    outputs/comparison/{date}/{time}/
        {exp_tag}/
            {controller_key}/
                data/
                    sim_data.mat
                    boresight_data.mat
                    result.json

Usage::

    python -m scripts.3_run_comparison
    python -m scripts.3_run_comparison sweep.num_trials=100 sweep.pick_rank=5
    python -m scripts.3_run_comparison sweep.num_trials=100 sweep.num_workers=4
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
import scipy.io as sio
from Basilisk.architecture import bskLogging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from sim.koz.koz import KeepoutZone
from sim.koz.koz_utils import perturb_quaternion
from sim.montecarlo.single_episode import (
    check_koz_violations,
    compute_control_effort,
    goal_diagnostics,
    run_episode,
)
from sim.utils.math_utils import quat2boresight_batch

logger = logging.getLogger(__name__)

# Suppress verbose per-episode logging from sub-modules
_QUIET_LOGGERS = [
    "sim.koz.koz_utils", "sim.utils.bsk_setup", "sim.montecarlo.single_episode",
    "src.utils.party_utils", "src.model.party_interface", "src.dataset.processor",
    "PartySMPC", "NMPC", "ASGP_SMPC", "L_SMPC", "UKF_SMPC",
]


# ---------------------------------------------------------------------------
#  Config loading helpers (same pattern as 3_run_mc.py)
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


def _load_noise_config(cfg: DictConfig, noise_name: str) -> DictConfig:
    """Load an experiment-specific noise config and merge into *cfg*."""
    noise_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", "noise", f"{noise_name}.yaml"
    )
    noise_override = OmegaConf.load(noise_path)
    cfg_copy = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg_copy, False)
    cfg_copy.noise = OmegaConf.merge(cfg_copy.get("noise", {}), noise_override)
    OmegaConf.set_struct(cfg_copy, True)
    return cfg_copy


def _load_controller_config(
    cfg: DictConfig, ctrl_entry, sc_name: str | None = None,
) -> DictConfig:
    """Load a controller config and merge into *cfg*, with per-controller overrides."""
    ctrl_config_name = ctrl_entry.config if hasattr(ctrl_entry, "config") else ctrl_entry["config"]
    ctrl_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", "controller", f"{ctrl_config_name}.yaml"
    )
    ctrl_cfg = OmegaConf.load(ctrl_path)

    # Apply per-controller overrides from comparison.yaml
    overrides = None
    if hasattr(ctrl_entry, "overrides"):
        overrides = ctrl_entry.overrides
    elif isinstance(ctrl_entry, dict) and "overrides" in ctrl_entry:
        overrides = ctrl_entry["overrides"]

    if overrides is not None:
        override_container = OmegaConf.to_container(OmegaConf.create(overrides), resolve=False)
        ctrl_cfg = OmegaConf.merge(ctrl_cfg, OmegaConf.create(override_container))

    # Resolve exp_name="auto" → use experiment's sc name
    if sc_name and OmegaConf.select(ctrl_cfg, "exp_name") == "auto":
        OmegaConf.update(ctrl_cfg, "exp_name", sc_name)

    cfg_copy = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg_copy, False)
    cfg_copy.controller = ctrl_cfg

    ctrl_dt = float(ctrl_entry.ctrl_dt if hasattr(ctrl_entry, "ctrl_dt") else ctrl_entry["ctrl_dt"])
    OmegaConf.update(cfg_copy, "sim.ctrl_dt", ctrl_dt)
    OmegaConf.set_struct(cfg_copy, True)

    return cfg_copy


def _safe_stats(arr: list[float]) -> dict:
    """Compute summary stats from a list, returning NaN-safe dict."""
    if not arr:
        return {"mean": None, "std": None, "min": None, "max": None}
    a = np.array(arr)
    return {
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "min": round(float(np.min(a)), 4),
        "max": round(float(np.max(a)), 4),
    }


# ---------------------------------------------------------------------------
#  Timeseries diagnostics
# ---------------------------------------------------------------------------

def _compute_timeseries(
    state_traj: np.ndarray,
    q_goal: np.ndarray,
    boresight_B: np.ndarray,
    koz_objects: list[KeepoutZone],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-timestep diagnostic traces."""
    # Attitude error
    quats = state_traj[:, :4]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    q_goal_n = q_goal / np.linalg.norm(q_goal)
    dots = np.abs(np.sum(quats * q_goal_n[np.newaxis, :], axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    att_err_deg = np.degrees(2.0 * np.arccos(dots))

    # Angular velocity norm
    omega_norm = np.linalg.norm(state_traj[:, 4:], axis=1)

    # KOZ gap traces
    _, _, _, koz_sep_traces = check_koz_violations(
        state_traj, boresight_B, koz_objects, return_traces=True,
    )
    gap_cols = []
    for koz in koz_objects:
        gap_cols.append(koz_sep_traces[koz.name] - koz.angle_deg)
    koz_min_gap_deg = np.min(np.column_stack(gap_cols), axis=1)

    return att_err_deg, omega_norm, koz_min_gap_deg


# ---------------------------------------------------------------------------
#  Per-controller episode runner (returns data dict without saving)
# ---------------------------------------------------------------------------

def _run_ctrl_episode(
    exp_cfg: DictConfig,
    ctrl_entry,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    omega_init: np.ndarray,
    boresight_B: np.ndarray,
    koz_objects: list[KeepoutZone],
    log_dt: float,
    sc_name: str | None = None,
) -> dict | None:
    """Run one episode for a single controller and return diagnostics."""
    ctrl_key = ctrl_entry.key if hasattr(ctrl_entry, "key") else ctrl_entry["key"]

    try:
        run_cfg = _load_controller_config(exp_cfg, ctrl_entry, sc_name=sc_name)
    except FileNotFoundError as e:
        logger.error("  %s — config not found: %s", ctrl_key, e)
        return None

    t0 = time.perf_counter()
    try:
        state_traj, ctrl_traj, controller = run_episode(
            run_cfg, q_init=q_init, q_goal=q_goal, omega_init=omega_init,
        )
    except Exception as e:
        logger.error("  %s FAILED: %s", ctrl_key, e)
        return None
    wall_time = time.perf_counter() - t0

    T = state_traj.shape[0]
    time_vec = np.arange(T) * log_dt

    violated_steps, total_steps, min_gap = check_koz_violations(
        state_traj, boresight_B, koz_objects,
    )
    reached, att_err_deg, omega_mag = goal_diagnostics(state_traj, q_goal)
    effort = compute_control_effort(ctrl_traj, dt=log_dt)

    att_err_trace, omega_norm_trace, koz_min_gap_trace = _compute_timeseries(
        state_traj, q_goal, boresight_B, koz_objects,
    )

    tt = list(getattr(controller, "total_times", []))
    wt = list(getattr(controller, "wall_times", []))
    solve_failures = int(getattr(controller, "solve_failures", 0))

    out = {
        "ctrl_key": ctrl_key,
        "state_traj": state_traj,
        "ctrl_traj": ctrl_traj,
        "time_vec": time_vec,
        "wall_time": wall_time,
        "violated_steps": int(violated_steps),
        "total_steps": int(total_steps),
        "min_gap": float(min_gap),
        "reached": bool(reached),
        "att_err_deg": round(att_err_deg, 4),
        "omega_mag": omega_mag,
        "effort": round(effort, 4),
        "total_times": tt,
        "wall_times": wt,
        "solve_failures": solve_failures,
        "att_err_trace": att_err_trace,
        "omega_norm_trace": omega_norm_trace,
        "koz_min_gap_trace": koz_min_gap_trace,
    }

    # GP prediction diagnostics (ASGP controllers only)
    if hasattr(controller, "gp_diag_pred") and controller.gp_diag_pred:
        out["gp_diag_pred"] = np.array(controller.gp_diag_pred)
        out["gp_diag_actual"] = np.array(controller.gp_diag_actual)
        out["gp_diag_var"] = np.array(controller.gp_diag_var)
        out["gp_diag_M"] = np.array(controller.gp_diag_M)

    return out


def _run_ctrl_episode_from_resolved(
    run_cfg_dict: dict,
    ctrl_key: str,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    omega_init: np.ndarray,
    boresight_B: np.ndarray,
    koz_objects: list[KeepoutZone],
    log_dt: float,
    rng_seed: int | None = None,
) -> dict | None:
    """Like ``_run_ctrl_episode`` but takes a pre-resolved config dict."""
    run_cfg = OmegaConf.create(run_cfg_dict)

    t0 = time.perf_counter()
    try:
        state_traj, ctrl_traj, controller = run_episode(
            run_cfg, q_init=q_init, q_goal=q_goal, omega_init=omega_init,
            rng_seed=rng_seed,
        )
    except Exception as e:
        logger.error("  %s FAILED: %s", ctrl_key, e)
        return None
    wall_time = time.perf_counter() - t0

    T = state_traj.shape[0]
    time_vec = np.arange(T) * log_dt

    violated_steps, total_steps, min_gap = check_koz_violations(
        state_traj, boresight_B, koz_objects,
    )
    reached, att_err_deg, omega_mag = goal_diagnostics(state_traj, q_goal)
    effort = compute_control_effort(ctrl_traj, dt=log_dt)

    att_err_trace, omega_norm_trace, koz_min_gap_trace = _compute_timeseries(
        state_traj, q_goal, boresight_B, koz_objects,
    )

    tt = list(getattr(controller, "total_times", []))
    wt = list(getattr(controller, "wall_times", []))
    solve_failures = int(getattr(controller, "solve_failures", 0))

    out = {
        "ctrl_key": ctrl_key,
        "state_traj": state_traj,
        "ctrl_traj": ctrl_traj,
        "time_vec": time_vec,
        "wall_time": wall_time,
        "violated_steps": int(violated_steps),
        "total_steps": int(total_steps),
        "min_gap": float(min_gap),
        "reached": bool(reached),
        "att_err_deg": round(att_err_deg, 4),
        "omega_mag": omega_mag,
        "effort": round(effort, 4),
        "total_times": tt,
        "wall_times": wt,
        "solve_failures": solve_failures,
        "att_err_trace": att_err_trace,
        "omega_norm_trace": omega_norm_trace,
        "koz_min_gap_trace": koz_min_gap_trace,
    }

    # GP prediction diagnostics (ASGP controllers only)
    if hasattr(controller, "gp_diag_pred") and controller.gp_diag_pred:
        out["gp_diag_pred"] = np.array(controller.gp_diag_pred)
        out["gp_diag_actual"] = np.array(controller.gp_diag_actual)
        out["gp_diag_var"] = np.array(controller.gp_diag_var)
        out["gp_diag_M"] = np.array(controller.gp_diag_M)

    return out


# ---------------------------------------------------------------------------
#  Save helpers
# ---------------------------------------------------------------------------

def _save_episode(
    data_dir: str,
    result: dict,
    exp_tag: str,
    exp_desc: str,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    boresight_cfgs: list[dict],
    boresight_B: np.ndarray,
    koz_objects: list[KeepoutZone],
    log_dt: float,
    extra_result_fields: dict | None = None,
) -> None:
    """Save sim_data.mat, boresight_data.mat, and result.json for one episode."""
    state_traj = result["state_traj"]
    ctrl_traj = result["ctrl_traj"]
    time_vec = result["time_vec"]

    # ---- sim_data.mat ----
    sim_data = {
        "time": time_vec,
        "state": state_traj,
        "control": ctrl_traj,
        "att_err_deg": result["att_err_trace"],
        "omega_norm": result["omega_norm_trace"],
        "koz_min_gap_deg": result["koz_min_gap_trace"],
    }
    sio.savemat(os.path.join(data_dir, "sim_data.mat"), sim_data)

    # ---- boresight_data.mat ----
    bs_names = [bc["name"] for bc in boresight_cfgs]
    mat_dict = {
        "boresight_names": np.array(bs_names, dtype=object),
        "boresight_vecs_B": boresight_B,
        "koz_names": np.array([k.name for k in koz_objects], dtype=object),
        "koz_directions": np.array([k.direction for k in koz_objects]),
        "koz_angles_deg": np.array([k.angle_deg for k in koz_objects]),
        "q_goal": q_goal,
        "num_episodes": 1,
    }
    for bs_cfg in boresight_cfgs:
        bs_vec = np.array(bs_cfg["vec"], dtype=float)
        bs_N = quat2boresight_batch(state_traj[:, :4], bs_vec)
        mat_dict[bs_cfg["name"]] = bs_N[np.newaxis, :, :]
    sio.savemat(os.path.join(data_dir, "boresight_data.mat"), mat_dict)

    # ---- gp_diag.mat (ASGP controllers only) ----
    if "gp_diag_pred" in result:
        gp_diag = {
            "gp_pred": result["gp_diag_pred"],
            "gp_actual": result["gp_diag_actual"],
            "gp_var": result["gp_diag_var"],
            "gp_M": result["gp_diag_M"],
        }
        sio.savemat(os.path.join(data_dir, "gp_diag.mat"), gp_diag)

    # ---- result.json ----
    T = state_traj.shape[0]
    wt = result["wall_times"]
    tt = result["total_times"]
    result_data = {
        "experiment": exp_tag,
        "experiment_desc": exp_desc,
        "controller": result["ctrl_key"],
        "simulation_time": float(time_vec[-1]) if T > 0 else 0.0,
        "wall_time_s": round(result["wall_time"], 2),
        "initial_attitude": q_init.tolist(),
        "target_attitude": q_goal.tolist(),
        "final_attitude_error_deg": result["att_err_deg"],
        "goal_reached": result["reached"],
        "control_effort": result["effort"],
        "koz_violated_steps": result["violated_steps"],
        "koz_total_steps": result["total_steps"],
        "koz_min_gap_deg": round(result["min_gap"], 2),
        "mpc_solve_failures": result["solve_failures"],
        "mpc_wall_times": _safe_stats([t * 1000 for t in wt]),
        "mpc_total_times": _safe_stats([t * 1000 for t in tt]),
    }
    if extra_result_fields:
        result_data.update(extra_result_fields)
    with open(os.path.join(data_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


def _log_result(result: dict, run_idx: int, total_runs: int, exp_tag: str) -> None:
    """Log a one-line summary for a controller episode."""
    tt = result["total_times"]
    mpc_mean_ms = float(np.mean(tt)) * 1000 if tt else 0.0
    logger.info(
        "  [%d/%d] %s/%-12s -> %.1fs | err=%.2f\u00b0 | goal=%s | effort=%.1f | "
        "koz_viol=%d/%d (gap=%.1f\u00b0) | mpc=%.1fms | fail=%d",
        run_idx, total_runs, exp_tag, result["ctrl_key"],
        result["wall_time"], result["att_err_deg"],
        "Y" if result["reached"] else "N", result["effort"],
        result["violated_steps"], result["total_steps"], result["min_gap"],
        mpc_mean_ms, result["solve_failures"],
    )


def _result_to_summary_row(result: dict, exp_tag: str) -> dict:
    """Convert episode result to a summary table row."""
    tt = result["total_times"]
    mpc_mean_ms = float(np.mean(tt)) * 1000 if tt else 0.0
    return {
        "exp": exp_tag,
        "ctrl": result["ctrl_key"],
        "wall_s": round(result["wall_time"], 1),
        "att_err": round(result["att_err_deg"], 2),
        "goal": result["reached"],
        "effort": round(result["effort"], 1),
        "koz_viol": result["violated_steps"],
        "koz_gap": round(result["min_gap"], 1),
        "mpc_ms": round(mpc_mean_ms, 1),
        "fail": result["solve_failures"],
    }


# ---------------------------------------------------------------------------
#  Sweep worker (multiprocessing)
#  Each task = one (controller, trial) combination.
# ---------------------------------------------------------------------------
#  Sweep worker (multiprocessing)
#  Each task = one trial = run ALL baseline controllers under the same ICs.
# ---------------------------------------------------------------------------
_SW_CTRL_CFGS: dict[str, dict] | None = None   # ctrl_key -> resolved cfg dict
_SW_Q_GOAL: np.ndarray | None = None
_SW_OMEGA_INIT: np.ndarray | None = None
_SW_BS_B: np.ndarray | None = None
_SW_KOZ_DATA: list[dict] | None = None         # serialisable KOZ descriptors
_SW_EXCLUDE_KEYS: set | None = None            # controllers excluded from scoring


def _init_sweep_worker(
    ctrl_cfgs: dict[str, dict],
    q_goal: np.ndarray,
    omega_init: np.ndarray,
    boresight_B: np.ndarray,
    koz_data: list[dict],
    exclude_keys: set,
) -> None:
    """Initializer for pool workers — set up globals and suppress logs."""
    global _SW_CTRL_CFGS, _SW_Q_GOAL, _SW_OMEGA_INIT, _SW_BS_B, _SW_KOZ_DATA, _SW_EXCLUDE_KEYS
    _SW_CTRL_CFGS = ctrl_cfgs
    _SW_Q_GOAL = q_goal
    _SW_OMEGA_INIT = omega_init
    _SW_BS_B = boresight_B
    _SW_KOZ_DATA = koz_data
    _SW_EXCLUDE_KEYS = exclude_keys
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


def _score_trial(args: tuple) -> dict:
    """Worker: run ALL baseline controllers for ONE trial, return combined KOZ gap score.

    Score = sum of min_gap across all baseline controllers.
    Lower score = collectively harder trial for the baselines.
    """
    trial_idx, rng_seed, q_init_trial = args

    koz_objects = [
        KeepoutZone(d["direction"], d["angle_deg"], d["name"])
        for d in _SW_KOZ_DATA
    ]

    ctrl_gaps: dict[str, float] = {}
    for ctrl_key, cfg_dict in _SW_CTRL_CFGS.items():
        if ctrl_key in _SW_EXCLUDE_KEYS:
            continue
        run_cfg = OmegaConf.create(cfg_dict)
        try:
            state_traj, _, _ = run_episode(
                run_cfg, q_init=q_init_trial, q_goal=_SW_Q_GOAL,
                omega_init=_SW_OMEGA_INIT,
                rng_seed=rng_seed,
            )
            _, _, min_gap = check_koz_violations(state_traj, _SW_BS_B, koz_objects)
            ctrl_gaps[ctrl_key] = float(min_gap)
        except Exception as e:
            ctrl_gaps[ctrl_key] = np.inf

    score = sum(ctrl_gaps.values())  # lower = harder for all baselines

    return {
        "trial_idx": trial_idx,
        "rng_seed": rng_seed,
        "q_init": q_init_trial.tolist(),
        "score": score,
        "ctrl_gaps": ctrl_gaps,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="comparison", version_base=None)
def run(cfg: DictConfig) -> None:
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    experiments = list(cfg.experiments)
    controllers = list(cfg.controllers)
    log_dt = float(cfg.sim.log_dt)

    # Sweep config
    sweep_cfg = cfg.sweep
    num_trials = sweep_cfg.num_trials
    pick_rank = sweep_cfg.pick_rank
    exclude_keys = set(sweep_cfg.exclude_from_ranking)
    num_workers = sweep_cfg.num_workers
    perturb_deg = sweep_cfg.perturb_deg
    base_seed = cfg.seed

    max_safe = max(1, os.cpu_count())
    if num_workers > max_safe:
        logger.warning("num_workers=%d > cpu_count=%d. Capping.", num_workers, max_safe)
        num_workers = max_safe

    n_ctrl = len(controllers)
    total_runs = len(experiments) * n_ctrl
    sweep_mode = num_trials > 1
    parallel = sweep_mode and num_workers > 1

    if sweep_mode:
        logger.info(
            "Sweep mode: %d trials \u00d7 %d experiments \u00d7 %d controllers, "
            "pick_rank=%d, num_workers=%d",
            num_trials, len(experiments), n_ctrl, pick_rank, num_workers,
        )
    else:
        logger.info(
            "Comparison: %d experiments \u00d7 %d controllers = %d runs",
            len(experiments), n_ctrl, total_runs,
        )

    run_idx = 0
    all_start = time.perf_counter()
    summary_rows: list[dict] = []

    for exp in experiments:
        sc_name = exp.sc if hasattr(exp, "sc") else exp["sc"]
        constraint_name = exp.constraint if hasattr(exp, "constraint") else exp["constraint"]
        exp_desc = exp.desc if hasattr(exp, "desc") else exp["desc"]
        exp_tag = sc_name

        # Load experiment-specific sc + constraint
        exp_cfg = _load_sc_config(cfg, sc_name)
        if constraint_name:
            exp_cfg = _load_constraint_config(exp_cfg, constraint_name)

        # Extract initial conditions
        q_init_base = np.array(exp_cfg.sc.attitude.quat_init, dtype=float)
        q_goal = np.array(exp_cfg.sc.attitude.quat_goal, dtype=float)
        omega_init = np.array(exp_cfg.sc.attitude.omega_init, dtype=float)

        # Pre-generate per-trial q_inits (same for all controllers within a trial)
        # trial 0 is also used for single-trial mode and excluded controllers
        q_init_per_trial: dict[int, np.ndarray] = {}
        for trial_idx in range(max(num_trials, 1)):
            if perturb_deg > 0.0:
                np.random.seed(base_seed + trial_idx + 271828)
                q_init_per_trial[trial_idx] = perturb_quaternion(q_init_base, perturb_deg)
            else:
                q_init_per_trial[trial_idx] = q_init_base.copy()
        np.random.seed(base_seed)  # restore global rng

        q_init = q_init_per_trial[0]  # used in single-trial mode

        # Constraint info
        koz_objects = KeepoutZone.from_cfg(exp_cfg.constraint.koz_list)
        boresight_cfgs = [
            {"name": bs.name, "vec": list(bs.vec)}
            for bs in exp_cfg.constraint.boresight_list
        ]
        boresight_B = np.array([np.array(bs["vec"], dtype=float) for bs in boresight_cfgs])
        boresight_B = boresight_B / np.linalg.norm(boresight_B, axis=1, keepdims=True)

        logger.info("")
        logger.info("=" * 80)
        logger.info("Experiment: %s \u2014 %s", exp_tag, exp_desc)
        logger.info("  q_init_base=%s, q_goal=%s%s",
                    q_init_base.tolist(), q_goal.tolist(),
                    f"  [perturb ±{perturb_deg}°]" if perturb_deg > 0.0 else "")
        logger.info("=" * 80)

        if sweep_mode:
            # ==============================================================
            #  SWEEP MODE — collective worst-case selection
            #  Each trial runs ALL baseline controllers under identical ICs.
            #  Score = sum of min_gaps across baselines (lower = harder).
            #  The collectively hardest trial is shared by ALL controllers.
            # ==============================================================

            # Pre-resolve all controller configs on the main process
            ctrl_run_cfgs: dict[str, dict] = {}
            ctrl_keys_ordered: list[str] = []
            for ctrl_entry in controllers:
                ctrl_key = ctrl_entry.key if hasattr(ctrl_entry, "key") else ctrl_entry["key"]

                noise_prefix = (
                    ctrl_entry.noise_prefix
                    if hasattr(ctrl_entry, "noise_prefix")
                    else ctrl_entry.get("noise_prefix")
                    if isinstance(ctrl_entry, dict)
                    else None
                )
                if noise_prefix:
                    ctrl_exp_cfg = _load_noise_config(exp_cfg, f"{noise_prefix}_{sc_name}")
                else:
                    ctrl_exp_cfg = exp_cfg

                try:
                    run_cfg = _load_controller_config(ctrl_exp_cfg, ctrl_entry, sc_name=sc_name)
                except FileNotFoundError as e:
                    logger.error("  %s — config not found: %s", ctrl_key, e)
                    continue

                ctrl_run_cfgs[ctrl_key] = OmegaConf.to_container(run_cfg, resolve=True)
                ctrl_keys_ordered.append(ctrl_key)

            # Serialisable KOZ descriptors (picklable for spawn)
            koz_data = [
                {"name": k.name, "direction": k.direction.tolist(), "angle_deg": k.angle_deg}
                for k in koz_objects
            ]

            # Build task list: one task per trial (runs all baselines per task)
            # perturb mode: fix rng_seed, vary q_init only
            # seed mode:    fix q_init, vary rng_seed only
            def _trial_seed(idx):
                return base_seed if perturb_deg > 0.0 else base_seed + idx

            sweep_tasks: list[tuple[int, int, np.ndarray]] = [
                (trial_idx, _trial_seed(trial_idx), q_init_per_trial[trial_idx])
                for trial_idx in range(num_trials)
            ]

            # ----------------------------------------------------------
            #  Pass 1: Score all trials (all baselines per trial)
            # ----------------------------------------------------------
            n_baseline = len([k for k in ctrl_keys_ordered if k not in exclude_keys])
            logger.info(
                "  Pass 1: scoring %d trials × %d baselines = %d runs (%d workers)...",
                num_trials, n_baseline, num_trials * n_baseline, num_workers,
            )
            trial_results: list[dict] = []
            t_pass1 = time.perf_counter()

            worker_initargs = (
                ctrl_run_cfgs, q_goal, omega_init, boresight_B, koz_data, exclude_keys,
            )
            if parallel:
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    num_workers,
                    initializer=_init_sweep_worker,
                    initargs=worker_initargs,
                ) as pool:
                    pbar = tqdm(total=num_trials, desc=f"[{exp_tag}] sweep", unit="trial")
                    for res in pool.imap_unordered(_score_trial, sweep_tasks):
                        trial_results.append(res)
                        pbar.update(1)
                    pbar.close()
            else:
                _init_sweep_worker(*worker_initargs)
                pbar = tqdm(total=num_trials, desc=f"[{exp_tag}] sweep", unit="trial")
                for task in sweep_tasks:
                    trial_results.append(_score_trial(task))
                    pbar.update(1)
                pbar.close()

            pass1_wall = time.perf_counter() - t_pass1
            logger.info("  Pass 1 done in %.0fs.", pass1_wall)

            # ----------------------------------------------------------
            #  Rank trials by combined score, pick collectively worst
            # ----------------------------------------------------------
            trial_results.sort(key=lambda r: r["score"])  # ascending: worst first
            effective_rank = min(pick_rank, len(trial_results))
            sel = trial_results[effective_rank - 1]

            scores = [r["score"] for r in trial_results]
            logger.info(
                "  Picked trial %d (rank %d/%d): combined_gap=%.2f°  "
                "[min=%.2f, p5=%.2f, med=%.2f, p95=%.2f, max=%.2f]",
                sel["trial_idx"], effective_rank, len(trial_results), sel["score"],
                np.min(scores), np.percentile(scores, 5), np.median(scores),
                np.percentile(scores, 95), np.max(scores),
            )
            logger.info("  Per-baseline gaps for selected trial:")
            for ctrl_key, gap in sel["ctrl_gaps"].items():
                logger.info("    %-14s gap=%.2f°", ctrl_key, gap)

            q_init_run = np.array(sel["q_init"])
            rng_seed_run = sel["rng_seed"]
            sweep_extra = {
                "sweep_num_trials": num_trials,
                "sweep_pick_rank": pick_rank,
                "sweep_trial_idx": sel["trial_idx"],
                "sweep_rng_seed": rng_seed_run,
                "sweep_combined_gap": round(sel["score"], 4),
                "sweep_ctrl_gaps": {k: round(v, 4) for k, v in sel["ctrl_gaps"].items()},
            }
            if perturb_deg > 0.0:
                sweep_extra["sweep_perturb_deg"] = perturb_deg

            # ----------------------------------------------------------
            #  Pass 2: Re-run ALL controllers with the selected trial ICs
            # ----------------------------------------------------------
            logger.info("  Pass 2: re-running all controllers with selected ICs...")

            for ctrl_key in ctrl_keys_ordered:
                run_idx += 1
                cfg_dict = ctrl_run_cfgs[ctrl_key]

                result = _run_ctrl_episode_from_resolved(
                    cfg_dict, ctrl_key,
                    q_init_run, q_goal, omega_init,
                    boresight_B, koz_objects, log_dt,
                    rng_seed=rng_seed_run,
                )
                if result is None:
                    continue

                ctrl_out_dir = os.path.join(output_dir, exp_tag, ctrl_key)
                data_dir = os.path.join(ctrl_out_dir, "data")
                os.makedirs(data_dir, exist_ok=True)

                _save_episode(
                    data_dir, result, exp_tag, exp_desc,
                    q_init_run, q_goal,
                    boresight_cfgs, boresight_B, koz_objects, log_dt,
                    extra_result_fields=sweep_extra,
                )
                _log_result(result, run_idx, total_runs, exp_tag)
                summary_rows.append(_result_to_summary_row(result, exp_tag))

        else:
            # ==============================================================
            #  SINGLE TRIAL MODE (original deterministic comparison)
            # ==============================================================
            np.random.seed(base_seed)

            for ctrl_entry in controllers:
                ctrl_key = ctrl_entry.key if hasattr(ctrl_entry, "key") else ctrl_entry["key"]
                run_idx += 1

                noise_prefix = (
                    ctrl_entry.noise_prefix
                    if hasattr(ctrl_entry, "noise_prefix")
                    else ctrl_entry.get("noise_prefix")
                    if isinstance(ctrl_entry, dict)
                    else None
                )
                if noise_prefix:
                    ctrl_exp_cfg = _load_noise_config(exp_cfg, f"{noise_prefix}_{sc_name}")
                else:
                    ctrl_exp_cfg = exp_cfg

                result = _run_ctrl_episode(
                    ctrl_exp_cfg, ctrl_entry,
                    q_init, q_goal, omega_init,
                    boresight_B, koz_objects, log_dt,
                    sc_name=sc_name,
                )
                if result is None:
                    continue

                ctrl_out_dir = os.path.join(output_dir, exp_tag, ctrl_key)
                data_dir = os.path.join(ctrl_out_dir, "data")
                os.makedirs(data_dir, exist_ok=True)

                _save_episode(
                    data_dir, result, exp_tag, exp_desc,
                    q_init, q_goal,
                    boresight_cfgs, boresight_B, koz_objects, log_dt,
                )
                _log_result(result, run_idx, total_runs, exp_tag)
                summary_rows.append(_result_to_summary_row(result, exp_tag))

    total_wall = time.perf_counter() - all_start

    # ------------------------------------------------------------------
    #  Summary table
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 120)
    logger.info(
        "%-8s  %-12s  %7s  %8s  %5s  %7s  %9s  %7s  %8s  %5s",
        "Exp", "Controller", "Wall(s)", "Err(\u00b0)", "Goal", "Effort",
        "KOZ viol", "Gap(\u00b0)", "MPC(ms)", "Fail",
    )
    logger.info("-" * 120)
    for r in summary_rows:
        logger.info(
            "%-8s  %-12s  %7.1f  %8.2f  %5s  %7.1f  %9d  %7.1f  %8.1f  %5d",
            r["exp"], r["ctrl"], r["wall_s"], r["att_err"],
            "Y" if r["goal"] else "N", r["effort"],
            r["koz_viol"], r["koz_gap"], r["mpc_ms"], r["fail"],
        )
    logger.info("=" * 120)
    if sweep_mode:
        logger.info("Sweep: %d trials, pick_rank=%d, workers=%d", num_trials, pick_rank, num_workers)
    logger.info("Total wall time: %.0fs", total_wall)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    run()
