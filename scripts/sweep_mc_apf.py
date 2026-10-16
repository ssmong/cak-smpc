"""MC-based gain sweep for APF controllers on exp1a.

Runs each gain configuration through N MC episodes and reports the best
configurations ranked by attitude RMSE and by KOZ violation rate.

Supports both ConvexPotential (Lee & Mesbahi 2014) and APFHua (Hua 2024).

Usage::

    cd /path/to/project
    source .venv/bin/activate
    python -m scripts.sweep_mc_apf potential --num_episodes 100
    python -m scripts.sweep_mc_apf apf_hua --num_episodes 100
"""

import argparse
import copy
import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from omegaconf import OmegaConf

from Basilisk.architecture import bskLogging

bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

from sim.koz.koz import KeepoutZone
from sim.koz.koz_utils import perturb_quaternion
from sim.montecarlo.mc_utils import random_omega
from sim.montecarlo.single_episode import (
    check_koz_violations,
    goal_diagnostics,
    run_episode,
)

CONFIGS_DIR = Path("configs")

GRIDS = {
    "potential": {
        "controller_yaml": "controller/potential.yaml",
        "params": {
            "alpha_bs": [2.0, 5.0, 10.0, 20.0],
            "beta_bs": [0.5, 1.0, 2.0, 5.0],
            "k1": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        },
    },
    "apf_hua": {
        "controller_yaml": "controller/apf_hua.yaml",
        "params": {
            "k_q": [1.0, 2.0, 5.0, 10.0],
            "k_omega": [2.0, 5.0, 10.0, 20.0],
            "beta": [1.0, 2.0, 5.0],
            "eta": [0.05, 0.1, 0.5, 1.0],
        },
    },
}

_QUIET_LOGGERS = [
    "sim.koz.koz_utils",
    "sim.utils.bsk_setup",
    "sim.montecarlo.single_episode",
    "ConvexPotential",
    "APFHua",
]

# ---------------------------------------------------------------------------
#  Worker process (multiprocessing)
# ---------------------------------------------------------------------------
_W_CFG_DICT: dict | None = None
_W_BS_B: np.ndarray | None = None
_W_VIOL_MARGIN: float = 0.01
_W_KOZ: list | None = None


def _init_worker(
    cfg_dict: dict,
    bs_B: np.ndarray,
    viol_margin: float,
) -> None:
    global _W_CFG_DICT, _W_BS_B, _W_VIOL_MARGIN, _W_KOZ
    _W_CFG_DICT = cfg_dict
    _W_BS_B = bs_B
    _W_VIOL_MARGIN = viol_margin
    koz_cfg = OmegaConf.create(cfg_dict["constraint"]["koz_list"])
    _W_KOZ = KeepoutZone.from_cfg(koz_cfg)
    import torch

    torch.set_num_threads(1)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _run_ep(args: tuple) -> dict:
    combo_idx, ep_idx, q_init, q_goal, omega_init, gain_overrides = args
    cfg_dict = copy.deepcopy(_W_CFG_DICT)
    for k, v in gain_overrides.items():
        cfg_dict["controller"][k] = v
    cfg = OmegaConf.create(cfg_dict)

    try:
        state_traj, ctrl_traj, _ = run_episode(
            cfg,
            q_init=q_init,
            q_goal=q_goal,
            omega_init=omega_init,
        )
    except Exception as e:
        return {"combo_idx": combo_idx, "ep_idx": ep_idx, "error": str(e)}

    violated_steps, total_steps, min_gap, _ = check_koz_violations(
        state_traj,
        _W_BS_B,
        _W_KOZ,
        return_traces=True,
        viol_margin_deg=_W_VIOL_MARGIN,
    )
    reached, att_err_deg, omega_mag = goal_diagnostics(state_traj, q_goal)

    return {
        "combo_idx": combo_idx,
        "ep_idx": ep_idx,
        "violated": violated_steps > 0,
        "min_gap_deg": float(min_gap),
        "att_err_deg": float(att_err_deg),
        "reached": bool(reached),
    }


# ---------------------------------------------------------------------------
#  Config builder
# ---------------------------------------------------------------------------
def build_config(controller_type: str) -> dict:
    """Build a resolved config dict for exp1a with the given controller."""
    sc_default = OmegaConf.load(CONFIGS_DIR / "sc" / "sc_default.yaml")
    sc_exp1a = OmegaConf.load(CONFIGS_DIR / "sc" / "exp1a.yaml")
    sc_exp1a_clean = OmegaConf.create(
        {k: v for k, v in sc_exp1a.items() if k != "defaults"}
    )
    sc = OmegaConf.merge(sc_default, sc_exp1a_clean)

    constraint = OmegaConf.load(CONFIGS_DIR / "constraint" / "exp1a.yaml")
    controller = OmegaConf.load(
        CONFIGS_DIR / GRIDS[controller_type]["controller_yaml"]
    )

    cfg = OmegaConf.create(
        {
            "sim": {
                "sim_time": 50.0,
                "sim_dt": 0.01,
                "ctrl_dt": 0.01,
                "log_dt": 0.01,
            },
            "env": {
                "gravity_order": 4,
                "enable_drag": True,
                "enable_srp": True,
                "enable_gg": True,
            },
        }
    )
    cfg.sc = sc
    cfg.controller = controller
    cfg.constraint = constraint

    return OmegaConf.to_container(cfg, resolve=True)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MC-based gain sweep for APF controllers on exp1a"
    )
    parser.add_argument("controller", choices=["potential", "apf_hua"])
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--perturb_deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg_dict = build_config(args.controller)
    grid = GRIDS[args.controller]
    param_names = list(grid["params"].keys())
    param_values = [grid["params"][k] for k in param_names]
    combos = list(itertools.product(*param_values))
    n_combos = len(combos)
    n_ep = args.num_episodes

    q_init_base = np.array(cfg_dict["sc"]["attitude"]["quat_init"], dtype=float)
    q_goal = np.array(cfg_dict["sc"]["attitude"]["quat_goal"], dtype=float)

    # Pre-generate ICs (shared across combos for fair comparison)
    ic_list: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_ep):
        q_init = perturb_quaternion(q_init_base, args.perturb_deg)
        omega_init = random_omega(0.0)
        ic_list.append((q_init, omega_init))

    bs_B = np.array(
        [bs["vec"] for bs in cfg_dict["constraint"]["boresight_list"]],
        dtype=float,
    )
    bs_B = bs_B / np.linalg.norm(bs_B, axis=1, keepdims=True)

    # Build all jobs: (combo_idx, ep_idx, q_init, q_goal, omega_init, gains)
    all_jobs = []
    combo_gains = []
    for ci, combo in enumerate(combos):
        gains = dict(zip(param_names, combo))
        combo_gains.append(gains)
        for ep, (q_init, omega_init) in enumerate(ic_list):
            all_jobs.append((ci, ep, q_init, q_goal, omega_init, gains))

    total_jobs = len(all_jobs)
    print(
        f"Sweep: {args.controller}, {n_combos} combos × {n_ep} episodes = {total_jobs} jobs"
    )
    print(f"Workers: {args.num_workers}, seed: {args.seed}")

    # Run all jobs through a single pool
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    combo_results: dict[int, list[dict]] = {ci: [] for ci in range(n_combos)}
    done = 0

    with ctx.Pool(
        args.num_workers,
        initializer=_init_worker,
        initargs=(cfg_dict, bs_B, 0.01),
        maxtasksperchild=200,
    ) as pool:
        for res in pool.imap_unordered(_run_ep, all_jobs, chunksize=2):
            done += 1
            ci = res["combo_idx"]
            if "error" not in res:
                combo_results[ci].append(res)
            if done % (n_ep * 5) == 0 or done == total_jobs:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (total_jobs - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{total_jobs}] {rate:.1f} ep/s, "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    wall = time.perf_counter() - t0
    print(f"\nDone in {wall / 60:.1f} min ({wall / total_jobs:.2f} s/ep)")

    # Aggregate per combo
    summary = []
    for ci in range(n_combos):
        gains = combo_gains[ci]
        eps = combo_results[ci]
        n = len(eps)
        if n == 0:
            summary.append({**gains, "completed": 0})
            continue

        rmses = [r["att_err_deg"] for r in eps]
        viols = [r["violated"] for r in eps]
        gaps = [r["min_gap_deg"] for r in eps]
        reaches = [r["reached"] for r in eps]

        summary.append(
            {
                **gains,
                "completed": n,
                "mean_rmse": round(float(np.mean(rmses)), 4),
                "std_rmse": round(float(np.std(rmses)), 4),
                "viol_rate": round(float(np.mean(viols)), 4),
                "reach_rate": round(float(np.mean(reaches)), 4),
                "median_gap": round(float(np.median(gaps)), 2),
                "worst_gap": round(float(np.min(gaps)), 2),
            }
        )

    # Report top 10 by RMSE
    valid = [s for s in summary if s.get("mean_rmse") is not None]

    print(f"\n{'='*70}")
    print("Best by RMSE (lower = better)")
    print(f"{'='*70}")
    by_rmse = sorted(valid, key=lambda r: r["mean_rmse"])
    for r in by_rmse[:10]:
        g = "  ".join(f"{k}={r[k]}" for k in param_names)
        print(
            f"  {g}  |  RMSE={r['mean_rmse']:.2f}±{r['std_rmse']:.2f}°  "
            f"viol={r['viol_rate']:.0%}  reach={r['reach_rate']:.0%}  "
            f"gap={r['median_gap']:.1f}°"
        )

    print(f"\n{'='*70}")
    print("Best by violation rate (lower = better, then RMSE)")
    print(f"{'='*70}")
    by_viol = sorted(valid, key=lambda r: (r["viol_rate"], r["mean_rmse"]))
    for r in by_viol[:10]:
        g = "  ".join(f"{k}={r[k]}" for k in param_names)
        print(
            f"  {g}  |  viol={r['viol_rate']:.0%}  "
            f"RMSE={r['mean_rmse']:.2f}°  reach={r['reach_rate']:.0%}  "
            f"gap={r['median_gap']:.1f}°"
        )

    print(f"\n{'='*70}")
    print("Best by reach rate (higher = better, then RMSE)")
    print(f"{'='*70}")
    by_reach = sorted(valid, key=lambda r: (-r["reach_rate"], r["mean_rmse"]))
    for r in by_reach[:10]:
        g = "  ".join(f"{k}={r[k]}" for k in param_names)
        print(
            f"  {g}  |  reach={r['reach_rate']:.0%}  "
            f"RMSE={r['mean_rmse']:.2f}°  viol={r['viol_rate']:.0%}  "
            f"gap={r['median_gap']:.1f}°"
        )

    # Save
    out_path = Path(f"outputs/sweep_mc_{args.controller}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
