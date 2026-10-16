"""Evaluate top ConvexPotential gain combos with step-level violation stats."""

import copy
import logging
import multiprocessing as mp
import os

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
from scripts.sweep_mc_apf import build_config

_W: dict = {}


def _init(cfg_dict: dict, bs_B: np.ndarray) -> None:
    _W["cfg"] = cfg_dict
    _W["bs"] = bs_B
    _W["koz"] = KeepoutZone.from_cfg(
        OmegaConf.create(cfg_dict["constraint"]["koz_list"])
    )
    import torch

    torch.set_num_threads(1)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
    for n in [
        "sim.koz.koz_utils",
        "sim.utils.bsk_setup",
        "sim.montecarlo.single_episode",
        "ConvexPotential",
    ]:
        logging.getLogger(n).setLevel(logging.WARNING)


def _run(args: tuple) -> dict:
    ci, ep, q_init, q_goal, omega, gains = args
    cfg_dict = copy.deepcopy(_W["cfg"])
    for k, v in gains.items():
        cfg_dict["controller"][k] = v
    cfg = OmegaConf.create(cfg_dict)
    try:
        st, ct, _ = run_episode(cfg, q_init=q_init, q_goal=q_goal, omega_init=omega)
    except Exception:
        return {"ci": ci, "error": True}
    vs, ts, mg, _ = check_koz_violations(
        st, _W["bs"], _W["koz"], return_traces=True, viol_margin_deg=0.01
    )
    reached, att_err, _ = goal_diagnostics(st, q_goal)
    return {
        "ci": ci,
        "vs": vs,
        "ts": ts,
        "mg": float(mg),
        "err": float(att_err),
        "reached": bool(reached),
    }


if __name__ == "__main__":
    np.random.seed(42)
    cfg_dict = build_config("potential")
    q0 = np.array(cfg_dict["sc"]["attitude"]["quat_init"])
    qg = np.array(cfg_dict["sc"]["attitude"]["quat_goal"])
    bs = np.array(
        [b["vec"] for b in cfg_dict["constraint"]["boresight_list"]], dtype=float
    )
    bs = bs / np.linalg.norm(bs, axis=1, keepdims=True)

    ics = [(perturb_quaternion(q0, 10.0), random_omega(0.0)) for _ in range(100)]

    candidates = [
        {"alpha_bs": 10.0, "beta_bs": 0.5, "k1": 0.01},
        {"alpha_bs": 5.0, "beta_bs": 1.0, "k1": 0.01},
        {"alpha_bs": 2.0, "beta_bs": 0.5, "k1": 0.01},
        {"alpha_bs": 5.0, "beta_bs": 0.5, "k1": 0.2},
        {"alpha_bs": 2.0, "beta_bs": 1.0, "k1": 0.01},
        {"alpha_bs": 2.0, "beta_bs": 0.5, "k1": 0.05},
        {"alpha_bs": 2.0, "beta_bs": 1.0, "k1": 0.05},
        {"alpha_bs": 5.0, "beta_bs": 0.5, "k1": 0.1},
        {"alpha_bs": 2.0, "beta_bs": 1.0, "k1": 0.1},
        {"alpha_bs": 2.0, "beta_bs": 2.0, "k1": 0.01},
    ]

    jobs = []
    for ci, g in enumerate(candidates):
        for ep, (qi, wi) in enumerate(ics):
            jobs.append((ci, ep, qi, qg, wi, g))

    print(f"Running {len(jobs)} episodes ({len(candidates)} combos × 100 ep)...")
    ctx = mp.get_context("spawn")
    results: dict[int, list] = {i: [] for i in range(len(candidates))}
    with ctx.Pool(20, initializer=_init, initargs=(cfg_dict, bs)) as pool:
        for r in pool.imap_unordered(_run, jobs, chunksize=4):
            if "error" not in r:
                results[r["ci"]].append(r)

    hdr = f"{'Gains':<35s} {'RMSE':>7s} {'EpViol':>7s} {'StepViol':>9s} {'Reach':>6s} {'MedGap':>7s} {'MinGap':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for ci, g in enumerate(candidates):
        eps = results[ci]
        if not eps:
            continue
        rmse = np.mean([e["err"] for e in eps])
        ev = np.mean([e["vs"] > 0 for e in eps])
        sv = np.mean([e["vs"] / e["ts"] for e in eps])
        reach = np.mean([e["reached"] for e in eps])
        gaps = [e["mg"] for e in eps]
        mgap = np.median(gaps)
        wgap = np.min(gaps)
        tag = ", ".join(f"{k}={v}" for k, v in g.items())
        print(f"{tag:<35s} {rmse:7.2f} {ev:7.0%} {sv:9.2%} {reach:6.0%} {mgap:7.1f} {wgap:7.1f}")
