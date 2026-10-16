"""Re-generate 3D boresight trajectory plots from saved MC .mat files.

Usage::

    # Single .mat file
    python -m scripts.4_plot_mc_boresight_3d outputs/mc_lsmpc/2026-03-16/04-12-32/data/exp1a_0.95.mat

    # Multiple .mat files at once
    python -m scripts.4_plot_mc_boresight_3d outputs/mc_lsmpc/2026-03-16/04-12-32/data/*.mat

    # Custom output directory
    python -m scripts.4_plot_mc_boresight_3d outputs/.../exp1a_0.95.mat --outdir figures/

Reads the following fields from each .mat file (saved by 3_run_mc.py):
    - boresight_names:  (N_bs,) string array
    - <bs_name>:        (N_ep, T, 3) inertial-frame boresight trajectories
    - koz_names:        (N_koz,) string array
    - koz_directions:   (N_koz, 3)
    - koz_angles_deg:   (N_koz,)
    - q_goal:           (4,)
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio

from sim.koz.koz import KeepoutZone
from sim.utils.plot_utils import plot_mc_boresight_3d

# ── Per-experiment 3D view angles ──────────────────────────────────────
# Keys are matched against the .mat file stem (e.g. "exp1a_0.95").
# Use prefix matching: "exp1a" matches "exp1a_0.95", "exp1a_0.99", etc.
# Falls back to --elev/--azim CLI args, then to (20, 45) default.
VIEW_MAP: dict[str, tuple[float, float]] = {
    # prefix: (elev, azim)
    "exp1a": (60, -25),
    "exp1b": (20, 0),
    "exp2a": (20, 0),
    "exp2b": (20, 0),
}


def _get_view(mat_stem: str, cli_elev: float, cli_azim: float,
              ) -> tuple[float, float]:
    """Resolve view angles: VIEW_MAP (longest prefix) > CLI > default."""
    best_prefix = ""
    for prefix in VIEW_MAP:
        if mat_stem.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
    if best_prefix:
        return VIEW_MAP[best_prefix]
    return (cli_elev, cli_azim)


def _load_and_plot(mat_path: str, outdir: str | None,
                   elev: float = 20, azim: float = 45) -> None:
    mat_path = os.path.abspath(mat_path)
    if not os.path.isfile(mat_path):
        print(f"[SKIP] File not found: {mat_path}")
        return

    m = sio.loadmat(mat_path, squeeze_me=True)

    # --- Boresight names ---
    raw_names = np.atleast_1d(m["boresight_names"])
    bs_names = [str(n).strip() for n in raw_names]

    # --- Boresight trajectories: dict[name] -> list of (T, 3) arrays ---
    #     Keys in the .mat may differ from parsed names due to MATLAB string
    #     encoding.  Try the parsed name first, then fall back to a fuzzy match.
    boresight_trajs: dict[str, list[np.ndarray]] = {}
    mat_keys = set(m.keys())
    for name in bs_names:
        if name in mat_keys:
            key = name
        else:
            # Fuzzy: find a key that contains the name (case-insensitive)
            candidates = [k for k in mat_keys if name.lower() in k.lower()]
            if not candidates:
                print(f"[WARN] Boresight '{name}' not found in .mat keys: {sorted(mat_keys)}")
                continue
            key = candidates[0]
        arr = np.asarray(m[key], dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (1, T, 3)
        trajs = []
        for i in range(arr.shape[0]):
            traj_i = arr[i]  # (T, 3)
            # Remove NaN-padded tail
            valid = ~np.isnan(traj_i[:, 0])
            trajs.append(traj_i[valid])
        boresight_trajs[name] = trajs

    # --- KOZ objects ---
    koz_names_raw = np.atleast_1d(m["koz_names"]).flatten()
    koz_dirs = np.asarray(m["koz_directions"], dtype=float)
    koz_angles = np.asarray(m["koz_angles_deg"], dtype=float).flatten()

    koz_objects = []
    for i in range(len(koz_angles)):
        name = str(koz_names_raw[i]).strip()
        koz_objects.append(KeepoutZone(
            name=name,
            direction=koz_dirs[i].flatten(),
            angle_deg=float(koz_angles[i]),
        ))

    # --- q_goal ---
    q_goal = np.asarray(m["q_goal"], dtype=float).flatten()

    # --- Output ---
    mat_stem = os.path.splitext(os.path.basename(mat_path))[0]
    if outdir is None:
        # Default: figures/ directory alongside the data/ directory
        outdir = os.path.join(os.path.dirname(os.path.dirname(mat_path)), "figures")

    view_elev, view_azim = _get_view(mat_stem, elev, azim)

    plot_mc_boresight_3d(
        boresight_trajs=boresight_trajs,
        koz_objects=koz_objects,
        q_goal=q_goal,
        figure_dir=outdir,
        save_prefix=mat_stem,
        title_suffix=f" [{mat_stem}]",
        elev=view_elev,
        azim=view_azim,
    )
    print(f"[OK] {mat_stem}_mc_boresight_3d.png -> {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-generate 3D boresight plots from saved MC .mat files."
    )
    parser.add_argument(
        "mat_files", nargs="+", help="Path(s) to .mat file(s) from 3_run_mc.py"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for figures (default: figures/ next to data/)"
    )
    parser.add_argument(
        "--elev", type=float, default=20,
        help="Default elevation angle for 3D view (overridden by VIEW_MAP)"
    )
    parser.add_argument(
        "--azim", type=float, default=45,
        help="Default azimuth angle for 3D view (overridden by VIEW_MAP)"
    )
    args = parser.parse_args()

    for mat_path in args.mat_files:
        _load_and_plot(mat_path, args.outdir, args.elev, args.azim)


if __name__ == "__main__":
    main()
