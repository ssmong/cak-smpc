"""Standalone tuning script for 2-link robot-arm inertia profiles.

Edit the PARAMETERS section below, then run::

    python scripts/plot/plot_inertia_profile.py

Produces:
    1. Interactive matplotlib 3x2 figure (Jxx/Jxy, Jyy/Jyz, Jzz/Jzx).
    2. ``inertia_profile.mat`` for MATLAB plotting.
"""

import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    # Force a test figure to verify the backend actually works
    _fig_test = plt.figure(); plt.close(_fig_test)
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sim.utils.inertia_transition import InertiaTransition

# =========================================================================
#  PARAMETERS — Edit these and re-run
# =========================================================================

# Base spacecraft inertia (without captured mass) — flat 9-element row-major
BASE_INERTIA = [31.0, 0.0, 0.0,
                 0.0, 33.0, 0.0,
                 0.0,  0.0, 28.0]

# --- Link 1 (shoulder → elbow) ---
MASS_1          = 1.5       # [kg] elbow joint / actuator mass
ARM_LENGTH_1    = 0.9       # [m]
THETA_1_0_DEG   = 0.0       # initial angle [deg]
THETA_1_F_DEG   = 160.0     # final angle [deg]

# --- Link 2 (elbow → end-effector + payload) ---
MASS_2          = 3.0       # [kg] end-effector + captured object
ARM_LENGTH_2    = 0.6       # [m]
THETA_2_0_DEG   = 0.0       # initial relative angle [deg]
THETA_2_F_DEG   = -110.0    # final relative angle [deg] (fold-in)
DELAY_SEC       = 4.0       # joint 2 starts this many seconds after joint 1

# --- Common geometry ---
R_PIVOT         = [0.6, 0.5, 0.5]   # arm pivot in body frame [m]
ROTATION_AXIS   = [2.0, 1.0, 1.0]   # hinge plane normal (body frame)
ARM_DIR_INIT    = None               # initial arm direction (None = auto)

# --- Transition timing ---
TIME_START_SEC  = 10.0      # when arm motion begins [s]
DURATION_SEC    = 20.0      # total arm motion duration [s]

# --- Preview window ---
SIM_TIME        = 60.0      # total plot time [s]
DT              = 0.05      # sample interval [s]

# --- Output ---
_PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_DIR        = os.path.join(_PROJECT_ROOT, "outputs")
MAT_FILENAME    = "inertia_profile.mat"

# =========================================================================
#  END PARAMETERS
# =========================================================================


def main() -> None:
    cfg = {
        "enabled": True,
        "time_sec": TIME_START_SEC,
        "transition": {
            "type": "robot_arm",
            "duration_sec": DURATION_SEC,
            "params": {
                "mass_1": MASS_1,
                "arm_length_1": ARM_LENGTH_1,
                "theta_1_0_deg": THETA_1_0_DEG,
                "theta_1_f_deg": THETA_1_F_DEG,
                "mass_2": MASS_2,
                "arm_length_2": ARM_LENGTH_2,
                "theta_2_0_deg": THETA_2_0_DEG,
                "theta_2_f_deg": THETA_2_F_DEG,
                "delay_sec": DELAY_SEC,
                "pivot": R_PIVOT,
                "rotation_axis": ROTATION_AXIS,
                "arm_dir_init": ARM_DIR_INIT,
            },
        },
    }

    pre = np.array(BASE_INERTIA, dtype=float)
    it = InertiaTransition(pre, pre, cfg)

    # Evaluate profile
    t, J = it.evaluate_profile(SIM_TIME, DT)

    # Console summary
    J_pre = it.pre_inertia.reshape(3, 3)
    J_post = it.post_inertia.reshape(3, 3)
    print("\n=== 2-Link Robot Arm Inertia Profile ===")
    print(f"  Link 1: m={MASS_1} kg, L={ARM_LENGTH_1} m, "
          f"theta {THETA_1_0_DEG:.0f} -> {THETA_1_F_DEG:.0f} deg")
    print(f"  Link 2: m={MASS_2} kg, L={ARM_LENGTH_2} m, "
          f"theta {THETA_2_0_DEG:.0f} -> {THETA_2_F_DEG:.0f} deg, "
          f"delay={DELAY_SEC:.1f} s")
    print(f"  Pivot: {R_PIVOT}, Axis: {ROTATION_AXIS}")
    print(f"  Transition: {TIME_START_SEC} -- {TIME_START_SEC + DURATION_SEC} s")
    print(f"  Pre  diag: [{J_pre[0,0]:.2f}, {J_pre[1,1]:.2f}, {J_pre[2,2]:.2f}]")
    print(f"  Post diag: [{J_post[0,0]:.2f}, {J_post[1,1]:.2f}, {J_post[2,2]:.2f}]")
    print(f"  Post off-diag: xy={J_post[0,1]:.3f}, "
          f"yz={J_post[1,2]:.3f}, zx={J_post[2,0]:.3f}")
    print("========================================\n")

    # Save .mat
    mat_path = os.path.join(SAVE_DIR, MAT_FILENAME)
    sio.savemat(mat_path, {
        "time": t,
        "J_upper": J,
        "pre_inertia": J_pre,
        "post_inertia": J_post,
        "mass_1": MASS_1,
        "mass_2": MASS_2,
        "arm_length_1": ARM_LENGTH_1,
        "arm_length_2": ARM_LENGTH_2,
        "theta_1_0_deg": THETA_1_0_DEG,
        "theta_1_f_deg": THETA_1_F_DEG,
        "theta_2_0_deg": THETA_2_0_DEG,
        "theta_2_f_deg": THETA_2_F_DEG,
        "delay_sec": DELAY_SEC,
        "pivot": np.array(R_PIVOT),
        "rotation_axis": np.array(ROTATION_AXIS),
        "time_start_sec": TIME_START_SEC,
        "duration_sec": DURATION_SEC,
    })
    print(f"Saved: {mat_path}")

    # --- Plot 3x2 grid ---
    unit = r" [kg$\cdot$m$^2$]"
    labels = [r"$J_{xx}$" + unit, r"$J_{yy}$" + unit, r"$J_{zz}$" + unit,
              r"$J_{xy}$" + unit, r"$J_{yz}$" + unit, r"$J_{zx}$" + unit]
    col_map = [[0, 3], [1, 4], [2, 5]]   # row -> [left_idx, right_idx]

    fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True)

    for row in range(3):
        for col in range(2):
            idx = col_map[row][col]
            ax = axes[row, col]
            ax.plot(t, J[:, idx], color="#1f5fa6", linewidth=1.5)
            ax.set_ylabel(labels[idx])
            ax.grid(True, alpha=0.15)
            ax.set_xlim(0, 50)
            ax.tick_params(direction="out")
            if col == 1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    fig.tight_layout()
    png_path = os.path.join(SAVE_DIR, "inertia_profile.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
