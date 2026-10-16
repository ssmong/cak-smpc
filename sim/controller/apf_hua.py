"""
Inverse-Proportional APF Controller.

Model-independent PD-like feedback controller combining:
- [29] Ma et al. (2020): ||q_ev||^2 positive-definite error with constant k1
- Hua et al. (AST 2024): dynamic weight assignment (eq.21), GNRON (eq.25),
  inverse-proportional barrier 1/(2 - q^T M̃ q)

Potential:
    V = ||q_ev||^2 * (β + Σ α_i(θ) / (2 - q^T M̃_i q))

Control law:
    u = (1/2) k_q Vec[∇V* ⊗ q] − k_ω ω

References:
    [1] B. Hua et al., AST, 2024. DOI: 10.1016/j.ast.2023.108738
    [2] G. Ma et al., J. Astronaut., 41(8), 2020.
"""

import time
import logging
import numpy as np
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from src.utils.math_numpy import quat_conj, quat_mul

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APFHua")


class APFHua(sysModel.SysModel):
    """Model-independent APF controller (Ma 2020 + Hua 2024 hybrid).

    Potential:
        V = ||q_ev||^2 * (β + Σ α_i(θ) / (2 - q^T M̃_i q))
    where ||q_ev||^2 = 1 - (q_d · q)^2 >= 0, β is constant attraction,
    and α_i are Hua's dynamic weights.

    Control law:
        u = (1/2) k_q Vec[∇V* ⊗ q] − k_ω ω
    """

    def __init__(
        self,
        k_q: float,
        k_omega: float,
        beta: float,
        eta: float,
        impact_margin_deg: float,
        delta_gnron: float,
        inertia: list,
        constraint_cfg: DictConfig,
        ctrl_dt: float,
        **kwargs,
    ):
        super().__init__()
        self.ModelTag = "APFHua"

        self.k_q = k_q
        self.k_omega = k_omega
        self.beta = beta
        self.eta = eta
        self.impact_margin_rad = np.radians(impact_margin_deg)
        self.delta_gnron = delta_gnron
        self.ctrl_dt = ctrl_dt

        inertia_arr = np.array(inertia, dtype=float)
        if inertia_arr.size != 9:
            raise ValueError(f"Inertia must be 9 elements, got {inertia_arr.size}")
        self.J = inertia_arr.reshape(3, 3)

        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        # Build (boresight, KOZ) constraint pairs
        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.pairs: list[dict] = []
        for bs in quad_koz.boresights:
            for koz in quad_koz.koz_objects:
                M = quad_koz._build_koz_matrix(
                    angle_rad=koz.angle_rad, x=koz.direction, y=bs)
                self.pairs.append({
                    "bs": bs,
                    "koz_dir": np.array(koz.direction, dtype=float),
                    "koz_angle": koz.angle_rad,
                    "impact_angle": koz.angle_rad + self.impact_margin_rad,
                    "M": M,
                })
        self.n_constraints = len(self.pairs)

        self.nx = 7
        self.nu = 3

        self._alpha_prev = np.zeros(self.n_constraints)

        self.total_times: list[float] = []
        self.wall_times: list[float] = []
        self.solve_failures: int = 0

        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        logger.info(
            f"[{self.ModelTag}] Initialized. k_q={k_q}, k_omega={k_omega}, "
            f"eta={eta}, impact_margin={impact_margin_deg}°, "
            f"n_koz={self.n_constraints}")

    def Reset(self, CurrentSimNanos: int):
        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] Resetting.")

        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")

        self._alpha_prev = np.zeros(self.n_constraints)
        self.total_times = []
        self.wall_times = []
        self.solve_failures = 0

        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        t_start = time.perf_counter()

        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega = np.array(nav_msg.omega_BN_B)
        q = rbk.MRP2EP(sigma_BN)

        ref_msg = self.attRefInMsg()
        sigma_RN = np.array(ref_msg.sigma_RN)
        q_d = rbk.MRP2EP(sigma_RN)

        if np.dot(q, q_d) < 0.0:
            q_d = -q_d

        # Error: ||q_ev||^2 = 1 - q_{e0}^2 where q_{e0} = q_d · q
        q_e0 = np.dot(q_d, q)
        att_err_sq = 1.0 - q_e0**2

        # Boresight angles and weight assignment (Hua eq.21)
        C_BN = rbk.EP2C(q)
        alphas = np.zeros(self.n_constraints)
        for i, p in enumerate(self.pairs):
            y_N = C_BN.T @ p["bs"]
            cos_theta = np.clip(np.dot(p["koz_dir"], y_N), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            if theta <= p["impact_angle"]:
                alphas[i] = self.eta / max(p["impact_angle"] - theta, 1e-4)

        # GNRON (Hua eq.25): zero out stagnant weights
        if self.delta_gnron > 0:
            d_alpha = np.abs(alphas - self._alpha_prev) / self.ctrl_dt
            stagnant = (alphas > 0) & (d_alpha < self.delta_gnron)
            alphas[stagnant] = 0.0
        self._alpha_prev = alphas.copy()

        # V = ||q_ev||^2 * (β + Σ α_i / (2 - q^T M̃q))
        # ∇V = -2 q_{e0} q_d * (β + Σα_i/margin)
        #     + ||q_ev||^2 * Σ 2α_i M̃q / margin^2
        barrier_sum = self.beta
        barrier_grad = np.zeros(4)
        for i, p in enumerate(self.pairs):
            if alphas[i] == 0.0:
                continue
            margin = max(2.0 - q @ p["M"] @ q, 1e-8)
            Mq = p["M"] @ q
            barrier_sum += alphas[i] / margin
            barrier_grad += alphas[i] * 2.0 * Mq / margin**2

        grad_V = -2.0 * q_e0 * q_d * barrier_sum + att_err_sq * barrier_grad

        # u = (1/2) k_q Vec[∇V* ⊗ q] − k_ω ω
        force_vec = quat_mul(quat_conj(grad_V), q)[1:4]
        torque = 0.5 * self.k_q * force_vec - self.k_omega * omega

        torque = np.clip(torque, self.u_min, self.u_max)

        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = torque.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

        t_end = time.perf_counter()
        self.wall_times.append(t_end - t_start)
        self.total_times.append(t_end - t_start)

    def get_wall_time_stats(self) -> dict:
        if not self.wall_times:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": np.mean(self.wall_times),
            "max": np.max(self.wall_times),
            "min": np.min(self.wall_times),
            "std": float(np.std(self.wall_times)),
            "count": len(self.wall_times),
        }

    def get_total_time_stats(self) -> dict:
        if not self.total_times:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": np.mean(self.total_times),
            "max": np.max(self.total_times),
            "min": np.min(self.total_times),
            "std": float(np.std(self.total_times)),
            "count": len(self.total_times),
        }
