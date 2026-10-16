"""
Convex Logarithmic Barrier Potential Controller (Lee & Mesbahi, 2014 TAES).

Model-dependent (modified backstepping) feedback control law for spacecraft
reorientation under attitude keep-out zone constraints. Uses a strictly convex
logarithmic barrier potential with nominal inertia.

Control law (eq.79):
    u = J*omega_c_dot + omega x (J*omega) + 0.5*C2*Vec[grad_V' * q] - z

where -R(omega)*J*omega = +omega x (J*omega) because R(omega) = -[omega]x (eq.19),
z is a saturated error (eq.78), and omega_c is the desired angular velocity
from the potential gradient (eq.67).

Reference:
    U. Lee and M. Mesbahi, "Feedback Control for Spacecraft Reorientation
    Under Attitude Constraints Via Convex Potentials," IEEE TAES, 2014.
    DOI: 10.1109/TAES.2014.120240
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
logger = logging.getLogger("ConvexPotential")


class ConvexPotential(sysModel.SysModel):
    """
    Convex logarithmic barrier potential controller (Lee & Mesbahi, 2014).

    Model-dependent (modified backstepping) variant using nominal inertia.

    Potential function (eq.39, multiplicative form):
        V(q) = ||q_d - q||^2 * lg(q)
        lg(q) = -k1 * sum_i log((2 - q^T M_tilde_i q) / 2)

    Control law (eq.79):
        u = J*omega_c_dot + omega x (J*omega) + 0.5*C2*Vec[grad_V' * q] - z
    """

    def __init__(
        self,
        alpha_bs: float,
        beta_bs: float,
        k1: float,
        inertia: list,
        constraint_cfg: DictConfig,
        ctrl_dt: float,
        **kwargs,
    ):
        super().__init__()
        self.ModelTag = "ConvexPotential"

        self.alpha_bs = alpha_bs
        self.beta_bs = beta_bs
        self.k1 = k1
        self.ctrl_dt = ctrl_dt

        # Nominal inertia
        inertia_arr = np.array(inertia, dtype=float)
        if inertia_arr.size != 9:
            raise ValueError(f"Inertia must be 9 elements, got {inertia_arr.size}")
        self.J = inertia_arr.reshape(3, 3)

        # Constraints
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        # KOZ matrices (M_tilde: q^T M_tilde q <= 2 is safe)
        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.num_bs = quad_koz.num_bs
        self.num_koz = quad_koz.num_koz
        self.koz_matrices = []
        for bs in quad_koz.boresights:
            for koz in quad_koz.koz_objects:
                M = quad_koz._build_koz_matrix(
                    angle_rad=koz.angle_rad, x=koz.direction, y=bs)
                self.koz_matrices.append(M)
        self.n_constraints = len(self.koz_matrices)

        # State/control dimensions
        self.nx = 7
        self.nu = 3

        # Timing
        self.total_times: list[float] = []
        self.wall_times: list[float] = []
        self.solve_failures: int = 0

        # BSK messages
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        logger.info(
            f"[{self.ModelTag}] Initialized. alpha_bs={alpha_bs}, beta_bs={beta_bs}, "
            f"k1={k1}, n_koz={self.n_constraints}")

    def Reset(self, CurrentSimNanos: int):
        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] Resetting: alpha_bs={self.alpha_bs}, "
            f"beta_bs={self.beta_bs}, k1={self.k1}")

        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")

        self.total_times = []
        self.wall_times = []
        self.solve_failures = 0

        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        t_start = time.perf_counter()

        # Read navigation state
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega = np.array(nav_msg.omega_BN_B)
        q = rbk.MRP2EP(sigma_BN)

        # Read reference attitude
        ref_msg = self.attRefInMsg()
        sigma_RN = np.array(ref_msg.sigma_RN)
        q_d = rbk.MRP2EP(sigma_RN)

        # Unwinding avoidance
        if np.dot(q, q_d) < 0.0:
            q_d = -q_d

        # Compute potential gradient (R^4)
        grad_V = self._compute_gradient(q, q_d)

        # Desired angular velocity: omega_c = Vec[-2 * conj(q) ⊗ grad_V] (eq.67)
        omega_c_quat = quat_mul(-2.0 * quat_conj(q), grad_V)
        omega_c = omega_c_quat[1:4]

        # Analytical omega_c_dot (eq.80):
        #   ω̇_c = −2 Vec[q̇* ⊗ ∇V + q* ⊗ (∇²V @ q̇)]
        hess_V = self._compute_hessian(q, q_d)
        omega_pure = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_mul(q, omega_pure)
        omega_c_dot = -2.0 * (quat_mul(quat_conj(q_dot), grad_V)
                              + quat_mul(quat_conj(q), hess_V @ q_dot))[1:4]

        # Modified error z (eq.78): z = alpha * arctan(beta * (omega - omega_c))
        omega_err = omega - omega_c
        z = self.alpha_bs * np.arctan(self.beta_bs * omega_err)

        # C2 = (1/(alpha*beta)) * [I + beta^2 * diag((omega - omega_c)^2)]  (from eq.79)
        C2 = (1.0 / (self.alpha_bs * self.beta_bs)) * (
            np.eye(3) + self.beta_bs**2 * np.diag(omega_err**2))

        # Vec[grad_V' ⊗ q]: gradient force term
        grad_V_conj = quat_conj(grad_V)
        force_quat = quat_mul(grad_V_conj, q)
        force_vec = force_quat[1:4]

        # Control law (eq.79): u = J*omega_c_dot + omega x (J*omega) + 0.5*C2*force_vec - z
        torque = (self.J @ omega_c_dot
                  + np.cross(omega, self.J @ omega)
                  + 0.5 * C2 @ force_vec
                  - z)

        # Clamp torque
        torque = np.clip(torque, self.u_min, self.u_max)
        self._u_curr = torque.copy()

        # Write output
        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = torque.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

        t_end = time.perf_counter()
        self.wall_times.append(t_end - t_start)
        self.total_times.append(t_end - t_start)

    def _compute_hessian(self, q: np.ndarray, q_d: np.ndarray) -> np.ndarray:
        """Compute 4x4 Hessian of V(q) = ||q_d - q||^2 * lg(q)."""
        att_sq = 2.0 - 2.0 * np.dot(q_d, q)
        grad_att_sq = 2.0 * (q - q_d)

        lg = 0.0
        grad_lg = np.zeros(4)
        hess_lg = np.zeros((4, 4))
        for M in self.koz_matrices:
            qMq = q @ M @ q
            margin = 2.0 - qMq
            if margin < 1e-8:
                margin = 1e-8
            Mq = M @ q
            lg += -self.k1 * np.log(margin / 2.0)
            grad_lg += self.k1 * 2.0 * Mq / margin
            hess_lg += self.k1 * (
                2.0 * M / margin + 4.0 * np.outer(Mq, Mq) / margin**2)

        return (2.0 * np.eye(4) * lg
                + np.outer(grad_att_sq, grad_lg)
                + np.outer(grad_lg, grad_att_sq)
                + att_sq * hess_lg)

    def _compute_gradient(self, q: np.ndarray, q_d: np.ndarray) -> np.ndarray:
        """Compute R^4 gradient of V(q) = ||q_d - q||^2 * lg(q).

        Gradient via product rule:
            grad_V = 2(q - q_d) * lg + (2 - 2*dot(q_d,q)) * grad_lg
        """
        lg = 0.0
        grad_lg = np.zeros(4)
        for M in self.koz_matrices:
            qMq = q @ M @ q
            margin = 2.0 - qMq
            if margin < 1e-8:
                margin = 1e-8
            lg += -self.k1 * np.log(margin / 2.0)
            grad_lg += self.k1 * 2.0 * (M @ q) / margin

        att_sq = 2.0 - 2.0 * np.dot(q_d, q)
        grad_att_sq = 2.0 * (q - q_d)
        return grad_att_sq * lg + att_sq * grad_lg

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
