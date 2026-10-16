"""
RAMP-Net MPC Controller for spacecraft attitude control.

Uses a trained PINN (RampNet) as the dynamics model inside a CasADi MPC.
The PyTorch model weights are extracted and reconstructed as CasADi symbolic
expressions, enabling exact automatic differentiation for Jacobian/Hessian.
Follows the same Basilisk SysModel pattern as sim/controller/nmpc.py.

Reference:
    Sanyal & Roy, "RAMP-Net: A Robust Adaptive MPC for Quadrotors via
    Physics-informed Neural Network", ICRA 2023.
"""

import os
import time
import logging
from collections import deque

import numpy as np
import casadi as ca
import torch
from omegaconf import DictConfig, OmegaConf

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from sim.utils.math_utils import quat_error_ca
from src.model.ramp_net import RampNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RampNet")


def _build_casadi_pinn(model: RampNet, dt: float) -> ca.Function:
    """Reconstruct the trained PINN as a CasADi symbolic function.

    Extracts weight matrices and bias vectors from the PyTorch model and
    builds equivalent CasADi MX expressions. This allows CasADi to compute
    exact Jacobians and Hessians via symbolic automatic differentiation,
    following the approach in Nicodemus et al. (2021) referenced by RAMP-Net.

    Args:
        model: Trained RampNet PyTorch model.
        dt: MPC time step (fixed time input to the PINN).

    Returns:
        CasADi Function: (x[7], u[3]) -> x_next[7]
    """
    nx = model.state_dim
    nu = model.control_dim

    # CasADi symbolic inputs
    x_sym = ca.MX.sym("x", nx)
    u_sym = ca.MX.sym("u", nu)

    # PINN input: [x(7), u(3), t(1)] with t fixed to dt
    t_val = ca.MX(dt)
    z = ca.vertcat(x_sym, u_sym, t_val)  # (11, 1)

    # Extract layers from nn.Sequential and reconstruct symbolically
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().cpu().numpy()  # (out, in)
            b = layer.bias.detach().cpu().numpy()     # (out,)
            z = ca.mtimes(ca.DM(W), z) + ca.DM(b.reshape(-1, 1))
        elif type(layer).__name__ == "Tanh":
            z = ca.tanh(z)
        elif type(layer).__name__ == "ReLU":
            z = ca.fmax(0, z)
        elif type(layer).__name__ == "SiLU":
            z = z * (1.0 / (1.0 + ca.exp(-z)))
        else:
            raise ValueError(f"Unsupported layer type: {type(layer).__name__}")

    # z is now (nx, 1) = PINN output
    pinn_fn = ca.Function("pinn", [x_sym, u_sym], [z], ["x", "u"], ["x_next"])
    return pinn_fn



class RampNet(sysModel.SysModel):
    """
    CasADi-based MPC for spacecraft attitude control using RAMP-Net PINN dynamics.

    Follows the same interface as sim/controller/nmpc.py:
    - Quaternion-based attitude dynamics (replaced by PINN)
    - Keep-Out Zone (KOZ) constraints as soft constraints
    - Torque bounds
    """

    def __init__(
        self,
        model_dir: str,
        horizon: int,
        mpc_dt: float,
        inertia: list,
        Q_diag: list,
        R_diag: list,
        F_diag: list,
        rho: float,
        max_iter: int,
        constraint_cfg: DictConfig,
        du_weight: float = 0.0,
        **kwargs,
    ):
        super(RampNet, self).__init__()
        self.ModelTag = "RampNet"

        if mpc_dt <= 0:
            raise ValueError(f"mpc_dt must be positive: {mpc_dt}")
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2: {horizon}")

        self.horizon = horizon
        self.mpc_dt = mpc_dt
        self.max_iter = max_iter
        self.rho = rho
        self.du_weight = du_weight

        # Inertia (for reference / KOZ, not used in PINN dynamics)
        inertia_arr = np.array(inertia)
        if inertia_arr.size != 9:
            raise ValueError(f"Inertia must be 9 elements (3x3 row-major), got {inertia_arr.size}")
        self.J = inertia_arr.reshape(3, 3)

        # Cost matrices (6D error: quat_err[3] + omega_err[3])
        self.Q = ca.diag(Q_diag)
        self.R = ca.diag(R_diag)
        self.F = ca.diag(F_diag)

        # Constraints
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.num_bs = quad_koz.num_bs
        self.num_koz = quad_koz.num_koz
        self.koz_matrices = np.zeros((self.num_bs, self.num_koz, 4, 4))
        for l, bs in enumerate(quad_koz.boresights):
            for m, koz in enumerate(quad_koz.koz_objects):
                self.koz_matrices[l, m] = quad_koz._build_koz_matrix(
                    angle_rad=koz.angle_rad, x=koz.direction, y=bs,
                )

        # State/control dimensions
        self.nx = 7
        self.nu = 3

        # Load trained PINN model and build CasADi symbolic function
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_dir)

        # Solver (initialized in Reset)
        self.opti = None
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)

        # History buffer for 3_run_sim.py live progress display
        x_init = kwargs.get("x_init", np.zeros(self.nx))
        self._hist_x: deque[np.ndarray] = deque(maxlen=2)
        self._hist_x.append(np.array(x_init, dtype=float))

        # Timing & failure tracking
        self.total_times: list[float] = []
        self.wall_times: list[float] = []
        self.solve_failures: int = 0

        # Basilisk messages
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        logger.info(f"[RampNet] Initialized. horizon={horizon}, mpc_dt={mpc_dt}s")

    def _load_model(self, model_dir: str):
        """Load trained RAMP-Net model and reconstruct as CasADi symbolic function."""
        # Use Hydra's original cwd to resolve the model path, since Hydra changes
        # the working directory to hydra.run.dir at runtime.
        try:
            import hydra
            orig_cwd = hydra.utils.get_original_cwd()
        except (ImportError, ValueError):
            orig_cwd = ""
        base_path = os.path.join(orig_cwd, "outputs/learning_ramp", model_dir)

        config_path = os.path.join(base_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        model_cfg = OmegaConf.load(config_path)

        device = torch.device("cpu")
        pinn = RampNet(
            state_dim=model_cfg.model.state_dim,
            control_dim=model_cfg.model.control_dim,
            hidden_dim=model_cfg.model.hidden_dim,
            num_layers=model_cfg.model.num_layers,
            activation=model_cfg.model.activation,
        ).to(device)

        ckpt_path = os.path.join(base_path, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        pinn.load_state_dict(checkpoint["model_state_dict"])
        pinn.eval()
        logger.info(f"[RampNet] Loaded PINN from {ckpt_path} (epoch {checkpoint['epoch']+1})")

        # Build CasADi symbolic function from PyTorch weights
        self.pinn_fn = _build_casadi_pinn(pinn, self.mpc_dt)
        logger.info("[RampNet] Built CasADi symbolic PINN (exact AD for Jacobian/Hessian)")

    def Reset(self, CurrentSimNanos: int):
        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] Resetting: mpc_dt={self.mpc_dt}s, horizon={self.horizon}",
        )

        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")

        self._setup_ocp()

        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self._hist_x.clear()
        self.total_times = []
        self.wall_times = []
        self.solve_failures = 0

        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        # Read navigation
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega_BN_B = np.array(nav_msg.omega_BN_B)

        q_curr = rbk.MRP2EP(sigma_BN)
        x_curr = np.concatenate([q_curr, omega_BN_B])
        self._hist_x.append(x_curr.copy())

        # Read reference
        ref_msg = self.attRefInMsg()
        sigma_RN = np.array(ref_msg.sigma_RN)
        q_goal = rbk.MRP2EP(sigma_RN)
        omega_goal = np.zeros(3)

        state_goal = np.concatenate([q_goal, omega_goal])
        x_goal = np.tile(state_goal.reshape(-1, 1), (1, self.horizon + 1))

        # Solve MPC
        torque = self._solve(x_curr, x_goal)

        # Write output
        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = torque.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

    def _setup_ocp(self) -> None:
        """Set up CasADi OCP with PINN dynamics as symbolic expressions."""
       

        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)

        # KOZ slack variables
        self._use_koz = (self.num_koz > 0) and (self.num_bs > 0)
        if self._use_koz:
            self.slack = self.opti.variable(self.num_bs * self.num_koz, self.horizon)
        else:
            self.slack = None
            self.bskLogger.bskLog(
                bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] KOZ constraints disabled.",
            )

        # Parameters
        self.x_curr = self.opti.parameter(self.nx)
        self.x_goal = self.opti.parameter(self.nx, self.horizon + 1)
        if self.du_weight > 0:
            self.u_prev_p = self.opti.parameter(self.nu)

        # Cost
        cost = 0.0
        self.opti.subject_to(self.X[:, 0] == self.x_curr)

        for k in range(self.horizon):
            q_err = quat_error_ca(self.X[:4, k], self.x_goal[:4, k])
            w_err = self.X[4:, k] - self.x_goal[4:, k]
            x_err = ca.vertcat(q_err, w_err)
            cost += ca.mtimes([x_err.T, self.Q, x_err])
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])

            # Control rate penalty
            if self.du_weight > 0:
                du_k = (self.U[:, k] - self.u_prev_p if k == 0
                        else self.U[:, k] - self.U[:, k - 1])
                cost += self.du_weight * ca.dot(du_k, du_k)

            # PINN dynamics constraint: x[k+1] = PINN(x[k], u[k])
            x_next_pinn = self.pinn_fn(self.X[:, k], self.U[:, k])  # (7, 1)


            # Quaternion normalization
            q_next = x_next_pinn[:4]
            w_next = x_next_pinn[4:]
            q_norm = ca.fmax(ca.norm_2(q_next), 1e-8)
            x_next = ca.vertcat(q_next / q_norm, w_next)

            self.opti.subject_to(self.X[:, k + 1] == x_next)
            self.opti.subject_to(
                self.opti.bounded(self.u_min, self.U[:, k], self.u_max)
            )

        # Terminal cost
        q_err_f = quat_error_ca(self.X[:4, self.horizon], self.x_goal[:4, self.horizon])
        w_err_f = self.X[4:, self.horizon] - self.x_goal[4:, self.horizon]
        x_err_f = ca.vertcat(q_err_f, w_err_f)
        cost += ca.mtimes([x_err_f.T, self.F, x_err_f])

        # KOZ soft constraints (same as NMPC)
        if self._use_koz:
            cost += self.rho * ca.sumsqr(self.slack)
            self.opti.subject_to(ca.vec(self.slack) >= 0.0)
            for k in range(self.horizon):
                q_curr = self.X[:4, k + 1]
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm_idx = l * self.num_koz + m
                        M_tilde = self.koz_matrices[l, m]
                        quad_val = ca.mtimes([q_curr.T, ca.DM(M_tilde), q_curr])
                        self.opti.subject_to(quad_val <= 2.0 + self.slack[lm_idx, k])

        self.opti.minimize(cost)

        # Solver options (same as NMPC — expand: True is safe with symbolic PINN)
        p_opts = {
            "expand": True,
            "print_time": False,
            "record_time": True,
            "verbose": False,
        }
        s_opts = {
            "print_level": 0,
            "max_iter": self.max_iter,
            "sb": "yes",
            "print_timing_statistics": "no",
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
        }
        self.opti.solver("ipopt", p_opts, s_opts)

        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] OCP setup complete.",
        )

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        t_start = time.perf_counter()

        self.opti.set_value(self.x_curr, x_curr)
        self.opti.set_value(self.x_goal, x_goal)
        if self.du_weight > 0:
            self.opti.set_value(self.u_prev_p, self._u_curr)
        self._set_initial_guess(x_curr)

        try:
            sol = self.opti.solve()
            self._prev_x_sol = sol.value(self.X)
            self._prev_u_sol = sol.value(self.U)
            self._u_curr = self._prev_u_sol[:, 0]
            stats = sol.stats()

        except RuntimeError as e:
            self.solve_failures += 1
            if self._prev_u_sol is not None:
                self._u_curr = self._prev_u_sol[:, 1]
                self._prev_u_sol = np.column_stack([
                    self._prev_u_sol[:, 1:],
                    self._prev_u_sol[:, -1],
                ])
                self._prev_x_sol = np.column_stack([
                    self._prev_x_sol[:, 1:],
                    self._prev_x_sol[:, -1],
                ])
                self.bskLogger.bskLog(
                    bskLogging.BSK_WARNING,
                    f"[{self.ModelTag}] MPC solve failed: {e}. Using shifted control.",
                )
            else:
                self.bskLogger.bskLog(
                    bskLogging.BSK_WARNING,
                    f"[{self.ModelTag}] MPC solve failed: {e}. Using fallback control.",
                )
            try:
                stats = self.opti.debug.stats()
            except RuntimeError:
                stats = None

        t_end = time.perf_counter()
        total_time = t_end - t_start
        if stats is not None and "t_wall_total" in stats:
            self.wall_times.append(float(stats["t_wall_total"]))
        self.total_times.append(total_time)

        return self._u_curr.copy()

    def _set_initial_guess(self, x_curr: np.ndarray) -> None:
        if self._prev_x_sol is not None and self._prev_u_sol is not None:
            x_warm = np.zeros((self.nx, self.horizon + 1))
            u_warm = np.zeros((self.nu, self.horizon))
            x_warm[:, :-1] = self._prev_x_sol[:, 1:]
            x_warm[:, -1] = self._prev_x_sol[:, -1]
            u_warm[:, :-1] = self._prev_u_sol[:, 1:]
            u_warm[:, -1] = self._prev_u_sol[:, -1]
            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
            if self._use_koz:
                self.opti.set_initial(
                    self.slack, np.zeros((self.num_bs * self.num_koz, self.horizon))
                )
        else:
            x_warm = np.tile(x_curr.reshape(-1, 1), (1, self.horizon + 1))
            u_warm = np.zeros((self.nu, self.horizon))
            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
            if self._use_koz:
                self.opti.set_initial(
                    self.slack, np.zeros((self.num_bs * self.num_koz, self.horizon))
                )

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
