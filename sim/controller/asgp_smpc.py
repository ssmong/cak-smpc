"""Adaptive Sparse GP MPC controller.

``ASGPSMPC`` — Basilisk MPC controller that uses a
:class:`~src.model.adaptive_sgp.MultiOutputASGP` for learned residual
dynamics and analytical Jacobians (Hewing et al. 2020, eq. 8).

GP predicts dω (3D) from (ω, u) input (6D).
Quaternion correction handled by nominal dynamics.
"""

import os
import time
import logging

import numpy as np
import casadi as ca
from scipy.linalg import solve_triangular
from scipy.stats import norm
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging, bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from sim.utils.math_utils import quat_error_ca
from src.model.adaptive_sgp import MultiOutputASGP
from src.utils.math_numpy import (
    attitude_dyn_dt,
    attitude_dyn_dt_jac,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASGP_SMPC")


# ---------------------------------------------------------------------------
#  ASGPSMPC  — Basilisk MPC controller
# ---------------------------------------------------------------------------


class ASGPSMPC(sysModel.SysModel):
    def __init__(
        self,
        model_root: str,
        exp_name: str,
        horizon: int,
        ctrl_dt: float,
        mpc_dt: float,
        inertia: list,
        Q_diag: list,
        R_diag: list,
        F_diag: list,
        rho: float,
        max_iter: int,
        integrator_steps: int,
        cc: DictConfig,
        gp: DictConfig,
        constraint_cfg: DictConfig,
        du_weight: float = 0.0,
        **kwargs,
    ):
        super(ASGPSMPC, self).__init__()
        self.ModelTag = "ASGP_SMPC"

        self.du_weight = du_weight
        self.nx = 7
        self.nu = 3
        self.horizon = int(horizon)
        self.ctrl_dt = float(ctrl_dt)
        self.mpc_dt = float(mpc_dt)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.integrator_steps = int(integrator_steps)

        self.mpc_call_interval = int(round(self.mpc_dt / self.ctrl_dt))
        if not np.isclose(self.mpc_call_interval * self.ctrl_dt, self.mpc_dt, rtol=1e-6):
            raise ValueError(
                f"mpc_dt ({self.mpc_dt}) must be integer multiple of ctrl_dt ({self.ctrl_dt})"
            )

        self.J = np.array(inertia, dtype=float).reshape(3, 3)
        self.J_inv = np.linalg.inv(self.J)
        self.Q = ca.diag(Q_diag)
        self.R = ca.diag(R_diag)
        self.F = ca.diag(F_diag)
        self.u_max = float(constraint_cfg.u_max)
        self.u_min = float(constraint_cfg.u_min)

        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.num_bs = quad_koz.num_bs
        self.num_koz = quad_koz.num_koz
        self.koz_matrices = np.zeros((self.num_bs, self.num_koz, 4, 4))
        for l, bs in enumerate(quad_koz.boresights):
            for m, koz in enumerate(quad_koz.koz_objects):
                self.koz_matrices[l, m] = quad_koz._build_koz_matrix(
                    koz.angle_rad, koz.direction, bs,
                )

        self.beta_cc = float(norm.ppf(float(cc.alpha)))
        self.max_tighten = cc.max_tighten
        self.var_floor = cc.var_floor
        self.var_scale = float(cc.var_scale)
        model_file = os.path.join(model_root, f"model_{exp_name}.npz")
        self.gp_model = MultiOutputASGP.from_pretrained(
            model_file=model_file,
            window_size=int(gp.window_size),
            num_inducing=int(gp.num_inducing),
            perc_th=float(gp.perc_th),
            lambda_forget=float(gp.lambda_forget),
        )

        logger.info(f"GP input_dim={self.gp_model.input_dim}, output_dim={self.gp_model.output_dim}")

        # Full-AGP vs Fast-AGP mode (Algorithm 2 vs Algorithm 1)
        self._gp_mode = str(gp.mode)  # "fast" or "full"
        self.gp_model._full_agp = (self._gp_mode == "full")
        self.gp_model._refresh_period = int(gp.refresh_period)
        self._hp_update_period = int(gp.hp_update_period)
        self._hp_num_steps = int(gp.hp_num_steps)
        self._hp_lr = float(gp.hp_lr)

        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        self.opti = None
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self._last_u_applied = np.zeros(self.nu)
        self._prev_x_meas = None
        self._ctrl_step_count = 0
        self._gp_update_count = 0
        self.wall_times = []
        self.total_times = []
        # GP prediction diagnostics
        self.gp_diag_pred = []    # predicted mean (raw space)
        self.gp_diag_actual = []  # actual residual (raw space)
        self.gp_diag_var = []     # predicted variance (raw space)
        self.gp_diag_M = []       # number of inducing points

    def Reset(self, CurrentSimNanos: int):
        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")
        self._setup_ocp()
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self._last_u_applied = np.zeros(self.nu)
        self._prev_x_meas = None
        self._ctrl_step_count = 0
        self._gp_update_count = 0
        self.wall_times = []
        self.total_times = []
        self.gp_diag_pred = []
        self.gp_diag_actual = []
        self.gp_diag_var = []
        self.gp_diag_M = []
        self.gp_diag_omega = []
        self.gp_diag_omega_nom = []
        zero = messaging.CmdTorqueBodyMsgPayload()
        zero.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        nav_msg = self.attNavInMsg()
        q_curr = rbk.MRP2EP(np.array(nav_msg.sigma_BN))
        w_curr = np.array(nav_msg.omega_BN_B)
        x_curr = np.concatenate([q_curr, w_curr])

        if self._prev_x_meas is not None:
            x_nom = self._nominal_step(self._prev_x_meas, self._last_u_applied, self.ctrl_dt)

            # 3D residual (dω only), 6D input (ω, u)
            resid = x_curr[4:] - x_nom[4:]
            z_prev_raw = np.concatenate([self._prev_x_meas[4:], self._last_u_applied])

            # Log GP prediction BEFORE update
            mu_pred, var_pred = self.gp_model.predict_np(z_prev_raw)
            self.gp_diag_pred.append(mu_pred.copy())
            self.gp_diag_actual.append(resid.copy())
            self.gp_diag_var.append(var_pred.copy())
            self.gp_diag_M.append(self.gp_model.models[0].M)
            self.gp_diag_omega.append(x_curr[4:].copy())
            self.gp_diag_omega_nom.append(x_nom[4:].copy())

            self.gp_model.update(z_prev_raw, resid)
            self._gp_update_count += 1

            # Full-AGP: optimize kernel HP + Xu_new location via ELBO
            if self._gp_mode == "full":
                if (
                    self._gp_update_count >= self._hp_update_period
                    and self._gp_update_count % self._hp_update_period == 0
                ):
                    self.gp_model.optimize_hyperparameters(
                        num_steps=self._hp_num_steps, lr=self._hp_lr,
                    )
                else:
                    self.gp_model.merge_pending_inducing()

        if self._ctrl_step_count % self.mpc_call_interval == 0:
            ref_msg = self.attRefInMsg()
            q_goal = rbk.MRP2EP(np.array(ref_msg.sigma_RN))
            x_goal = np.concatenate([q_goal, np.zeros(3)])
            self._u_curr = self._solve(x_curr, x_goal)

        out = messaging.CmdTorqueBodyMsgPayload()
        out.torqueRequestBody = self._u_curr.tolist()
        self.cmdTorqueOutMsg.write(out, CurrentSimNanos, self.moduleID)
        self._last_u_applied = self._u_curr.copy()
        self._prev_x_meas = x_curr.copy()
        self._ctrl_step_count += 1

    # -- OCP setup ----------------------------------------------------------

    def _setup_ocp(self):
        # No ca.integrator — inline RK4 is used directly in _f_full_ca

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)
        self.x_curr_p = self.opti.parameter(self.nx)
        self.x_goal_p = self.opti.parameter(self.nx, self.horizon + 1)
        if self.du_weight > 0:
            self.u_prev_p = self.opti.parameter(self.nu)

        # -- GP parameters for symbolic nonlinear dynamics --------------------
        M_max = self.gp_model._max_inducing
        Din_gp = self.gp_model.input_dim
        Dout_gp = self.gp_model.output_dim
        self._M_max = M_max
        self._Din_gp = Din_gp
        self._Dout_gp = Dout_gp

        self.gp_Xu_p = self.opti.parameter(M_max, Din_gp)
        self.gp_w_p = [self.opti.parameter(M_max) for _ in range(Dout_gp)]

        # Kernel hyperparameters — opti parameters so Full-AGP can update them
        self._gp_ls = [self.opti.parameter(Din_gp) for _ in range(Dout_gp)]
        self._gp_vs = [self.opti.parameter() for _ in range(Dout_gp)]
        self._gp_z_mean = ca.DM(self.gp_model._z_mean)
        self._gp_z_std = ca.DM(self.gp_model._z_std)
        self._gp_y_mean = ca.DM(self.gp_model._y_mean)
        self._gp_y_std = ca.DM(self.gp_model._y_std)

        # -- KOZ slacks -------------------------------------------------------
        self._use_koz = (self.num_bs > 0) and (self.num_koz > 0)
        if self._use_koz:
            self.slack = self.opti.variable(self.num_bs * self.num_koz, self.horizon)
            self.tighten_p = self.opti.parameter(self.num_bs * self.num_koz, self.horizon)
        else:
            self.slack = None
            self.tighten_p = None

        # -- cost + constraints -----------------------------------------------
        cost = 0.0
        self.opti.subject_to(self.X[:, 0] == self.x_curr_p)
        for k in range(self.horizon):
            q_err = quat_error_ca(self.X[:4, k], self.x_goal_p[:4, k])
            w_err = self.X[4:, k] - self.x_goal_p[4:, k]
            x_err = ca.vertcat(q_err, w_err)
            cost += ca.mtimes([x_err.T, self.Q, x_err])
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])

            # Control rate penalty
            if self.du_weight > 0:
                du_k = (self.U[:, k] - self.u_prev_p if k == 0
                        else self.U[:, k] - self.U[:, k - 1])
                cost += self.du_weight * ca.dot(du_k, du_k)

            # Nonlinear dynamics: f_nom(x,u) + GP correction
            x_next = self._f_full_ca(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == x_next)
            self.opti.subject_to(self.opti.bounded(self.u_min, self.U[:, k], self.u_max))

        q_err_f = quat_error_ca(self.X[:4, self.horizon], self.x_goal_p[:4, self.horizon])
        w_err_f = self.X[4:, self.horizon] - self.x_goal_p[4:, self.horizon]
        cost += ca.mtimes([
            ca.vertcat(q_err_f, w_err_f).T,
            self.F,
            ca.vertcat(q_err_f, w_err_f),
        ])

        if self._use_koz:
            cost += self.rho * ca.sumsqr(self.slack)
            self.opti.subject_to(ca.vec(self.slack) >= 0.0)
            for k in range(self.horizon):
                q_k = self.X[:4, k + 1]
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm_idx = l * self.num_koz + m
                        quad_val = ca.mtimes([q_k.T, ca.DM(self.koz_matrices[l, m]), q_k])
                        rhs = 2.0 - self.tighten_p[lm_idx, k] + self.slack[lm_idx, k]
                        self.opti.subject_to(quad_val <= rhs)

        self.opti.minimize(cost)
        self.opti.solver(
            "ipopt",
            {"expand": True, "print_time": False, "record_time": True, "verbose": False},
            {"print_level": 0, "max_iter": self.max_iter, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4},
        )

    # -- solve --------------------------------------------------------------

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.opti.set_value(self.x_curr_p, x_curr)
        self.opti.set_value(self.x_goal_p, np.tile(x_goal.reshape(-1, 1), (1, self.horizon + 1)))
        if self.du_weight > 0:
            self.opti.set_value(self.u_prev_p, self._u_curr)

        # GP weight vectors + kernel hyperparameters for symbolic dynamics
        Xu_padded, w_list = self._extract_gp_weights()
        self.opti.set_value(self.gp_Xu_p, Xu_padded)
        for j in range(self._Dout_gp):
            self.opti.set_value(self.gp_w_p[j], w_list[j])
            m = self.gp_model.models[j]
            self.opti.set_value(self._gp_ls[j], m.kernel.lengthscale.detach().numpy().reshape(-1))
            self.opti.set_value(self._gp_vs[j], float(m.kernel.variance.detach()))

        # Covariance propagation (pre-computed Jacobians) → tightening + x_nom
        x_nom, tighten = self._build_local_models(x_curr)
        if self._use_koz:
            self.opti.set_value(self.tighten_p, tighten)
        self._set_initial_guess(x_nom)
        try:
            sol = self.opti.solve()
            self._prev_x_sol = sol.value(self.X)
            self._prev_u_sol = sol.value(self.U)
            u = self._prev_u_sol[:, 0]
            stats = sol.stats()
        except RuntimeError as e:
            if self._prev_u_sol is not None:
                u = self._prev_u_sol[:, 1]
                self._prev_u_sol = np.column_stack([
                    self._prev_u_sol[:, 1:], self._prev_u_sol[:, -1],
                ])
            else:
                u = np.zeros(self.nu)
            self.bskLogger.bskLog(bskLogging.BSK_WARNING, f"[{self.ModelTag}] MPC solve failed: {e}")
            stats = self.opti.debug.stats()
        t1 = time.perf_counter()
        self.wall_times.append(float(stats["t_wall_total"]))
        self.total_times.append(float(t1 - t0))
        return u.copy()

    def _set_initial_guess(self, x_nom: np.ndarray):
        if self._prev_x_sol is not None and self._prev_u_sol is not None:
            x_warm = np.zeros((self.nx, self.horizon + 1))
            u_warm = np.zeros((self.nu, self.horizon))
            x_warm[:, :-1] = self._prev_x_sol[:, 1:]
            x_warm[:, -1] = self._prev_x_sol[:, -1]
            u_warm[:, :-1] = self._prev_u_sol[:, 1:]
            u_warm[:, -1] = self._prev_u_sol[:, -1]
            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
        else:
            self.opti.set_initial(self.X, x_nom.T)
            self.opti.set_initial(self.U, np.zeros((self.nu, self.horizon)))
        if self._use_koz:
            self.opti.set_initial(self.slack, np.zeros((self.num_bs * self.num_koz, self.horizon)))

    # -- local linearization (analytical Jacobian) ---------------------------

    def _build_local_models(self, x_curr: np.ndarray):
        """Propagate nominal trajectory and compute covariance tightening.

        Dynamics constraint is handled symbolically in the OCP (nonlinear).
        This method only computes:
          - x_nom: nominal trajectory for warm-starting.
          - tighten: chance-constraint tightening from Jacobian-based
            covariance propagation (Hewing et al. 2020, eq. 8b).
        """
        if self._prev_u_sol is not None:
            u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)
            u_nom[:-1] = self._prev_u_sol[:, 1:].T
            u_nom[-1] = self._prev_u_sol[:, -1]
        else:
            u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)

        x_nom = np.zeros((self.horizon + 1, self.nx), dtype=np.float64)
        x_nom[0] = x_curr.copy()
        A_seq = np.zeros((self.horizon, self.nx, self.nx), dtype=np.float64)
        q_proc_seq = np.zeros((self.horizon, self.nx, self.nx), dtype=np.float64)

        for k in range(self.horizon):
            xk, uk = x_nom[k], u_nom[k]

            # Nominal dynamics + analytical Jacobians (RK4 chain rule)
            x_nom_k, A_nom, _ = attitude_dyn_dt_jac(
                xk, uk, self.mpc_dt, self.J, self.J_inv,
                steps=self.integrator_steps,
            )

            # GP prediction + analytical Jacobian of GP mean
            gp_active = self._gp_update_count > 0

            z_raw = np.concatenate([xk[4:], uk])  # 6D: (ω, u)
            if gp_active:
                mu_res, var_res, dmu_dz = self.gp_model.predict_with_jacobian(z_raw)
            else:
                mu_res = np.zeros(3, dtype=np.float64)
                var_res = np.zeros(3, dtype=np.float64)
                dmu_dz = np.zeros((3, 6), dtype=np.float64)

            # Apply GP correction: quaternion from nominal only, omega corrected
            x_nom[k + 1, :4] = x_nom_k[:4]
            x_nom[k + 1, 4:] = x_nom_k[4:] + mu_res

            # Bd (7×3): constant [0₄ₓ₃; I₃]
            Bd = np.zeros((self.nx, 3), dtype=np.float64)
            Bd[4:, :] = np.eye(3)

            # Combined Jacobian: dmu/dx_full = [0₃ₓ₄ | dmu/dω₃ₓ₃]
            dmu_dw = dmu_dz[:, :3]  # (3, 3) — Jacobian w.r.t. ω
            dmu_dx_full = np.zeros((3, self.nx), dtype=np.float64)
            dmu_dx_full[:, 4:] = dmu_dw
            A_seq[k] = A_nom + Bd @ dmu_dx_full

            # Process noise: only ω block from GP variance (scaled) + floor
            q_proc = np.zeros((self.nx, self.nx), dtype=np.float64)
            scale = self.var_scale if self._gp_mode == "fast" else 1.0
            q_proc[4:, 4:] = np.diag(np.maximum(var_res * scale, self.var_floor))
            q_proc += 1e-12 * np.eye(self.nx)
            q_proc_seq[k] = q_proc

        # Chance constraint tightening
        tighten = np.zeros((self.num_bs * self.num_koz, self.horizon), dtype=np.float64)
        if self._use_koz:
            P = np.zeros((self.nx, self.nx), dtype=np.float64)
            for k in range(self.horizon):
                P = A_seq[k] @ P @ A_seq[k].T + q_proc_seq[k]
                qk = x_nom[k + 1, :4]
                Pq = P[:4, :4]
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm_idx = l * self.num_koz + m
                        grad_q = 2.0 * (self.koz_matrices[l, m] @ qk)
                        sigma = np.sqrt(max(float(grad_q.T @ Pq @ grad_q), 0.0))
                        tighten_raw = self.beta_cc * sigma
                        tighten[lm_idx, k] = np.clip(
                            tighten_raw, 0.0, self.max_tighten,
                        )
        return x_nom, tighten

    # -- CasADi symbolic dynamics ------------------------------------------

    def _attitude_dynamics(self, x, u):
        """Continuous-time attitude dynamics (CasADi symbolic).

        Identical to NMPC._attitude_dynamics / UKFSMPC._attitude_dynamics.
        """
        q0, q1, q2, q3 = x[0], x[1], x[2], x[3]
        w = x[4:]
        q_dot = 0.5 * ca.vertcat(
            -q1 * w[0] - q2 * w[1] - q3 * w[2],
             q0 * w[0] - q3 * w[1] + q2 * w[2],
             q3 * w[0] + q0 * w[1] - q1 * w[2],
            -q2 * w[0] + q1 * w[1] + q0 * w[2],
        )
        Jw = ca.mtimes(ca.DM(self.J), w)
        w_dot = ca.mtimes(ca.DM(self.J_inv), u - ca.cross(w, Jw))
        return ca.vertcat(q_dot, w_dot)

    def _gp_mean_ca(self, x, u):
        """CasADi symbolic GP mean (denormalized to raw residual scale)."""
        z_raw = ca.vertcat(x[4:], u)  # 6D: (ω, u)
        z_scaled = (z_raw - self._gp_z_mean) / self._gp_z_std

        mu_s = []
        for j in range(self._Dout_gp):
            diff = self.gp_Xu_p - ca.repmat(z_scaled.T, self._M_max, 1)
            scaled_diff = diff / ca.repmat(self._gp_ls[j].T, self._M_max, 1)
            dist2 = ca.sum2(scaled_diff ** 2)
            k_vec = self._gp_vs[j] * ca.exp(-0.5 * dist2)
            mu_s.append(ca.dot(self.gp_w_p[j], k_vec))
        # Denormalize: mu_raw = mu_s * y_std + y_mean
        return ca.vertcat(*mu_s) * self._gp_y_std + self._gp_y_mean

    def _symbolic_rk4_step(self, x, u):
        """Inline RK4 step — same as NMPC (faster than ca.integrator)."""
        h = self.mpc_dt / self.integrator_steps
        xk = x
        for _ in range(self.integrator_steps):
            k1 = self._attitude_dynamics(xk, u)
            k2 = self._attitude_dynamics(xk + 0.5 * h * k1, u)
            k3 = self._attitude_dynamics(xk + 0.5 * h * k2, u)
            k4 = self._attitude_dynamics(xk + h * k3, u)
            xk = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        q = xk[:4]
        q_norm = ca.fmax(ca.norm_2(q), 1e-8)
        return ca.vertcat(q / q_norm, xk[4:])

    def _f_full_ca(self, x, u):
        """Full nonlinear dynamics: f_nom(x,u) + GP correction.

        q_next = q_nom  (nominal dynamics only)
        w_next = w_nom + mu_gp  (additive dω correction)
        """
        # Nominal dynamics via inline RK4 (same as NMPC)
        x_nom = self._symbolic_rk4_step(x, u)
        q_nom = x_nom[:4]
        w_nom = x_nom[4:]

        mu_gp = self._gp_mean_ca(x, u)

        q_next = q_nom
        w_next = w_nom + mu_gp

        return ca.vertcat(q_next, w_next)

    # -- GP weight extraction -----------------------------------------------

    def _extract_gp_weights(self):
        """Pre-compute GP weight vectors so mu_j = w_j^T @ k(Xu, z).

        Returns:
            Xu_padded: (M_max, Din) inducing points (in scaled space).
            w_list:    list of (M_max,) weight vectors per output.
        """
        gp = self.gp_model
        M_curr = gp.models[0].M
        Xu_padded = np.zeros((self._M_max, self._Din_gp), dtype=np.float64)
        if M_curr > 0:
            Xu_padded[:M_curr] = gp.models[0].Xu.detach().numpy()

        gp_active = self._gp_update_count > 0
        w_list = []
        for j, m in enumerate(gp.models):
            if not gp_active or m.N == 0 or M_curr == 0:
                w_list.append(np.zeros(self._M_max, dtype=np.float64))
                continue

            Wt = m.W.detach().t().contiguous().numpy()
            D = (m.noise.detach() / m.Lambda.detach()).numpy()
            W_Dinv = Wt / D
            W_Dinv_y = W_Dinv @ m.y.detach().numpy()

            K = W_Dinv @ Wt.T
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, K.diagonal() + 1.0)
            L = np.linalg.cholesky(K)

            Luu = m.Luu.detach().numpy()
            if not np.all(np.isfinite(Luu)):
                w_list.append(np.zeros(self._M_max, dtype=np.float64))
                continue
            alpha = solve_triangular(L, W_Dinv_y, lower=True)
            Luu_inv = solve_triangular(Luu, np.eye(M_curr), lower=True)
            C = solve_triangular(L, Luu_inv, lower=True)
            w = C.T @ alpha

            w_padded = np.zeros(self._M_max, dtype=np.float64)
            w_padded[:M_curr] = w
            w_list.append(w_padded)

        return Xu_padded, w_list

    # -- dynamics -----------------------------------------------------------

    def _nominal_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        return attitude_dyn_dt(x, u, dt, self.J, self.J_inv, steps=self.integrator_steps)

    # -- timing stats -------------------------------------------------------

    def get_wall_time_stats(self) -> dict:
        if not self.wall_times:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": float(np.mean(self.wall_times)),
            "max": float(np.max(self.wall_times)),
            "min": float(np.min(self.wall_times)),
            "std": float(np.std(self.wall_times)),
            "count": len(self.wall_times),
        }

    def get_total_time_stats(self) -> dict:
        if not self.total_times:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": float(np.mean(self.total_times)),
            "max": float(np.max(self.total_times)),
            "min": float(np.min(self.total_times)),
            "std": float(np.std(self.total_times)),
            "count": len(self.total_times),
        }
