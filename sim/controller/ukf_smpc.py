"""
UKF-based Stochastic NMPC for spacecraft attitude control (SCP coupling).

Uses Sequential Convex Programming to couple uncertainty propagation with
the optimized control sequence.  At each MPC step the controller:
  1. Propagates sigma points in NumPy along the current nominal u → tightening
  2. Solves the tightened NLP (same size as deterministic NMPC)
  3. Re-propagates using the new u; repeats until convergence or max iterations

This achieves the accuracy benefit of a coupled formulation (tightening
depends on the *optimised* u, not just u_prev) without embedding sigma
points as NLP decision variables.

Includes a 13-state augmented UKF estimator (q4 + omega3 + j6) for
online joint attitude and inertia estimation.  The estimator outputs:
  1. J_hat   (3×3)  — injected into the CasADi NLP as a parameter
  2. P_{q,ω} (7×7)  — used as initial covariance P0 for prediction UKF
  3. P_J     (6×6)  — inflates Sigma_w via finite-difference sensitivity

Reference:
    Bradford & Imsland (2018), "Economic stochastic model predictive
    control using the unscented Kalman filter", IFAC-PapersOnLine.
"""

import time
import logging
import numpy as np
import casadi as ca
from scipy.stats import norm
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from sim.utils.math_utils import quat_error_ca
from src.utils.math_numpy import (
    attitude_dyn_dt, attitude_dyn_dt_batch,
    j_vec2mat, j_mat_to_vec,
)
from sim.estimator.augmented_ukf import AugmentedUKF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UKF_SMPC")


class UKFSMPC(sysModel.SysModel):
    """
    CasADi-based Stochastic NMPC with SCP-coupled UKF tightening.

    - Same NLP size as deterministic NMPC (no extra sigma-point variables)
    - KOZ tightening updated via SCP outer loop (re-propagate → re-solve)
    - Process noise Sigma_w calibrated offline, inflated by inertia uncertainty
    - Robust horizon N_robust: covariance propagated for
      min(horizon, N_robust) steps, then frozen
    """

    def __init__(self,
                 horizon: int,
                 mpc_dt: float,
                 est_dt: float,
                 inertia: list,
                 Q_diag: list,
                 R_diag: list,
                 F_diag: list,
                 rho: float,
                 max_iter: int,
                 constraint_cfg: DictConfig,
                 integrator_steps: int,
                 du_weight: float,
                 cc: DictConfig,
                 estimator: DictConfig,
                 **kwargs):
        super(UKFSMPC, self).__init__()
        self.ModelTag = "UKF_SMPC"

        # --- Validate -------------------------------------------------------
        if mpc_dt <= 0:
            raise ValueError(f"mpc_dt must be positive: {mpc_dt}")
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2: {horizon}")

        self.horizon = horizon
        self.mpc_dt = mpc_dt
        self.est_dt = est_dt
        self.max_iter = max_iter

        # MPC solve cadence: solve every mpc_steps estimator updates
        self._mpc_steps = max(1, int(round(mpc_dt / est_dt)))
        self._step_counter = 0
        self.rho = rho
        self.integrator_steps = integrator_steps
        self.du_weight = du_weight

        # --- SCP coupling parameters ----------------------------------------
        self.scp_max_iter = int(cc.scp_max_iter)
        self.scp_tol = float(cc.scp_tol)

        # --- Inertia (nominal) ----------------------------------------------
        inertia_arr = np.array(inertia)
        if inertia_arr.size != 9:
            raise ValueError(f"Inertia must be 9 elements (3x3 row-major), got {inertia_arr.size}")
        self.J = inertia_arr.reshape(3, 3)
        self.J_inv = np.linalg.inv(self.J)

        # --- Cost matrices (6D: quat_err[3] + omega_err[3]) -----------------
        self.Q = ca.diag(Q_diag)
        self.R = ca.diag(R_diag)
        self.F = ca.diag(F_diag)

        # --- Constraints ----------------------------------------------------
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        # KOZ quadratic matrices
        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.num_bs = quad_koz.num_bs
        self.num_koz = quad_koz.num_koz
        self.koz_matrices = np.zeros((self.num_bs, self.num_koz, 4, 4))
        for l, bs in enumerate(quad_koz.boresights):
            for m, koz in enumerate(quad_koz.koz_objects):
                self.koz_matrices[l, m] = quad_koz._build_koz_matrix(
                    angle_rad=koz.angle_rad,
                    x=koz.direction,
                    y=bs,
                )

        # --- Chance constraint (Boole's inequality) -------------------------
        self.cc_alpha = float(cc.alpha)
        risk_total = 1.0 - self.cc_alpha
        if self.horizon > 0:
            risk_per = risk_total / float(self.horizon)
            self.k_alpha = float(norm.ppf(1.0 - risk_per))
        else:
            self.k_alpha = 0.0

        self.Sigma_w = np.diag(np.array(cc.Sigma_w_diag, dtype=float))
        self.P0 = self.Sigma_w.copy()
        self.N_robust = int(cc.N_robust)
        self.ut_alpha = float(cc.ut_alpha)
        self.ut_beta = float(cc.ut_beta)
        self.ut_kappa = float(cc.ut_kappa)

        # --- Augmented-state estimator (13-state UKF) -------------------------
        self.j_nom = j_mat_to_vec(self.J)
        self._base_Sigma_w = self.Sigma_w.copy()

        x0_state = np.concatenate([np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)])
        P0_state = self.Sigma_w.copy()
        P0_J = np.diag(np.array(estimator.P_J_0_diag, dtype=float))
        # Q_state_diag is calibrated at mpc_dt; scale to est_dt (variance ~ dt)
        est_scale = est_dt / mpc_dt
        Q_state = np.diag(np.array(estimator.Q_state_diag, dtype=float)) * est_scale
        Q_j = np.diag(np.array(estimator.Q_J_diag, dtype=float)) * est_scale
        R_meas = np.diag(np.array(estimator.R_meas_diag, dtype=float))

        self._estimator = AugmentedUKF(
            x0_state=x0_state,
            j_nom=self.j_nom,
            P0_state=P0_state,
            P0_J=P0_J,
            Q_state=Q_state,
            Q_j=Q_j,
            R_meas=R_meas,
            dt=est_dt,
            integrator_steps=integrator_steps,
            ut_alpha=float(estimator.ut_alpha),
            ut_beta=float(estimator.ut_beta),
            ut_kappa=float(estimator.ut_kappa),
        )
        self._j_hat_vec = self.j_nom.copy()
        self._estimator_cfg = estimator  # keep for reset
        self._estimator_initialized = False

        # Pre-compute UT constants (n = 7, 2n+1 = 15 sigma points)
        self.nx = 7
        self.nu = 3
        n = self.nx
        lam = self.ut_alpha**2 * (n + self.ut_kappa) - n
        self._ut_lam = lam
        self._ut_n_pts = 2 * n + 1
        self._ut_scale = n + lam

        # Pre-compute weights (fixed for all steps)
        self._Wm = np.full(self._ut_n_pts, 0.5 / self._ut_scale)
        self._Wc = np.full(self._ut_n_pts, 0.5 / self._ut_scale)
        self._Wm[0] = lam / self._ut_scale
        self._Wc[0] = lam / self._ut_scale + (1.0 - self.ut_alpha**2 + self.ut_beta)

        # Solver (initialized in Reset)
        self.opti = None

        # Warm start
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)

        # Timing & solver tracking
        self.total_times: list[float] = []
        self.wall_times: list[float] = []
        self.solve_failures: int = 0
        self.solver_statuses: list[str] = []
        self.scp_iters: list[int] = []

        # Estimator history (for diagnostics / plotting)
        self._j_hat_history: list[np.ndarray] = []
        self._P_J_diag_history: list[np.ndarray] = []
        self._x_est_history: list[np.ndarray] = []

        # BSK messages
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        logger.info(
            f"[UKF_SMPC] Initialized. horizon={horizon}, mpc_dt={mpc_dt}s, "
            f"est_dt={est_dt}s (MPC every {self._mpc_steps} steps), "
            f"cc_alpha={self.cc_alpha}, k_alpha={self.k_alpha:.3f}, "
            f"N_robust={self.N_robust}, n_sigma_pts={self._ut_n_pts}, "
            f"scp_max_iter={self.scp_max_iter}, scp_tol={self.scp_tol}")

    # ======================================================================
    #  BSK lifecycle
    # ======================================================================

    def Reset(self, CurrentSimNanos: int):
        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] Resetting: mpc_dt={self.mpc_dt}s, "
            f"horizon={self.horizon}, J_diag={np.diag(self.J).tolist()}")

        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")

        self._setup_ocp()

        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self.total_times = []
        self.wall_times = []
        self.solve_failures = 0
        self.solver_statuses = []
        self.scp_iters = []
        self._j_hat_history = []
        self._P_J_diag_history = []
        self._x_est_history = []

        # Reset estimator to nominal
        x0_state = np.concatenate([np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)])
        P0_state = self._base_Sigma_w.copy()
        P0_J = np.diag(np.array(self._estimator_cfg.P_J_0_diag, dtype=float))
        self._estimator.reset(x0_state, self.j_nom, P0_state, P0_J)
        self._j_hat_vec = self.j_nom.copy()
        self._estimator_initialized = False
        self._step_counter = self._mpc_steps  # force MPC solve on first call
        self.Sigma_w = self._base_Sigma_w.copy()

        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega_BN_B = np.array(nav_msg.omega_BN_B)

        q_curr = rbk.MRP2EP(sigma_BN)
        x_meas = np.concatenate([q_curr, omega_BN_B])

        # --- Estimator predict → update (every est_dt) ---
        if not self._estimator_initialized:
            # First call: seed estimator with actual measurement instead of
            # identity quaternion, so it doesn't start confidently wrong.
            P0_state = self._base_Sigma_w.copy()
            P0_J = np.diag(np.array(self._estimator_cfg.P_J_0_diag, dtype=float))
            self._estimator.reset(x_meas, self.j_nom, P0_state, P0_J)
            self._estimator_initialized = True
        else:
            self._estimator.predict(self._u_curr)
        self._estimator.update(x_meas)

        # Extract estimator outputs
        x_est = self._estimator.get_state_mean()[:7]
        J_hat = self._estimator.get_J_hat()
        self._j_hat_vec = self._estimator.get_j_vec()
        P_qw = self._estimator.get_P_state()
        P_J = self._estimator.get_P_J()

        # Record estimator history
        self._j_hat_history.append(self._j_hat_vec.copy())
        self._P_J_diag_history.append(np.diag(P_J).copy())
        self._x_est_history.append(x_est.copy())

        # Update MPC internals with latest estimator outputs
        self.P0 = P_qw
        self.J = J_hat
        self.J_inv = np.linalg.inv(J_hat)

        # --- MPC solve cadence: only re-solve every mpc_steps calls ---
        self._step_counter += 1
        if self._step_counter >= self._mpc_steps:
            self._step_counter = 0

            self._update_sigma_w(x_est, P_J)

            ref_msg = self.attRefInMsg()
            sigma_RN = np.array(ref_msg.sigma_RN)
            q_goal = rbk.MRP2EP(sigma_RN)
            omega_goal = np.zeros(3)

            state_goal = np.concatenate([q_goal, omega_goal])
            x_goal = np.tile(state_goal.reshape(-1, 1), (1, self.horizon + 1))

            self._solve(x_est, x_goal)  # updates self._u_curr internally

        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = self._u_curr.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

    # ======================================================================
    #  OCP setup
    # ======================================================================

    def _setup_ocp(self) -> None:
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)

        self._use_koz = (self.num_koz > 0) and (self.num_bs > 0)
        if self._use_koz:
            n_lm = self.num_bs * self.num_koz
            self.slack = self.opti.variable(n_lm, self.horizon)
            self.tighten_p = self.opti.parameter(n_lm, self.horizon)
        else:
            self.slack = None
            self.tighten_p = None

        # Parameters
        self.x_curr = self.opti.parameter(self.nx)
        self.x_goal = self.opti.parameter(self.nx, self.horizon + 1)
        self.u_prev_p = self.opti.parameter(self.nu)
        self.J_param = self.opti.parameter(6)  # [J11,J12,J13,J22,J23,J33]

        # --- Build cost & constraints ---------------------------------------
        cost = 0.0
        self.opti.subject_to(self.X[:, 0] == self.x_curr)

        for k in range(self.horizon):
            # Stage cost
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

            # Dynamics (symbolic RK4 with parametric J)
            x_next = self._symbolic_rk4_step(
                self.X[:, k], self.U[:, k], self.J_param,
                self.mpc_dt, self.integrator_steps)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

            # Input bounds
            self.opti.subject_to(self.opti.bounded(
                self.u_min, self.U[:, k], self.u_max))

        # Terminal cost
        q_err_f = quat_error_ca(
            self.X[:4, self.horizon], self.x_goal[:4, self.horizon])
        w_err_f = self.X[4:, self.horizon] - self.x_goal[4:, self.horizon]
        x_err_f = ca.vertcat(q_err_f, w_err_f)
        cost += ca.mtimes([x_err_f.T, self.F, x_err_f])

        # KOZ constraints with UKF tightening
        if self._use_koz:
            cost += self.rho * ca.sumsqr(self.slack)
            self.opti.subject_to(ca.vec(self.slack) >= 0.0)
            for k in range(self.horizon):
                q_k = self.X[:4, k + 1]
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm = l * self.num_koz + m
                        M_tilde = ca.DM(self.koz_matrices[l, m])
                        quad_val = ca.mtimes([q_k.T, M_tilde, q_k])
                        rhs = (2.0
                               - self.tighten_p[lm, k]
                               + self.slack[lm, k])
                        self.opti.subject_to(quad_val <= rhs)

        self.opti.minimize(cost)

        # Solver options
        p_opts = {'expand': True, 'print_time': False, 'record_time': True, 'verbose': False}
        s_opts = {
            'print_level': 0,
            'max_iter': self.max_iter,
            'sb': 'yes',
            'print_timing_statistics': 'no',
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    # ======================================================================
    #  Solve (SCP outer loop)
    # ======================================================================

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        t_start = time.perf_counter()

        self.opti.set_value(self.x_curr, x_curr)
        self.opti.set_value(self.x_goal, x_goal)
        self.opti.set_value(self.u_prev_p, self._u_curr)
        self.opti.set_value(self.J_param, self._j_hat_vec)
        self._set_initial_guess(x_curr)

        # Initial nominal u for tightening (shifted previous solution)
        if self._prev_u_sol is not None:
            u_nom = np.zeros((self.horizon, self.nu))
            u_nom[:-1] = self._prev_u_sol[:, 1:].T
            u_nom[-1] = self._prev_u_sol[:, -1]
        else:
            u_nom = np.zeros((self.horizon, self.nu))

        stats = None
        n_scp = 0

        for scp_it in range(self.scp_max_iter):
            n_scp = scp_it + 1

            # --- Tightening from current u_nom ---
            if self._use_koz:
                tighten = self._compute_tightening(x_curr, u_nom)
                self.opti.set_value(self.tighten_p, tighten)

            # --- NLP solve ---
            try:
                sol = self.opti.solve()
                self._prev_x_sol = sol.value(self.X)
                self._prev_u_sol = sol.value(self.U)
                self._u_curr = self._prev_u_sol[:, 0]
                stats = sol.stats()

            except RuntimeError as e:
                self.solve_failures += 1
                stats = self.opti.debug.stats()
                return_status = str(stats.get('return_status', 'unknown'))

                if return_status in ('Maximum_Iterations_Exceeded',
                                     'Solved_To_Acceptable_Level'):
                    try:
                        debug_x = self.opti.debug.value(self.X)
                        debug_u = self.opti.debug.value(self.U)
                        debug_u = np.clip(debug_u, self.u_min, self.u_max)

                        self._prev_x_sol = debug_x
                        self._prev_u_sol = debug_u
                        self._u_curr = debug_u[:, 0]

                        self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                            f"[{self.ModelTag}] SCP iter {n_scp}: "
                            f"Solver {return_status}, using last iterate.")
                    except Exception:
                        self._fallback_control(e)
                        break
                else:
                    self._fallback_control(e)
                    break

            # --- Check SCP convergence ---
            u_new = self._prev_u_sol.T  # (horizon, 3)
            du_norm = np.linalg.norm(u_new - u_nom)
            u_scale = max(np.linalg.norm(u_new), 1e-8)

            if du_norm / u_scale < self.scp_tol:
                break

            # Update nominal for next SCP iteration
            u_nom = u_new.copy()

            # Warm-start next NLP from current solution
            self.opti.set_initial(self.X, self._prev_x_sol)
            self.opti.set_initial(self.U, self._prev_u_sol)

        self.scp_iters.append(n_scp)

        t_end = time.perf_counter()
        if stats is not None:
            self.wall_times.append(float(stats['t_wall_total']))
        self.total_times.append(t_end - t_start)
        if stats is not None:
            self.solver_statuses.append(str(stats.get('return_status', 'unknown')))

        return self._u_curr.copy()

    def _fallback_control(self, error: Exception) -> None:
        """Fallback to shifted previous solution when solver truly fails."""
        if self._prev_u_sol is not None:
            self._u_curr = self._prev_u_sol[:, min(1, self._prev_u_sol.shape[1] - 1)]
            self._prev_u_sol = np.column_stack([
                self._prev_u_sol[:, 1:], self._prev_u_sol[:, -1]])
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] Solve failed: {error}. Using shifted prev.")
        else:
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] Solve failed: {error}. No prev, zero torque.")

    # ======================================================================
    #  UKF tightening
    # ======================================================================

    def _compute_tightening(self, x_curr: np.ndarray,
                            u_nom: np.ndarray) -> np.ndarray:
        """Compute KOZ tightening via UKF covariance propagation.

        Args:
            x_curr: (7,)        current state.
            u_nom:  (horizon, 3) nominal control sequence.

        Returns:
            tighten: (n_lm, horizon) tightening values.
        """
        n_lm = self.num_bs * self.num_koz
        tighten = np.zeros((n_lm, self.horizon))

        x_mean_seq, P_seq = self._ukf_propagate(x_curr, u_nom)

        for k in range(self.horizon):
            P_qq = P_seq[k + 1, :4, :4]
            q_k = x_mean_seq[k + 1, :4]

            for l in range(self.num_bs):
                for m in range(self.num_koz):
                    lm = l * self.num_koz + m
                    a_q = 2.0 * self.koz_matrices[l, m] @ q_k
                    sigma2 = a_q @ P_qq @ a_q
                    tighten[lm, k] = self.k_alpha * np.sqrt(max(sigma2, 0.0))

        return tighten

    # ======================================================================
    #  Sigma_w inflation (inertia uncertainty → process noise)
    # ======================================================================

    def _update_sigma_w(self, x_est: np.ndarray, P_J: np.ndarray) -> None:
        """Inflate Sigma_w by inertia uncertainty via finite-difference discrete sensitivity.

        Computes full 7×6 discrete Jacobian dx_next/dj by perturbing each of the
        6 inertia elements through the same RK4 integrator used everywhere else.
        """
        eps = 1e-5
        j_hat = self._j_hat_vec
        x0_nom = attitude_dyn_dt(
            x_est, self._u_curr, self.mpc_dt,
            self.J, self.J_inv, steps=self.integrator_steps)

        G = np.zeros((7, 6))
        for i in range(6):
            j_pert = j_hat.copy()
            j_pert[i] += eps
            J_pert = j_vec2mat(j_pert)
            J_pert_inv = np.linalg.inv(J_pert)
            x_pert = attitude_dyn_dt(
                x_est, self._u_curr, self.mpc_dt,
                J_pert, J_pert_inv, steps=self.integrator_steps)
            G[:, i] = (x_pert - x0_nom) / eps

        Sigma_extra = G @ P_J @ G.T
        Sigma_extra = 0.5 * (Sigma_extra + Sigma_extra.T)
        self.Sigma_w = self._base_Sigma_w + Sigma_extra

    # ======================================================================
    #  UKF sigma-point propagation
    # ======================================================================

    def _ukf_propagate(self,
                       x0: np.ndarray,
                       u_seq: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate mean and covariance via unscented transform.

        Robust horizon: covariance propagated for min(horizon, N_robust)
        steps, then frozen.

        Args:
            x0:    (7,)   initial state mean.
            u_seq: (N, 3) nominal control sequence.
        Returns:
            x_mean_seq: (N+1, 7)    mean trajectory.
            P_seq:      (N+1, 7, 7) covariance sequence.
        """
        N = u_seq.shape[0]
        N_rob = min(self.N_robust, N)

        x_mean_seq = np.zeros((N + 1, self.nx))
        P_seq = np.zeros((N + 1, self.nx, self.nx))
        x_mean_seq[0] = x0.copy()
        P_seq[0] = self.P0.copy()

        for k in range(N):
            if k < N_rob:
                # Generate sigma points
                sigma_pts = self._generate_sigma_points(
                    x_mean_seq[k], P_seq[k])            # (15, 7)

                # Propagate all sigma points (vectorized)
                sigma_pts_next = self._batch_integrate(
                    sigma_pts, u_seq[k])                 # (15, 7)

                # Recover mean and covariance
                x_mean_seq[k + 1], P_seq[k + 1] = self._ut_recover(
                    sigma_pts_next)
            else:
                # Beyond robust horizon: mean only, covariance frozen
                x_mean_seq[k + 1] = self._single_integrate(
                    x_mean_seq[k], u_seq[k])
                P_seq[k + 1] = P_seq[N_rob].copy()

        return x_mean_seq, P_seq

    def _generate_sigma_points(self,
                               x_mean: np.ndarray,
                               P: np.ndarray) -> np.ndarray:
        """Generate 2n+1 = 15 sigma points.

        Args:
            x_mean: (7,) state mean.
            P: (7,7) state covariance.
        Returns:
            sigma_pts: (15, 7) sigma points.
        """
        n = self.nx
        S = _safe_cholesky(self._ut_scale * P)  # (7, 7)

        sigma_pts = np.empty((self._ut_n_pts, n))
        sigma_pts[0] = x_mean
        sigma_pts[1:n + 1] = x_mean + S.T       # rows of S^T
        sigma_pts[n + 1:] = x_mean - S.T

        # Normalize quaternion part of each sigma point
        _normalize_quat_batch(sigma_pts)
        return sigma_pts

    def _ut_recover(self,
                    sigma_pts: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover mean and covariance from propagated sigma points.

        Vectorized: no Python loop over sigma points.

        Args:
            sigma_pts: (2n+1, 7) propagated sigma points.
        Returns:
            x_mean: (7,) recovered mean.
            P: (7,7) recovered covariance (includes Sigma_w).
        """
        # Weighted mean: Wm @ sigma_pts → (7,)
        x_mean = self._Wm @ sigma_pts

        # Normalize quaternion
        q_norm = np.linalg.norm(x_mean[:4])
        if q_norm > 1e-8:
            x_mean[:4] /= q_norm

        # Weighted covariance: diff^T diag(Wc) diff + Sigma_w
        diff = sigma_pts - x_mean                 # (15, 7)
        P = (diff.T * self._Wc) @ diff            # (7, 7)
        P += self.Sigma_w
        P = 0.5 * (P + P.T)                       # enforce symmetry

        return x_mean, P

    # ======================================================================
    #  NumPy dynamics (delegates to src.utils.math_numpy)
    # ======================================================================

    def _batch_integrate(self, pts: np.ndarray,
                         u: np.ndarray) -> np.ndarray:
        """Integrate batch of states over one mpc_dt (sub-stepped RK4)."""
        return attitude_dyn_dt_batch(
            pts, u, self.mpc_dt, self.J, self.J_inv,
            steps=self.integrator_steps,
        )

    def _single_integrate(self, x: np.ndarray,
                          u: np.ndarray) -> np.ndarray:
        """Integrate single state over one mpc_dt."""
        return attitude_dyn_dt(
            x, u, self.mpc_dt, self.J, self.J_inv,
            steps=self.integrator_steps,
        )

    # ======================================================================
    #  CasADi symbolic dynamics (parametric J)
    # ======================================================================

    @staticmethod
    def _attitude_dynamics_param(x, u, J_sym, J_inv_sym):
        """Continuous-time attitude dynamics with parametric J (CasADi SX).

        Args:
            x:         (7,) SX state [q0,q1,q2,q3, w1,w2,w3].
            u:         (3,) SX control torque.
            J_sym:     (3,3) SX inertia matrix.
            J_inv_sym: (3,3) SX inverse inertia matrix.
        """
        q0, q1, q2, q3 = x[0], x[1], x[2], x[3]
        w = x[4:]
        q_dot = 0.5 * ca.vertcat(
            -q1 * w[0] - q2 * w[1] - q3 * w[2],
             q0 * w[0] - q3 * w[1] + q2 * w[2],
             q3 * w[0] + q0 * w[1] - q1 * w[2],
            -q2 * w[0] + q1 * w[1] + q0 * w[2],
        )
        Jw = ca.mtimes(J_sym, w)
        w_dot = ca.mtimes(J_inv_sym, u - ca.cross(w, Jw))
        return ca.vertcat(q_dot, w_dot)

    def _symbolic_rk4_step(self, x, u, j_vec, dt, substeps):
        """CasADi symbolic RK4 with parametric inertia.

        J and J_inv are computed once from j_vec and reused across all substeps.

        Args:
            x:        (7,) SX state.
            u:        (3,) SX control.
            j_vec:    (6,) SX inertia vector [J11,J12,J13,J22,J23,J33].
            dt:       float total step size.
            substeps: int   number of RK4 sub-steps.
        Returns:
            x_next: (7,) SX propagated state (quaternion normalized).
        """
        # Build J and J_inv once
        J_sym = ca.vertcat(
            ca.horzcat(j_vec[0], j_vec[1], j_vec[2]),
            ca.horzcat(j_vec[1], j_vec[3], j_vec[4]),
            ca.horzcat(j_vec[2], j_vec[4], j_vec[5]),
        )
        # Analytic 3×3 inverse (ca.inv uses LinsolQr which fails with MX + IPOPT)
        a, b, c = j_vec[0], j_vec[1], j_vec[2]
        d, e, f = j_vec[3], j_vec[4], j_vec[5]
        # J = [[a,b,c],[b,d,e],[c,e,f]]
        det = (a * (d * f - e * e)
               - b * (b * f - e * c)
               + c * (b * e - d * c))
        inv_det = 1.0 / det
        J_inv_sym = inv_det * ca.vertcat(
            ca.horzcat(d * f - e * e, c * e - b * f, b * e - c * d),
            ca.horzcat(c * e - b * f, a * f - c * c, b * c - a * e),
            ca.horzcat(b * e - c * d, b * c - a * e, a * d - b * b),
        )

        h = dt / substeps
        xk = x
        for _ in range(substeps):
            k1 = self._attitude_dynamics_param(xk, u, J_sym, J_inv_sym)
            k2 = self._attitude_dynamics_param(xk + 0.5 * h * k1, u, J_sym, J_inv_sym)
            k3 = self._attitude_dynamics_param(xk + 0.5 * h * k2, u, J_sym, J_inv_sym)
            k4 = self._attitude_dynamics_param(xk + h * k3, u, J_sym, J_inv_sym)
            xk = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Normalize quaternion
        q = xk[:4]
        q_norm = ca.fmax(ca.norm_2(q), 1e-8)
        return ca.vertcat(q / q_norm, xk[4:])

    # ======================================================================
    #  Warm start
    # ======================================================================

    def _set_initial_guess(self, x_curr: np.ndarray) -> None:
        n_lm = self.num_bs * self.num_koz
        if self._prev_x_sol is not None and self._prev_u_sol is not None:
            x_warm = np.column_stack([
                self._prev_x_sol[:, 1:], self._prev_x_sol[:, -1:]])
            u_warm = np.column_stack([
                self._prev_u_sol[:, 1:], self._prev_u_sol[:, -1:]])
            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
        else:
            self.opti.set_initial(
                self.X, np.tile(x_curr.reshape(-1, 1), (1, self.horizon + 1)))
            self.opti.set_initial(self.U, np.zeros((self.nu, self.horizon)))

        if self._use_koz:
            self.opti.set_initial(self.slack, np.zeros((n_lm, self.horizon)))

    # ======================================================================
    #  Timing stats
    # ======================================================================

    def get_wall_time_stats(self) -> dict:
        if not self.wall_times:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
        arr = np.array(self.wall_times)
        return {
            'mean': float(arr.mean()), 'max': float(arr.max()),
            'min': float(arr.min()), 'std': float(arr.std()),
            'count': len(self.wall_times),
        }

    def get_total_time_stats(self) -> dict:
        if not self.total_times:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
        arr = np.array(self.total_times)
        return {
            'mean': float(arr.mean()), 'max': float(arr.max()),
            'min': float(arr.min()), 'std': float(arr.std()),
            'count': len(self.total_times),
        }

    # ======================================================================
    #  Estimator diagnostics
    # ======================================================================

    def get_j_hat_history(self) -> np.ndarray:
        """(N, 6) estimated inertia vector at each est_dt step."""
        return np.array(self._j_hat_history) if self._j_hat_history else np.empty((0, 6))

    def get_P_J_diag_history(self) -> np.ndarray:
        """(N, 6) diagonal of P_J at each est_dt step."""
        return np.array(self._P_J_diag_history) if self._P_J_diag_history else np.empty((0, 6))

    def get_x_est_history(self) -> np.ndarray:
        """(N, 7) estimated [q, omega] at each est_dt step."""
        return np.array(self._x_est_history) if self._x_est_history else np.empty((0, 7))


# ======================================================================
#  Module-level utilities
# ======================================================================

def _normalize_quat_batch(x: np.ndarray) -> None:
    """In-place quaternion normalization for (M, 7) array."""
    q_norms = np.linalg.norm(x[:, :4], axis=1, keepdims=True)
    np.maximum(q_norms, 1e-8, out=q_norms)
    x[:, :4] /= q_norms


def _safe_cholesky(A: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Cholesky with fallback jitter."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
