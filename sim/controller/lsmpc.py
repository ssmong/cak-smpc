"""
Linear Stochastic MPC (L-SMPC) for spacecraft attitude control.

Linearizes the nominal dynamics along the predicted trajectory and propagates
a Gaussian covariance through the resulting time-varying linear model with an
offline-calibrated Σ_w, yielding a chance-constrained SOCP.

The KOZ constraints remain in their original quadratic form (q^T M q <= 2)
and are tightened by the propagated covariance.  The resulting SOCP
(quadratic cost, linear dynamics, SOC KOZ) is solved directly via
CLARABEL (interior-point conic solver).

Unlike UKF-SMPC, L-SMPC does NOT estimate inertia online.
It uses the nominal inertia throughout, and absorbs model mismatch
into a larger process noise covariance (Sigma_w) calibrated offline.

Reference:
    Bradford & Imsland (2018), "Economic stochastic model predictive
    control using the unscented Kalman filter", IFAC-PapersOnLine.
"""

import time
import logging
import numpy as np
import clarabel
import scipy.sparse as spa
from scipy.stats import norm
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from src.utils.math_numpy import attitude_dyn_dt_jac, quat_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("L_SMPC")


# ======================================================================
#  Module-level utilities
# ======================================================================

def _quat_error_jacobian(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Analytical 3x4 Jacobian of quat_error(q, q_ref) w.r.t. q.

    quat_error returns the vector part of conj(q) ⊗ q_ref:
        qe = q0*qr_vec - qr0*q_vec - cross(q_vec, qr_vec)

    Jacobian:
        dqe/dq0     = qr_vec
        dqe/dq_vec  = -qr0*I_3 + [qr_vec]_x
    """
    qr0 = q_ref[0]
    qr_v = q_ref[1:4]
    skew_qr = np.array([
        [0.0, -qr_v[2], qr_v[1]],
        [qr_v[2], 0.0, -qr_v[0]],
        [-qr_v[1], qr_v[0], 0.0],
    ], dtype=np.float64)
    J_q = np.zeros((3, 4), dtype=np.float64)
    J_q[:, 0] = qr_v
    J_q[:, 1:] = -qr0 * np.eye(3) + skew_qr
    return J_q


class LSMPC(sysModel.SysModel):
    """
    Linear Stochastic MPC (chance-constrained SOCP).

    - Linearizes nonlinear dynamics along the nominal trajectory
    - Propagates covariance analytically: P_{k+1} = A_k P_k A_k^T + Σ_w
    - KOZ constraints kept in original quadratic form, tightened by covariance
    - SOCP solved directly via Clarabel (interior-point conic solver)
    - Nominal inertia used throughout (no online estimation)
    - Robust horizon N_robust: covariance frozen after N_robust steps
    """

    def __init__(self,
                 horizon: int,
                 mpc_dt: float,
                 inertia: list,
                 Q_diag: list,
                 R_diag: list,
                 F_diag: list,
                 rho: float,
                 max_iter: int,
                 constraint_cfg: DictConfig,
                 du_weight: float,
                 cc: DictConfig,
                 **kwargs):
        super().__init__()
        self.ModelTag = "L_SMPC"

        if mpc_dt <= 0:
            raise ValueError(f"mpc_dt must be positive: {mpc_dt}")
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2: {horizon}")

        self.horizon = horizon
        self.mpc_dt = mpc_dt
        self.max_iter = max_iter
        self.rho = rho
        self.du_weight = du_weight

        # State / control dimensions
        self.nx = 7   # q(4) + omega(3)
        self.ne = 6   # quat_err(3) + omega_err(3)
        self.nu = 3

        # --- Inertia (nominal, fixed) ----------------------------------------
        inertia_arr = np.array(inertia)
        if inertia_arr.size != 9:
            raise ValueError(
                f"Inertia must be 9 elements (3x3 row-major), got {inertia_arr.size}")
        self.J = inertia_arr.reshape(3, 3)
        self.J_inv = np.linalg.inv(self.J)

        # --- Cost matrices (6D error space) ----------------------------------
        self.Q_diag = np.asarray(Q_diag, dtype=float)
        self.R_diag = np.asarray(R_diag, dtype=float)
        self.F_diag = np.asarray(F_diag, dtype=float)

        # --- Constraints ----------------------------------------------------
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        # KOZ quadratic matrices & Cholesky factors
        quad_koz = QuadKOZ.from_cfg(constraint_cfg)
        self.num_bs = quad_koz.num_bs
        self.num_koz = quad_koz.num_koz
        self._use_koz = (self.num_koz > 0) and (self.num_bs > 0)
        self._n_lm = self.num_bs * self.num_koz if self._use_koz else 0

        self.koz_matrices = np.zeros((self.num_bs, self.num_koz, 4, 4))
        self._koz_chol = np.zeros((self.num_bs, self.num_koz, 4, 4))
        for l, bs in enumerate(quad_koz.boresights):
            for m, koz in enumerate(quad_koz.koz_objects):
                M = quad_koz._build_koz_matrix(
                    angle_rad=koz.angle_rad, x=koz.direction, y=bs)
                self.koz_matrices[l, m] = M
                M_sym = 0.5 * (M + M.T)
                self._koz_chol[l, m] = np.linalg.cholesky(M_sym)

        # --- Chance constraint (Boole's inequality) -------------------------
        self.cc_alpha = float(cc.alpha)
        risk_total = 1.0 - self.cc_alpha
        if self.horizon > 0:
            risk_per = risk_total / float(self.horizon)
            self.k_alpha = float(norm.ppf(1.0 - risk_per))
        else:
            self.k_alpha = 0.0

        # --- Covariance propagation config ----------------------------------
        self.Sigma_w = np.diag(np.array(cc.Sigma_w_diag, dtype=float))
        self.P0 = self.Sigma_w.copy()
        self.N_robust = int(cc.N_robust)

        # Warm start
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)

        # Timing & solver tracking
        self.total_times: list[float] = []
        self.wall_times: list[float] = []
        self.solve_failures: int = 0
        self.solver_statuses: list[str] = []

        # BSK messages
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        logger.info(
            f"[L_SMPC] Initialized. horizon={horizon}, mpc_dt={mpc_dt}s, "
            f"cc_alpha={self.cc_alpha}, k_alpha={self.k_alpha:.3f}, "
            f"N_robust={self.N_robust}")

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

        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self.total_times = []
        self.wall_times = []
        self.solve_failures = 0
        self.solver_statuses = []

        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega_BN_B = np.array(nav_msg.omega_BN_B)

        q_curr = rbk.MRP2EP(sigma_BN)
        x_curr = np.concatenate([q_curr, omega_BN_B])

        ref_msg = self.attRefInMsg()
        sigma_RN = np.array(ref_msg.sigma_RN)
        q_goal = rbk.MRP2EP(sigma_RN)
        omega_goal = np.zeros(3)
        x_goal = np.concatenate([q_goal, omega_goal])

        torque = self._solve(x_curr, x_goal)

        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = torque.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

    # ======================================================================
    #  Main solve
    # ======================================================================

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        t_start = time.perf_counter()
        solve_time = 0.0

        try:
            x_nom, A_seq, B_seq, d_seq = self._linearize_trajectory(x_curr)
            P_seq = self._linear_covariance_propagation(A_seq)

            tighten = None
            if self._use_koz:
                tighten = self._compute_tightening(x_nom, P_seq)

            H_seq, c_seq, H_N, c_N = self._compute_error_linearization(
                x_nom, x_goal)

            x_opt, solve_time = self._build_and_solve(
                x_curr, A_seq, B_seq, d_seq,
                H_seq, c_seq, H_N, c_N, tighten)

            N, nx, nu = self.horizon, self.nx, self.nu
            X_flat = x_opt[:nx * (N + 1)]
            U_flat = x_opt[nx * (N + 1):nx * (N + 1) + nu * N]

            X_sol = X_flat.reshape(N + 1, nx).T
            U_sol = U_flat.reshape(N, nu).T

            self._prev_x_sol = X_sol
            self._prev_u_sol = np.clip(U_sol, self.u_min, self.u_max)
            self._u_curr = self._prev_u_sol[:, 0]
            status = 'solved'

        except Exception as e:
            self.solve_failures += 1
            self._fallback_control(e)
            status = 'exception'

        t_end = time.perf_counter()
        self.wall_times.append(solve_time)
        self.total_times.append(t_end - t_start)
        self.solver_statuses.append(status)

        return self._u_curr.copy()

    def _fallback_control(self, error: Exception) -> None:
        if self._prev_u_sol is not None:
            self._u_curr = self._prev_u_sol[
                :, min(1, self._prev_u_sol.shape[1] - 1)]
            self._prev_u_sol = np.column_stack([
                self._prev_u_sol[:, 1:], self._prev_u_sol[:, -1]])
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] Solve failed: {error}. Using shifted prev.")
        else:
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] Solve failed: {error}. No prev, zero torque.")

    # ======================================================================
    #  SOCP construction & solve (direct Clarabel)
    # ======================================================================

    def _build_and_solve(
        self,
        x_curr: np.ndarray,
        A_seq: np.ndarray,
        B_seq: np.ndarray,
        d_seq: np.ndarray,
        H_seq: np.ndarray,
        c_seq: np.ndarray,
        H_N: np.ndarray,
        c_N: np.ndarray,
        tighten: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Build sparse SOCP and solve with Clarabel.

        Decision variables:  [X_0..X_N | U_0..U_{N-1} | S_0..S_{N-1}]
        where X_k ∈ R^nx, U_k ∈ R^nu, S_k ∈ R^n_lm (KOZ slack).
        """
        N = self.horizon
        nx, nu, ne = self.nx, self.nu, self.ne
        n_lm = self._n_lm

        n_X = nx * (N + 1)
        n_U = nu * N
        n_S = n_lm * N
        n_var = n_X + n_U + n_S

        # =============================================================
        # Cost: min 0.5 x^T P x + q^T x
        # =============================================================
        P_trips = []   # (row, col, val) upper-triangular
        q_vec = np.zeros(n_var)

        for k in range(N):
            x_off = k * nx
            u_off = n_X + k * nu

            # e_k^T Q e_k  where e_k = H_k x_k + c_k
            HtQH = H_seq[k].T @ np.diag(self.Q_diag) @ H_seq[k]
            ctQH = c_seq[k] @ np.diag(self.Q_diag) @ H_seq[k]
            self._add_sym_block(P_trips, x_off, HtQH, nx)
            q_vec[x_off:x_off + nx] += 2.0 * ctQH

            # u_k^T R u_k
            for i in range(nu):
                P_trips.append((u_off + i, u_off + i, 2.0 * self.R_diag[i]))

            # du penalty
            if self.du_weight > 0:
                w = self.du_weight
                for i in range(nu):
                    P_trips.append((u_off + i, u_off + i, 2.0 * w))
                if k == 0:
                    q_vec[u_off:u_off + nu] += -2.0 * w * self._u_curr
                else:
                    u_off_prev = n_X + (k - 1) * nu
                    for i in range(nu):
                        P_trips.append((u_off_prev + i, u_off_prev + i, 2.0 * w))
                        r, c = sorted([u_off_prev + i, u_off + i])
                        P_trips.append((r, c, -2.0 * w))

        # Terminal cost
        x_off_N = N * nx
        HtFH = H_N.T @ np.diag(self.F_diag) @ H_N
        ctFH = c_N @ np.diag(self.F_diag) @ H_N
        self._add_sym_block(P_trips, x_off_N, HtFH, nx)
        q_vec[x_off_N:x_off_N + nx] += 2.0 * ctFH

        # Slack penalty: rho * ||S||^2
        if self._use_koz:
            for idx in range(n_S):
                P_trips.append((n_X + n_U + idx, n_X + n_U + idx, 2.0 * self.rho))

        P_rows, P_cols, P_vals = zip(*P_trips) if P_trips else ([], [], [])
        P_sparse = spa.csc_matrix(
            (list(P_vals), (list(P_rows), list(P_cols))), shape=(n_var, n_var))

        # =============================================================
        # Equality constraints: X_0 = x_curr, dynamics
        # =============================================================
        n_eq = nx * (N + 1)
        eq_trips = []
        b_eq = np.zeros(n_eq)

        for i in range(nx):
            eq_trips.append((i, i, 1.0))
        b_eq[:nx] = x_curr

        for k in range(N):
            rb = (k + 1) * nx
            x_k = k * nx
            x_k1 = (k + 1) * nx
            u_k = n_X + k * nu

            for i in range(nx):
                eq_trips.append((rb + i, x_k1 + i, 1.0))

            A_k = A_seq[k]
            for i in range(nx):
                for j in range(nx):
                    if abs(A_k[i, j]) > 1e-15:
                        eq_trips.append((rb + i, x_k + j, -A_k[i, j]))

            B_k = B_seq[k]
            for i in range(nx):
                for j in range(nu):
                    if abs(B_k[i, j]) > 1e-15:
                        eq_trips.append((rb + i, u_k + j, -B_k[i, j]))

            b_eq[rb:rb + nx] = d_seq[k]

        er, ec, ev = zip(*eq_trips) if eq_trips else ([], [], [])
        A_eq = spa.csc_matrix(
            (list(ev), (list(er), list(ec))), shape=(n_eq, n_var))

        # =============================================================
        # Inequality: torque bounds + slack >= 0
        # =============================================================
        n_ineq = 2 * nu * N + n_S
        ineq_trips = []
        b_ineq = np.zeros(n_ineq)
        row = 0

        for k in range(N):
            u_off = n_X + k * nu
            for i in range(nu):
                ineq_trips.append((row, u_off + i, 1.0))
                b_ineq[row] = self.u_max
                row += 1
            for i in range(nu):
                ineq_trips.append((row, u_off + i, -1.0))
                b_ineq[row] = -self.u_min
                row += 1

        if self._use_koz:
            for idx in range(n_S):
                ineq_trips.append((row, n_X + n_U + idx, -1.0))
                row += 1

        ir, ic, iv = zip(*ineq_trips) if ineq_trips else ([], [], [])
        A_ineq = spa.csc_matrix(
            (list(iv), (list(ir), list(ic))), shape=(n_ineq, n_var))

        # =============================================================
        # SOC constraints for KOZ
        #
        # q^T M q <= rhs  where rhs = 2 - tighten + slack, M = L L^T
        # ⟺  ||L^T q||^2 <= rhs
        # ⟺  ||(2 L^T q,  rhs - 1)||_2 <= rhs + 1     [standard trick]
        #
        # Clarabel SOC: s = b - A x,  s ∈ SOC  means s[0] >= ||s[1:]||
        # Cone dim = 6:  [rhs+1,  2 L^T q,  rhs-1]
        # =============================================================
        soc_sizes = []
        soc_trips = []
        b_soc_list = []

        if self._use_koz and tighten is not None:
            soc_row = 0
            for k in range(N):
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm = l * self.num_koz + m
                        LT = self._koz_chol[l, m].T
                        t_val = tighten[lm, k]
                        s_idx = n_X + n_U + k * n_lm + lm
                        q_off = (k + 1) * nx

                        # Row 0 (t-component): s_0 = (3 - t_val) + S
                        soc_trips.append((soc_row, s_idx, -1.0))
                        b_soc_list.append(3.0 - t_val)
                        soc_row += 1

                        # Rows 1-4: s_i = 2 L^T[i,:] @ q
                        for i in range(4):
                            for j in range(4):
                                if abs(LT[i, j]) > 1e-15:
                                    soc_trips.append((soc_row, q_off + j, -2.0 * LT[i, j]))
                            b_soc_list.append(0.0)
                            soc_row += 1

                        # Row 5: s_5 = (1 - t_val) + S
                        soc_trips.append((soc_row, s_idx, -1.0))
                        b_soc_list.append(1.0 - t_val)
                        soc_row += 1

                        soc_sizes.append(6)

            n_soc = soc_row
        else:
            n_soc = 0

        # =============================================================
        # Assemble & solve
        # =============================================================
        parts_A = [A_eq, A_ineq]
        parts_b = [b_eq, b_ineq]

        if n_soc > 0:
            sr, sc_arr, sv = zip(*soc_trips)
            A_soc = spa.csc_matrix(
                (list(sv), (list(sr), list(sc_arr))), shape=(n_soc, n_var))
            parts_A.append(A_soc)
            parts_b.append(np.array(b_soc_list))

        A_all = spa.vstack(parts_A, format='csc')
        b_all = np.concatenate(parts_b)

        cones = [
            clarabel.ZeroConeT(n_eq),
            clarabel.NonnegativeConeT(n_ineq),
        ]
        for sz in soc_sizes:
            cones.append(clarabel.SecondOrderConeT(sz))

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.max_iter
        settings.tol_gap_abs = 1e-4
        settings.tol_gap_rel = 1e-4
        settings.tol_feas = 1e-4

        solver = clarabel.DefaultSolver(P_sparse, q_vec, A_all, b_all, cones, settings)
        result = solver.solve()

        if result.status not in (clarabel.SolverStatus.Solved,
                                 clarabel.SolverStatus.AlmostSolved):
            raise RuntimeError(f"Clarabel: {result.status}")

        return np.array(result.x), result.solve_time

    @staticmethod
    def _add_sym_block(trips: list, offset: int, M: np.ndarray, n: int):
        """Add upper-triangular entries of 2*M to triplet list (for 0.5 x^T P x)."""
        for i in range(n):
            for j in range(i, n):
                val = 2.0 * M[i, j]
                if abs(val) > 1e-15:
                    trips.append((offset + i, offset + j, val))

    # ======================================================================
    #  Linearization along nominal trajectory
    # ======================================================================

    def _linearize_trajectory(self, x_curr: np.ndarray):
        """Propagate nominal trajectory and compute Jacobians via RK4."""
        N = self.horizon
        x_nom = np.zeros((N + 1, self.nx))
        A_seq = np.zeros((N, self.nx, self.nx))
        B_seq = np.zeros((N, self.nx, self.nu))
        d_seq = np.zeros((N, self.nx))

        x_nom[0] = x_curr.copy()

        if self._prev_u_sol is not None:
            u_nom = np.zeros((N, self.nu))
            u_nom[:-1] = self._prev_u_sol[:, 1:].T
            u_nom[-1] = self._prev_u_sol[:, -1]
        else:
            u_nom = np.zeros((N, self.nu))

        for k in range(N):
            x_next, A_k, B_k = attitude_dyn_dt_jac(
                x_nom[k], u_nom[k], self.mpc_dt, self.J, self.J_inv)
            x_nom[k + 1] = x_next
            A_seq[k] = A_k
            B_seq[k] = B_k
            d_seq[k] = x_next - A_k @ x_nom[k] - B_k @ u_nom[k]

        return x_nom, A_seq, B_seq, d_seq

    # ======================================================================
    #  Linear covariance propagation
    # ======================================================================

    def _linear_covariance_propagation(self, A_seq: np.ndarray) -> np.ndarray:
        """P_{k+1} = A_k P_k A_k^T + Sigma_w, frozen after N_robust steps."""
        N = A_seq.shape[0]
        N_rob = min(self.N_robust, N)

        P_seq = np.zeros((N + 1, self.nx, self.nx))
        P_seq[0] = self.P0.copy()

        for k in range(N):
            if k < N_rob:
                A_k = A_seq[k]
                P_new = A_k @ P_seq[k] @ A_k.T + self.Sigma_w
                P_seq[k + 1] = 0.5 * (P_new + P_new.T)
            else:
                P_seq[k + 1] = P_seq[N_rob].copy()

        return P_seq

    # ======================================================================
    #  KOZ chance-constraint tightening
    # ======================================================================

    def _compute_tightening(self, x_nom: np.ndarray, P_seq: np.ndarray) -> np.ndarray:
        """Compute KOZ tightening: tighten = k_alpha * sqrt(a^T P_qq a)
        where a = 2 M_tilde q_nom."""
        N = self.horizon
        n_lm = self._n_lm
        tighten = np.zeros((n_lm, N))

        for k in range(N):
            P_qq = P_seq[k + 1, :4, :4]
            q_k = x_nom[k + 1, :4]

            for l in range(self.num_bs):
                for m in range(self.num_koz):
                    lm = l * self.num_koz + m
                    a_q = 2.0 * self.koz_matrices[l, m] @ q_k
                    sigma2 = a_q @ P_qq @ a_q
                    tighten[lm, k] = self.k_alpha * np.sqrt(max(sigma2, 0.0))

        return tighten

    # ======================================================================
    #  Error linearization
    # ======================================================================

    def _compute_error_linearization(
        self, x_nom: np.ndarray, x_goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linearize 6D error: e_k = [quat_error; dw] ≈ H_k x_k + c_k."""
        N = self.horizon
        ne, nx = self.ne, self.nx
        H_seq = np.zeros((N, ne, nx))
        c_seq = np.zeros((N, ne))

        q_goal, w_goal = x_goal[:4], x_goal[4:]

        for k in range(N):
            H_seq[k], c_seq[k] = self._error_linearize_at(x_nom[k], q_goal, w_goal)

        H_N, c_N = self._error_linearize_at(x_nom[N], q_goal, w_goal)
        return H_seq, c_seq, H_N, c_N

    def _error_linearize_at(self, x_nom_k, q_goal, w_goal):
        q_nom, w_nom = x_nom_k[:4], x_nom_k[4:]
        e_nom = np.concatenate([quat_error(q_nom, q_goal), w_nom - w_goal])

        H = np.zeros((6, 7), dtype=np.float64)
        H[:3, :4] = _quat_error_jacobian(q_nom, q_goal)
        H[3:, 4:] = np.eye(3)

        return H, e_nom - H @ x_nom_k

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
