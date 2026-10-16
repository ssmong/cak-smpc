import gc
import os
import time
import logging
from collections import deque
import numpy as np
import torch
import clarabel
import scipy.sparse as spa
from scipy.stats import chi2 as _chi2_dist
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging, bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros

from sim.koz import LinearKOZ, BoresightHandler
from src.model.modules import get_latent_QF, get_latent_QF_batch
from src.utils import load_party_interface
from src.utils.math_numpy import slerp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PartySMPC")



class PartySMPC(sysModel.SysModel):
    def __init__(self,
                 model_dir: str,
                 horizon: int,
                 mpc_dt: float,
                 ctrl_dt: float,
                 Q_diag: list[float],
                 R_diag: list[float],
                 F_diag: list[float],
                 rho: float,
                 max_iter: int,
                 x_init: np.ndarray,
                 cc: DictConfig,
                 w_adaptive: DictConfig,
                 constraint_cfg: DictConfig,
                 qp_solver: str = "osqp",
                 record_linearization_error: bool = False,
                 lin_margin_file: str | None = None,
                 lin_margin_key: str = "margin_max",
                 scp_relinearize: DictConfig | None = None,
                 du_weight: float = 0.0,
                 du_max: float | None = None,
                 use_incremental_ctx: bool = False,
                 use_custom_mamba: bool = False,
                 **kwargs):
        """
        Args:
            model_dir: Path to trained model directory (relative to outputs/learning/)
            horizon: MPC prediction horizon steps
            ctrl_dt: Control loop timestep [s] - must divide model's hist_dt evenly
            Q_diag: State cost diagonal [latent_dim]
            R_diag: Control cost diagonal [3]
            F_diag: Terminal state cost diagonal [latent_dim]
            rho: Soft constraint penalty weight for KOZ violation
            max_iter: Maximum IPOPT iterations
            x_init: Initial state [q0, q1, q2, q3, w1, w2, w3]
            cc: Chance constraint configuration
            w_adaptive: Adaptive learning configuration for the covariance matrix W
            constraint_cfg: Constraint configuration (torque bounds, KOZ list, boresight)
            record_linearization_error: If True, record linearization error at each MPC solve.
            lin_margin_file: Path to YAML with empirical linearization margin. None = no margin.
            lin_margin_key: Key within the margin YAML to use (e.g. "margin_max", "margin_p95").
        """
        super(PartySMPC, self).__init__()
        self.ModelTag = "PartySMPC"
        
        # Load model and extract config (hist_dt, hist_len from training)
        self.model, model_cfg = load_party_interface(model_dir, use_custom_mamba=use_custom_mamba)

        # Time parameters: mpc_dt from config, validated against model's pred_dt
        self.mpc_dt = mpc_dt
        model_pred_dt = model_cfg.pred_dt
        if not np.isclose(self.mpc_dt, model_pred_dt, rtol=1e-6):
            raise ValueError(
                f"Config mpc_dt ({self.mpc_dt}s) does not match model pred_dt ({model_pred_dt}s). "
                f"They must be identical."
            )
        self.hist_dt = model_cfg.hist_dt     # History sampling interval
        self.hist_len = model_cfg.hist_len   # History length
        self.ctrl_dt = ctrl_dt               # Control loop timestep
        
        # ctrl_dt can be <= hist_dt; history is subsampled at hist_dt intervals
        self.hist_sample_interval = int(round(self.hist_dt / self.ctrl_dt))
        if not np.isclose(self.hist_sample_interval * self.ctrl_dt, self.hist_dt, rtol=1e-6):
            raise ValueError(
                f"hist_dt ({self.hist_dt}s) must be an integer multiple of ctrl_dt ({self.ctrl_dt}s). "
                f"Got ratio: {self.hist_dt / self.ctrl_dt}"
            )

        # Compute MPC call interval (how many ctrl_dt steps per mpc_dt)
        self.mpc_call_interval = int(round(self.mpc_dt / self.ctrl_dt))
        if not np.isclose(self.mpc_call_interval * self.ctrl_dt, self.mpc_dt, rtol=1e-6):
            raise ValueError(
                f"mpc_dt ({self.mpc_dt}s) must be an integer multiple of ctrl_dt ({self.ctrl_dt}s). "
                f"Got ratio: {self.mpc_dt / self.ctrl_dt}"
            )

        # Warmup time: need hist_len * hist_dt seconds to accumulate history
        self.warmup_time = self.hist_len * self.hist_dt
        
        self.horizon = horizon
        self.rho = rho
        self.max_iter = max_iter
        self.qp_solver = qp_solver.lower()
        if self.qp_solver not in ("osqp", "clarabel"):
            raise ValueError(f"qp_solver must be 'osqp' or 'clarabel', got '{qp_solver}'")
        # Eigenvalue floor for Q_latent / F_latent PSD projection.
        # Higher value (e.g. 1e-2) keeps cond(P) manageable for ADMM (OSQP).
        # Interior-point solvers (Clarabel) tolerate lower floors (e.g. 1e-4, 1e-6).
        self._eig_floor_ratio = float(kwargs.get("eig_floor_ratio", 1e-2))

        self._prev_step_data = None
        self.enable_w_update = w_adaptive.enable_w_update
        self.w_gamma = w_adaptive.gamma
        self.w_warmup_steps = w_adaptive.warmup_steps
        self.w_s_max = w_adaptive.s_max
        self.w_inflate_alpha = float(w_adaptive.inflate_alpha)

        self._W_history: list[np.ndarray] = []
        self._z_history: list[np.ndarray] = []
        self._ctx_history: list = []  # diagnostic only
        self._v_history: list[np.ndarray] = []   # v (MPC decision) at each MPC solve
        self._z_goal_history: list[np.ndarray] = []  # z_goal at each MPC solve
        self._z_cost_Q_history: list[float] = []  # ||z_err||_{Q_latent}^2 at each MPC solve
        self._s_curr = 1.0          # Online scalar scale for W_offline
        self._s_history: list[float] = []  # s_final at each MPC solve (for diagnostics)
        self._w_sample_count = 0

        # Model dimensions
        self.nx = self.model.state_dim      # State dimension (7: q + omega)
        self.nz = self.model.latent_dim     # Latent dimension
        self.nu = self.model.control_dim    # Control dimension (3)

        w_path = os.path.join("outputs/learning_party", model_dir, "stats", "W_offline.npy")
        if os.path.exists(w_path):
            self.W_offline = np.load(w_path).astype(np.float64)
            logger.info(f"Loaded W_offline from {w_path}")
        else:
            logger.error(f"W_offline not found at {w_path}.")
            raise FileNotFoundError(f"W_offline not found at {w_path}.")

        # Precompute Cholesky of W_offline for Mahalanobis distance
        self._W_offline_chol = np.linalg.cholesky(self.W_offline)
        # χ²(nz) percentile threshold: s_inst > 1 only when chi2 exceeds α-quantile
        self._chi2_thresh = _chi2_dist.ppf(self.w_inflate_alpha, self.nz) / self.nz

        # Linearization error recording
        self.record_lin_error = record_linearization_error
        self._lin_errors: list[np.ndarray] = []
        self._lin_devs: list[float] = []  # ||z_opt - z_nom||_inf per MPC solve

        # Delta-u penalty: du_weight * ||u_k - u_{k-1}||^2 (0 = disabled)
        self.du_weight = float(du_weight)
        # Hard trust region on first control step: |u_0 - u_prev| ≤ du_max
        self.du_max = float(du_max) if du_max is not None else None

        # Incremental context encoding: RNN-like single-step Mamba update
        self.use_incremental_ctx = use_incremental_ctx
        self._new_hist_count = 0
        self._ctx_inference_params = None

        # SCP re-linearization (single extra iteration when solution deviates from nominal)
        if scp_relinearize is not None and scp_relinearize.enabled:
            self._scp_enabled = True
            self._scp_threshold = float(scp_relinearize.threshold)
        else:
            self._scp_enabled = False
            self._scp_threshold = 0.0

        # Constraint handlers
        self.bs_handler = BoresightHandler(boresight_cfg=constraint_cfg.boresight_list, model=self.model)
        self.koz_handler = LinearKOZ(
            koz_list=constraint_cfg.koz_list,
            cc_cfg=cc,
            horizon=horizon,
            margin_file=lin_margin_file,
            margin_key=lin_margin_key,
            w_beta=float(cc.w_beta),
        )
        self.num_bs = self.bs_handler.num_bs
        self.num_koz = self.koz_handler.num_koz

        # Torque bounds
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min

        # Cost matrices (will be computed in latent space)
        self.Q_diag = Q_diag
        self.R_diag = R_diag
        self.F_diag = F_diag

        # Quaternion continuity: training data uses unwrapped quaternions
        # (dot(q_t, q_{t-1}) >= 0), but Basilisk MRP2EP always gives q0 >= 0.
        # We must unwrap to match the convention the Koopman model was trained on.
        self._q_prev = x_init[:4].copy()

        # History buffers (deques for efficient FIFO)
        # hist_x: [x_{t-H}, ..., x_t] -> H+1 states
        # hist_u: [u_{t-H}, ..., u_{t-1}] -> H controls
        self._hist_x: deque[np.ndarray] = deque(maxlen=self.hist_len + 1)
        self._hist_u: deque[np.ndarray] = deque(maxlen=self.hist_len)

        # Initialize history with x_init
        for _ in range(self.hist_len + 1):
            self._hist_x.append(x_init.copy())
        for _ in range(self.hist_len):
            self._hist_u.append(np.zeros(self.nu))

        # Solver state
        self._prev_z_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)

        # Inter-solve state
        self._z_opt = None              # (nz, N+1) latest MPC optimal trajectory
        self._v_opt = None              # (nu, N)   latest MPC optimal inputs

        self._solve_time_offset = 0     # ctrl_dt steps since last MPC solve

        # Cached z_goal: fixed once determined, prevents oscillation from q-sign flipping
        self._z_goal_cached = None
        self._q_goal_raw_cached = None  # MRP2EP result (before unwrap) to detect reference change
        self._x_goal_for_ws = None      # x_goal with matching q sign for SLERP warm-start

        self._z_err_norm_history: list[float] = []
        self._z_pred_err_history: list[float] = []  # ||z_curr - z_predicted|| per MPC solve (one-step prediction error)
        self._q_goal_err_deg_history: list[float] = []  # geodesic angle to q_goal [deg] per MPC solve
        self._v_plan_history: list[np.ndarray] = []  # u_sol (nu, N) at each MPC solve
        self._A_diff_history: list[float] = []  # ||A_k - A_{k-1}||_F
        self._A_rel_diff_history: list[float] = []  # ||A_k - A_{k-1}||_F / ||A_k||_F
        self._A_norm_history: list[float] = []  # ||A_k||_F
        self._B_diff_history: list[float] = []  # ||B_k - B_{k-1}||_F
        self._ctx_diff_history: list[float] = []  # ||ctx_k - ctx_{k-1}||
        self._u_diff_history: list[float] = []  # ||v_k[0] - v_{k-1}[0]||  (consecutive MPC first-step control change)
        self._B_sv_history: list[tuple[float, float, float]] = []  # (σ_min, σ_max, cond) of B per solve
        self._A_eig_most_neg_history: list[tuple[float, float, float]] = []  # (real, imag, abs) of most-negative-real eigenvalue

        # Timing
        self._ctrl_step_count = 0
        self.wall_times: list[float] = []
        self.total_times: list[float] = []
        self.solve_failures: int = 0
        self._is_warmup = True
        
        # Cached A, B matrices (updated at mpc_dt intervals)
        self._A_cached: np.ndarray | None = None
        self._B_cached: np.ndarray | None = None
        self._ctx_cached: np.ndarray | None = None

        # Input messages
        # - attNavInMsg: absolute spacecraft attitude/rate from simpleNav (B/N, w_BN_B)
        # - attRefInMsg: reference attitude from guidance (R/N)
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        self.vehConfigInMsg = messaging.VehicleConfigMsgReader()  # For compatibility with run_sim.py
        
        # Output messages
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        
        logger.info(f"[PartySMPC] Initialized. horizon={horizon}, mpc_dt={self.mpc_dt}s, "
                    f"ctrl_dt={ctrl_dt}s, hist_len={self.hist_len}, warmup={self.warmup_time}s, "
                    f"qp_solver={self.qp_solver}")

    def Reset(self, CurrentSimNanos: int):
        """
        Initialize the PartySMPC controller.
        Called once at simulation start.
        
        Args:
            CurrentSimNanos: Current simulation time [ns]
        """
        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, 
            f"[{self.ModelTag}] Resetting: mpc_dt={self.mpc_dt}s, ctrl_dt={self.ctrl_dt}s, "
            f"hist_len={self.hist_len}, warmup_time={self.warmup_time}s")
        
        # Validate message connections
        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")
        
        # Build OCP
        self._setup_qp()

        # Warmup PyTorch vmap(jacrev) — must NOT use torch.no_grad() since jacrev needs grad
        z_dummy = torch.randn(self.horizon + 1, self.nz,
                              device=self.model.device, dtype=self.model.dtype)
        get_latent_QF_batch(self.model, z_dummy, self.Q_diag, self.F_diag,
                            eig_floor_ratio=self._eig_floor_ratio)

        # Reset state
        self._ctrl_step_count = 0
        self._is_warmup = True
        self._prev_z_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self.wall_times = []
        self.solve_failures = 0
        self._A_cached = None
        self._B_cached = None
        self._ctx_cached = None

        # Incremental context state
        self._ctx_inference_params = None
        self._x_prev_for_ctx: torch.Tensor | None = None
        self._profile_ctx_step_ms: float | None = None
        self._ctx_step_time_history: list[float] = []  # ms per incremental step
        self._new_hist_count = 0  # new history tokens since last MPC solve

        self._s_curr = 1.0
        self._s_history = []
        self._w_sample_count = 0
        self._prev_step_data = None
        self._lin_errors = []
        self._lin_devs = []
        self._v_history = []
        self._z_goal_history = []
        self._z_cost_Q_history = []
        self._z_err_norm_history = []
        self._z_pred_err_history = []
        self._q_goal_err_deg_history = []
        self._v_plan_history = []
        self._A_diff_history = []
        self._A_rel_diff_history = []
        self._A_norm_history = []
        self._B_diff_history = []
        self._ctx_diff_history = []
        self._u_diff_history = []
        self._B_sv_history = []
        self._A_eig_most_neg_history = []
        self._tighten_history: list[np.ndarray] = []  # per-solve (N,) CC tightening in cosine units
        # Inter-solve state
        self._z_opt = None
        self._v_opt = None
        self._solve_time_offset = 0

        # Write zero torque initially
        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)
        
        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, 
            f"[{self.ModelTag}] OCP setup complete. num_bs={self.num_bs}, num_koz={self.num_koz}")

    def UpdateState(self, CurrentSimNanos: int):
        """
        Main control loop - called at ctrl_dt intervals.
        
        Args:
            CurrentSimNanos: Current simulation time [ns]
        """
        current_time = CurrentSimNanos * macros.NANO2SEC
        
        # Read absolute navigation attitude state (not relative attitude error)
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega_BN_B = np.array(nav_msg.omega_BN_B)
        
        # Convert MRP to quaternion and unwrap for continuity.
        # Basilisk MRP2EP always returns q0 >= 0 (MRP shadow switching),
        # but the Koopman model was trained on continuous (unwrapped) quaternions
        # where q0 can be negative. Flip sign when dot(q_curr, q_prev) < 0.
        q_curr = rbk.MRP2EP(sigma_BN)
        if np.dot(q_curr, self._q_prev) < 0:
            q_curr = -q_curr
        self._q_prev = q_curr.copy()
        x_curr = np.concatenate([q_curr, omega_BN_B])
        
        # Update history buffers at hist_dt intervals (subsample when ctrl_dt < hist_dt)
        if self._ctrl_step_count % self.hist_sample_interval == 0:
            self._hist_x.append(x_curr.copy())
            self._hist_u.append(self._u_curr.copy())
            if self.use_incremental_ctx:
                self._new_hist_count += 1
        
        # Check warmup status
        if self._is_warmup:
            if current_time >= self.warmup_time:
                self._is_warmup = False
                self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, 
                    f"[{self.ModelTag}] Warmup complete at t={current_time:.2f}s. Starting MPC.")
            else:
                self._ctrl_step_count += 1
                return
        
        # Check if it's time to solve MPC (every mpc_call_interval steps)
        if self._ctrl_step_count % self.mpc_call_interval == 0:
            ref_msg = self.attRefInMsg()
            sigma_RN = np.array(ref_msg.sigma_RN)
            q_goal = rbk.MRP2EP(sigma_RN)
            # Unwrap goal to match current trajectory's hemisphere
            if np.dot(q_goal, q_curr) < 0:
                q_goal = -q_goal

            omega_goal = np.zeros(3)  # Regulation assumption
            x_goal = np.concatenate([q_goal, omega_goal])

            self._u_curr = self._solve(x_curr, x_goal)
            self._solve_time_offset = 0
        else:
            # Inter-solve: ZOH (hold first-step control from last MPC solve)
            pass

        # Write output
        self._write_torque(CurrentSimNanos)
        self._ctrl_step_count += 1

    def _update_W(self, z_curr: np.ndarray) -> np.ndarray:
        """Running-max with geometric decay: scalar covariance adaptation.

        Maintains s >= 1 such that W = s * W_offline.  Responds to spikes in
        a single step (via max) and decays geometrically toward 1.0 when
        prediction errors are nominal.

        DRO interpretation: s_t is the radius of a moment-based ambiguity set
        {s * W_offline : 1 <= s <= s_t}.  The raw chi2/nz statistic is
        normalized by χ²_α(nz)/nz so that s_inst = 1.0 corresponds to
        the inflate_alpha quantile; nominal errors trigger inflation only
        (1 - inflate_alpha) fraction of the time.
        """
        if not self.enable_w_update or self._prev_step_data is None:
            self._s_history.append(1.0)
            return self.W_offline.copy()
        z_prev, u_prev, A_prev, B_prev = self._prev_step_data

        z_pred = A_prev @ z_prev + B_prev @ u_prev
        err = z_curr - z_pred  # (nz,)

        # Mahalanobis statistic: v = L^{-1} e, chi2 = v·v ~ χ²(nz) under nominal
        v = np.linalg.solve(self._W_offline_chol, err)
        chi2_stat = float(v @ v)
        # Normalize so that s_inst = 1.0 at the α-quantile of χ²(nz)
        s_inst = (chi2_stat / self.nz) / self._chi2_thresh

        # Running-max with geometric decay toward 1.0
        s_decayed = self.w_gamma * self._s_curr + (1.0 - self.w_gamma)
        self._s_curr = max(1.0, s_inst, s_decayed)
        self._w_sample_count += 1

        # Warmup ramp: gradually trust online estimate
        ramp = min(self._w_sample_count, self.w_warmup_steps) / self.w_warmup_steps

        # Ceiling cap to prevent extreme inflation
        s_capped = min(self._s_curr, self.w_s_max)

        # Final: interpolate from 1.0 (pure offline) toward s_capped
        s_final = 1.0 + ramp * (s_capped - 1.0)
        self._s_history.append(s_final)

        return s_final * self.W_offline

    def _write_torque(self, CurrentSimNanos: int):
        """Write current torque command to output message."""
        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = self._u_curr.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)

    def _setup_qp(self) -> None:
        """Pre-compute QP dimensions and fixed sparse structure.

        Caches all sparsity patterns that do not change between solves,
        so _solve_qp only fills in numerical values.
        """
        self._use_koz = self.num_koz > 0 and self.num_bs > 0
        if not self._use_koz:
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] num_koz={self.num_koz}, num_bs={self.num_bs}: "
                f"KOZ constraints disabled.")

        N, nz, nu = self.horizon, self.nz, self.nu
        n_lm = self.num_bs * self.num_koz if self._use_koz else 0
        self._qp_n_lm = n_lm
        self._qp_nZ = nz * (N + 1)
        self._qp_nV = nu * N
        self._qp_nS = n_lm * N
        self._qp_n = self._qp_nZ + self._qp_nV + self._qp_nS
        self._qp_solver = None  # created on first solve

        n = self._qp_n
        nZ, nV, nS = self._qp_nZ, self._qp_nV, self._qp_nS
        I_nz = spa.eye(nz, format='csc')
        I_N = spa.eye(N, format='csc')

        # --- Fixed constraint matrices ---
        # C1: z_0 = z_curr  (nz equalities)
        self._C1 = spa.hstack([I_nz, spa.csc_matrix((nz, n - nz))], format='csc')

        # C3: control bounds selector (nV rows)
        self._C3 = spa.hstack([spa.csc_matrix((nV, nZ)),
                               spa.eye(nV, format='csc'),
                               spa.csc_matrix((nV, nS))], format='csc')

        # KOZ constraint index arrays (vectorized)
        if self._use_koz:
            # Pre-compute row, col indices for C4_z
            j_arr = np.arange(n_lm)
            k_arr = np.arange(N)
            jj, kk = np.meshgrid(j_arr, k_arr, indexing='ij')  # (n_lm, N)
            jj_flat = jj.ravel()  # (n_lm*N,)
            kk_flat = kk.ravel()
            # row index: j * N + k
            self._koz_row_base = (jj_flat * N + kk_flat)  # (n_lm*N,)
            # col offset: (k+1) * nz
            self._koz_col_offset = (kk_flat + 1) * nz  # (n_lm*N,)
            # For each (j,k) pair, columns span col_offset .. col_offset+nz-1
            # Expand to per-element indices: (n_lm*N*nz,)
            self._koz_rows = np.repeat(self._koz_row_base, nz)
            col_offsets_expanded = np.repeat(self._koz_col_offset, nz)
            nz_range = np.tile(np.arange(nz), n_lm * N)
            self._koz_cols = col_offsets_expanded + nz_range
            # a_k_ca index for value lookup: (j*N+k, i) → row j*N+k in a_k_ca
            self._koz_akca_rows = np.repeat(jj_flat * self.horizon + kk_flat, nz)
            self._koz_akca_cols = nz_range

            # C5: slack non-negativity (fixed)
            self._C5 = spa.hstack([spa.csc_matrix((nS, nZ + nV)),
                                   spa.eye(nS, format='csc')], format='csc')

            # rhs_ca index arrays
            self._rhs_j_flat = jj_flat
            self._rhs_k_flat = kk_flat

        # Delta-u off-diagonal pattern (fixed if N > 1)
        if N > 1:
            du_rows = []
            du_cols = []
            for k in range(N - 1):
                r0 = nZ + k * nu
                r1 = nZ + (k + 1) * nu
                for i in range(nu):
                    du_rows.append(r0 + i)
                    du_cols.append(r1 + i)
            self._du_off_rows = np.array(du_rows)
            self._du_off_cols = np.array(du_cols)
        else:
            self._du_off_rows = np.array([], dtype=int)
            self._du_off_cols = np.array([], dtype=int)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION,
            f"[{self.ModelTag}] QP setup complete. n_var={n}, "
            f"num_koz={self.num_koz}, num_bs={self.num_bs}")

    def _solve_qp(self, A_sys: np.ndarray, B_sys: np.ndarray,
                   Q_lat_seq: np.ndarray, F_lat: np.ndarray, R_lat: np.ndarray,
                   z_curr: np.ndarray, z_goal: np.ndarray,
                   z_warm: np.ndarray, u_warm: np.ndarray,
                   u_prev: np.ndarray | None = None,
                   a_k_ca: np.ndarray | None = None,
                   rhs_ca: np.ndarray | None = None):
        """Build and solve the MPC QP using pre-cached sparsity patterns.

        Variables: x = [z_0; ...; z_N; u_0; ...; u_{N-1}; s_flat]

        Args:
            Q_lat_seq: (N, d, d) stage-varying latent cost matrices.

        Returns (z_sol, u_sol, wall_time_s).
        Raises RuntimeError on solve failure.
        """
        N, nz, nu = self.horizon, self.nz, self.nu
        nZ, nV, nS = self._qp_nZ, self._qp_nV, self._qp_nS
        n_lm = self._qp_n_lm
        n = self._qp_n

        # === P: cost Hessian (stage-varying Q, fixed R, F) ===
        use_du = self.du_weight > 0 and u_prev is not None and N > 0
        w2 = 2.0 * self.du_weight if use_du else 0.0

        # Build R block with du penalty baked in
        R2 = 2.0 * R_lat
        if use_du:
            R_blocks = []
            for k in range(N):
                diag_add = w2 * (2.0 if k < N - 1 else 1.0)
                R_blocks.append(spa.csc_matrix(R2 + diag_add * np.eye(nu)))
        else:
            R_block = spa.csc_matrix(R2)
            R_blocks = [R_block] * N

        P_blocks = [spa.csc_matrix(2.0 * Q_lat_seq[k]) for k in range(N)]
        P_blocks.append(spa.csc_matrix(2.0 * F_lat))
        P_blocks.extend(R_blocks)
        if nS > 0:
            P_blocks.append(2.0 * self.rho * spa.eye(nS, format='csc'))
        P_full = spa.block_diag(P_blocks, format='csc')

        # Delta-u off-diagonal (cached pattern)
        if use_du and len(self._du_off_rows) > 0:
            du_vals = np.full(len(self._du_off_rows), -w2)
            du_off = spa.csc_matrix(
                (du_vals, (self._du_off_rows, self._du_off_cols)), shape=(n, n))
            P_full = P_full + du_off + du_off.T

        P = spa.triu(P_full, format='csc')

        # === q: linear cost (vectorized) ===
        q = np.zeros(n)
        # Q_lat_seq @ z_goal for all N stages at once: (N, nz, nz) @ (nz,) → (N, nz)
        q_stages = -2.0 * (Q_lat_seq @ z_goal)  # (N, nz)
        q[:N * nz] = q_stages.ravel()
        q[N * nz:(N + 1) * nz] = -2.0 * F_lat @ z_goal

        if use_du:
            q[nZ:nZ + nu] -= 2.0 * self.du_weight * u_prev

        # === Constraint matrix (using cached C1, C3) ===
        # C2: dynamics  A z_k + B v_k - z_{k+1} = 0
        I_N = spa.eye(N, format='csc')
        A_sp = spa.csc_matrix(A_sys)
        B_sp = spa.csc_matrix(B_sys)
        I_nz = spa.eye(nz, format='csc')

        C2_z = (spa.hstack([spa.kron(I_N, A_sp, format='csc'),
                            spa.csc_matrix((nz * N, nz))], format='csc')
                + spa.hstack([spa.csc_matrix((nz * N, nz)),
                              spa.kron(I_N, -I_nz, format='csc')], format='csc'))
        C2_v = spa.kron(I_N, B_sp, format='csc')
        C2 = spa.hstack([C2_z, C2_v, spa.csc_matrix((nz * N, nS))], format='csc')

        A_eq = spa.vstack([self._C1, C2], format='csc')
        b_eq = np.concatenate([z_curr, np.zeros(nz * N)])

        # Inequalities
        b_ub = np.full(nV, self.u_max)
        b_lb = np.full(nV, -self.u_min)
        if self.du_max is not None and u_prev is not None:
            b_ub[:nu] = np.minimum(self.u_max, u_prev + self.du_max)
            b_lb[:nu] = np.minimum(-self.u_min, self.du_max - u_prev)
        A_ineq_parts = [self._C3, -self._C3]
        b_ineq_parts = [b_ub, b_lb]

        if self._use_koz and a_k_ca is not None:
            n_koz = n_lm * N
            # Vectorized KOZ constraint assembly (no Python loops)
            koz_vals = a_k_ca[self._koz_akca_rows, self._koz_akca_cols]
            C4_z = spa.csc_matrix(
                (koz_vals, (self._koz_rows, self._koz_cols)),
                shape=(n_koz, nZ))
            C4 = spa.hstack([C4_z, spa.csc_matrix((n_koz, nV)),
                             -spa.eye(nS, format='csc')], format='csc')
            b4 = rhs_ca[self._rhs_j_flat, self._rhs_k_flat]

            A_ineq_parts.extend([C4, -self._C5])
            b_ineq_parts.extend([b4, np.zeros(nS)])

        A_ineq = spa.vstack(A_ineq_parts, format='csc')
        b_ineq = np.concatenate(b_ineq_parts)

        # === Solve ===
        if self.qp_solver == "osqp":
            x_opt, wall_time = self._solve_osqp(
                P, q, A_eq, b_eq, A_ineq, b_ineq,
                z_warm, u_warm, nZ, nV, nS)
        else:
            x_opt, wall_time = self._solve_clarabel(
                P, q, A_eq, b_eq, A_ineq, b_ineq)

        z_sol = x_opt[:nZ].reshape(nz, N + 1, order='F')
        u_sol = x_opt[nZ:nZ + nV].reshape(nu, N, order='F')

        return z_sol, u_sol, wall_time

    def _solve_clarabel(self, P, q, A_eq, b_eq, A_ineq, b_ineq):
        """Solve QP with Clarabel (interior-point). Robust to mild ill-conditioning."""
        A_clar = spa.vstack([A_eq, A_ineq], format='csc')
        b_clar = np.concatenate([b_eq, b_ineq])
        cones = [clarabel.ZeroConeT(A_eq.shape[0]),
                 clarabel.NonnegativeConeT(A_ineq.shape[0])]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.max_iter
        settings.tol_gap_abs = 1e-4
        settings.tol_gap_rel = 1e-4
        settings.tol_feas = 1e-4

        solver = clarabel.DefaultSolver(P, q, A_clar, b_clar, cones, settings)
        result = solver.solve()

        if result.status not in (clarabel.SolverStatus.Solved,
                                 clarabel.SolverStatus.AlmostSolved):
            raise RuntimeError(f"Clarabel: {result.status}")

        return np.array(result.x), result.solve_time

    def _solve_osqp(self, P, q, A_eq, b_eq, A_ineq, b_ineq,
                     z_warm, u_warm, nZ, nV, nS):
        """Solve QP with OSQP (ADMM). Fast with warm-start.

        OSQP form: min 0.5 x^T P x + q^T x  s.t.  l <= A x <= u
        """
        import osqp

        A_osqp = spa.vstack([A_eq, A_ineq], format='csc')
        l = np.concatenate([b_eq, np.full(A_ineq.shape[0], -np.inf)])
        u = np.concatenate([b_eq, b_ineq])

        # Warm-start primal: shifted previous solution
        x_warm = np.concatenate([
            z_warm.flatten('F'), u_warm.flatten('F'), np.zeros(nS)
        ])

        if self._qp_solver is not None:
            # Re-use cached solver: update data only (same sparsity pattern)
            try:
                self._qp_solver.update(
                    Px=spa.triu(P).data, q=q,
                    Ax=A_osqp.data, l=l, u=u)
                self._qp_solver.warm_start(x=x_warm)
            except Exception:
                # Sparsity pattern changed (e.g. KOZ active set) → re-setup
                self._qp_solver = None

        if self._qp_solver is None:
            self._qp_solver = osqp.OSQP()
            self._qp_solver.setup(
                P, q, A_osqp, l, u,
                eps_abs=1e-4, eps_rel=1e-4,
                max_iter=self.max_iter,
                polish=True, warm_start=True,
                verbose=False, adaptive_rho=True)
            self._qp_solver.warm_start(x=x_warm)

        result = self._qp_solver.solve()

        if result.info.status not in ('solved', 'solved_inaccurate'):
            raise RuntimeError(f"OSQP: {result.info.status}")

        return np.array(result.x), result.info.solve_time

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        gc.disable()  # prevent GC pause during solve (re-enabled in finally)
        t_start = time.perf_counter()
        _t = time.perf_counter  # alias for profiling

        # Prepare history tensors for model
        hist_x = np.array(list(self._hist_x))  # (H+1, nx)
        hist_u = np.array(list(self._hist_u))  # (H, nu)

        hist_x_torch = torch.from_numpy(hist_x).float().to(self.model.device)
        hist_u_torch = torch.from_numpy(hist_u).float().to(self.model.device)

        # Get MPC initials from model (z_curr, A, B, ctx)
        _t0 = _t()
        if self.use_incremental_ctx and self._ctx_inference_params is not None:
            # Incremental: feed only new history tokens since last solve
            K = self._new_hist_count  # number of new tokens since last solve
            # hist_x_new: last K+1 states, hist_u_new: last K controls
            hist_x_new = hist_x_torch[-(K + 1):]  # (K+1, nx)
            hist_u_new = hist_u_torch[-K:]         # (K, nu)
            _t_step = _t()
            z_curr, z_logvar, A, B, ctx, self._ctx_inference_params, _mpc_init_profile = (
                self.model.get_mpc_initials_incremental(
                    hist_x_new, hist_u_new, self._ctx_inference_params))
            self._profile_ctx_step_ms = (_t() - _t_step) * 1000.0
        elif self.use_incremental_ctx and self._ctx_inference_params is None:
            # First solve: full forward to prime the cache
            ip = self.model.init_ctx_inference_params(batch_size=1)
            z_curr, z_logvar, A, B, ctx, self._ctx_inference_params, _mpc_init_profile = (
                self.model.get_mpc_initials_with_cache(
                    hist_x_torch, hist_u_torch, inference_params=ip))
            self._profile_ctx_step_ms = None  # no step measurement on first solve
        else:
            z_curr, z_logvar, A, B, ctx, _mpc_init_profile = self.model.get_mpc_initials(
                hist_x_torch, hist_u_torch)
            self._profile_ctx_step_ms = None
        self._profile_mpc_init_detail = _mpc_init_profile
        if self.use_incremental_ctx:
            self._new_hist_count = 0  # reset counter after consuming
        self._profile_mpc_init = _t() - _t0
        if self._profile_ctx_step_ms is not None:
            self._ctx_step_time_history.append(self._profile_ctx_step_ms)
        # Track A/B/ctx changes between consecutive solves
        ctx_np = ctx.detach().squeeze(0).cpu().numpy()
        A_norm_fro = float(np.linalg.norm(A, 'fro'))
        self._A_norm_history.append(A_norm_fro)
        if self._A_cached is not None:
            A_diff_fro = float(np.linalg.norm(A - self._A_cached, 'fro'))
            self._A_diff_history.append(A_diff_fro)
            self._A_rel_diff_history.append(A_diff_fro / max(A_norm_fro, 1e-10))
            self._B_diff_history.append(float(np.linalg.norm(B - self._B_cached, 'fro')))
            self._ctx_diff_history.append(float(np.linalg.norm(ctx_np - self._ctx_cached)))
        else:
            self._A_diff_history.append(0.0)
            self._A_rel_diff_history.append(0.0)
            self._B_diff_history.append(0.0)
            self._ctx_diff_history.append(0.0)
        self._ctx_cached = ctx_np
        # One-step prediction error: compare z_curr with z_opt[:,1] from previous solve
        if self._z_opt is not None:
            z_predicted = self._z_opt[:, 1]  # what previous solve predicted for this time
            self._z_pred_err_history.append(float(np.linalg.norm(z_curr - z_predicted)))
            # Decode both to quaternion and compute geodesic angle
            with torch.no_grad():
                z_pred_t = torch.from_numpy(z_predicted).float().to(self.model.device).unsqueeze(0)
                z_curr_t = torch.from_numpy(z_curr).float().to(self.model.device).unsqueeze(0)
                x_pred = self.model.decode(z_pred_t).squeeze(0).cpu().numpy()
                x_act = self.model.decode(z_curr_t).squeeze(0).cpu().numpy()
            q_pred = x_pred[:4] / np.linalg.norm(x_pred[:4])
            q_act = x_act[:4] / np.linalg.norm(x_act[:4])
            dot_q = np.abs(np.dot(q_pred, q_act))
            self._q_goal_err_deg_history.append(
                float(np.degrees(2.0 * np.arccos(np.clip(dot_q, 0.0, 1.0)))))
        else:
            self._z_pred_err_history.append(0.0)
            self._q_goal_err_deg_history.append(0.0)

        self._A_cached = A
        self._B_cached = B

        # A eigenvalue diagnostics: check for negative-real eigenvalues (cause oscillating control)
        eigs_A = np.linalg.eigvals(A)
        eigs_neg_real = eigs_A[eigs_A.real < 0]  # eigenvalues with negative real part
        if len(eigs_neg_real) > 0:
            most_neg = eigs_neg_real[np.argmin(eigs_neg_real.real)]
            self._A_eig_most_neg_history.append((float(most_neg.real), float(most_neg.imag), float(abs(most_neg))))
        else:
            self._A_eig_most_neg_history.append((0.0, 0.0, 0.0))

        # B singular value diagnostics
        sv = np.linalg.svd(B, compute_uv=False)
        self._B_sv_history.append((float(sv[-1]), float(sv[0]), float(sv[0] / max(sv[-1], 1e-15))))

        W = self._update_W(z_curr)
        self._W_history.append(W)
        self._z_history.append(z_curr.copy())
        self._ctx_history.append(ctx.detach().cpu())

        # Encode goal state with caching to prevent z_goal oscillation.
        # Quaternion double-cover: enc(q) ≠ enc(-q).  Near dot(q_curr,q_goal)≈0
        # the naive unwrap flips q_goal sign each solve, causing z_goal to jump.
        # Fix: compute z_goal ONCE (first solve) and cache it.  Only recompute
        # when the physical reference target changes.
        q_goal_raw = rbk.MRP2EP(rbk.EP2MRP(x_goal[:4]))  # canonical q_w>=0 form
        need_recompute = (self._z_goal_cached is None
                          or self._q_goal_raw_cached is None
                          or not np.allclose(q_goal_raw, self._q_goal_raw_cached, atol=1e-4))
        if need_recompute:
            x_goal_pos = x_goal.copy()
            x_goal_neg = x_goal.copy()
            x_goal_neg[:4] = -x_goal_neg[:4]
            x_pos_torch = torch.from_numpy(x_goal_pos).float().to(self.model.device)
            x_neg_torch = torch.from_numpy(x_goal_neg).float().to(self.model.device)
            with torch.no_grad():
                z_pos, _ = self.model.encode(x_pos_torch.unsqueeze(0))
                z_neg, _ = self.model.encode(x_neg_torch.unsqueeze(0))
            z_pos = z_pos.squeeze(0).cpu().numpy().astype(np.float64)
            z_neg = z_neg.squeeze(0).cpu().numpy().astype(np.float64)
            # Pick the sign giving shorter path from z_curr (first solve only)
            if np.linalg.norm(z_curr - z_pos) <= np.linalg.norm(z_curr - z_neg):
                self._z_goal_cached = z_pos
                self._x_goal_for_ws = x_goal_pos
            else:
                self._z_goal_cached = z_neg
                self._x_goal_for_ws = x_goal_neg
            self._q_goal_raw_cached = q_goal_raw.copy()
        z_goal = self._z_goal_cached
        x_goal_for_ws = self._x_goal_for_ws
        self._z_goal_history.append(z_goal.copy())

        # Build goal trajectory (constant goal for regulation)
        z_goal_traj = np.tile(z_goal.reshape(-1, 1), (1, self.horizon + 1))

        # Compute warm start (sets self._z_warm, self._u_warm)
        self._compute_warm_start(z_curr, z_goal, x_curr, x_goal_for_ws)

        # Build nominal trajectory for stage-varying Q and KOZ linearization.
        # Forward-propagate from z_curr using current dynamics and warm-start controls.
        if self._prev_z_sol is not None:
            z_nom = np.zeros((self.horizon, self.nz))
            z_nom[0] = A @ z_curr + B @ self._u_warm[:, 0]
            for k in range(1, self.horizon):
                z_nom[k] = A @ z_nom[k - 1] + B @ self._u_warm[:, k]
        else:
            z_nom = self._z_warm[:, 1:].T  # Cold start: SLERP-based

        # Compute stage-varying latent cost matrices via batch Jacobians
        # along [z_curr, z_nom[0], ..., z_nom[N-2]] for Q_k and z_nom[N-1] for F.
        _t0 = _t()
        z_jac_points = np.vstack([z_curr.reshape(1, -1), z_nom])  # (N+1, d)
        z_jac_torch = torch.from_numpy(z_jac_points).to(self.model.device, dtype=self.model.dtype)
        Q_lat_seq, F_latent = get_latent_QF_batch(
            self.model, z_jac_torch, self.Q_diag, self.F_diag,
            eig_floor_ratio=self._eig_floor_ratio)
        self._profile_QF_jac = _t() - _t0
        R_latent = np.diag(self.R_diag)

        # Build KOZ constraint data if enabled
        _t0 = _t()
        a_k_ca, rhs_ca = None, None
        if self._use_koz:
            z_nom_torch = torch.from_numpy(z_nom).to(self.model.device, dtype=self.model.dtype)
            bs_N_batch, J_bs_batch = self.bs_handler.compute(z_nom_torch)

            P0 = np.diag(np.exp(z_logvar))                      # (D,) log-var → (D, D) covariance
            P_seq_full = self._compute_P_seq(A, P0, W)          # (N+1, D, D)  A for closed-loop propagation
            P_seq = P_seq_full[1:, :, :]                       # (N,   D, D)  -> k=1..N

            a_k, rhs, active_idx, cos_theta_nom = self.koz_handler.build_constraints(
                z_nom, bs_N_batch, J_bs_batch, P_seq
            )

            a_k_ca = np.zeros((self.num_bs * self.num_koz * self.horizon, self.nz), dtype=np.float64)
            rhs_ca = np.zeros((self.num_bs * self.num_koz, self.horizon), dtype=np.float64)

            time_idx, bs_idx, koz_idx = active_idx
            lm_idx = bs_idx * self.num_koz + koz_idx
            row_idx = lm_idx * self.horizon + time_idx
            a_k_ca[row_idx, :] = a_k
            rhs_ca[lm_idx, time_idx] = rhs

            # Record per-step CC tightening (cosine units)
            if self.record_lin_error:
                tighten_step = np.zeros(self.horizon)
                if len(active_idx[0]) > 0:
                    k_alpha = self.koz_handler.k_alpha
                    for k in range(self.horizon):
                        mask = active_idx[0] == k
                        if not mask.any():
                            continue
                        a_k_k = a_k[mask]
                        P_k = P_seq[k]
                        sigma2 = np.einsum('ni, ij, nj -> n', a_k_k, P_k, a_k_k)
                        sigma_b = np.sqrt(np.maximum(sigma2, 0.0))
                        tighten_step[k] = float(np.max(k_alpha * sigma_b))
                self._tighten_history.append(tighten_step)
        self._profile_koz = _t() - _t0

        # Solve QP
        _t0 = _t()
        try:
            # u_prev for Δu penalty & trust region: shifted previous plan (warm start first step)
            u_prev_du = self._u_warm[:, 0].copy()
            z_sol, u_sol, qp_wall = self._solve_qp(
                A, B, Q_lat_seq, F_latent, R_latent,
                z_curr, z_goal, self._z_warm, self._u_warm,
                u_prev_du, a_k_ca, rhs_ca)
            self._profile_qp = _t() - _t0

            # --- SCP re-linearization: if solution deviates from nominal, redo KOZ + QP ---
            if self._scp_enabled and self._use_koz:
                z_sol_traj = z_sol[:, 1:].T  # (N, D)
                dev = np.max(np.abs(z_sol_traj - z_nom))
                if dev > self._scp_threshold:
                    _t0_scp = _t()
                    z_nom2 = z_sol_traj
                    z_nom2_torch = torch.from_numpy(z_nom2).to(
                        self.model.device, dtype=self.model.dtype)
                    bs_N2, J_bs2 = self.bs_handler.compute(z_nom2_torch)
                    a_k2, rhs2, active_idx2, cos_theta_nom2 = \
                        self.koz_handler.build_constraints(z_nom2, bs_N2, J_bs2, P_seq)

                    a_k_ca2 = np.zeros_like(a_k_ca)
                    rhs_ca2 = np.zeros_like(rhs_ca)
                    t2, b2, k2 = active_idx2
                    lm2 = b2 * self.num_koz + k2
                    a_k_ca2[lm2 * self.horizon + t2, :] = a_k2
                    rhs_ca2[lm2, t2] = rhs2

                    # Re-solve with first solution as warm start
                    z_sol, u_sol, qp_wall2 = self._solve_qp(
                        A, B, Q_lat_seq, F_latent, R_latent,
                        z_curr, z_goal, z_sol, u_sol,
                        u_prev_du, a_k_ca2, rhs_ca2)
                    qp_wall += qp_wall2
                    self._profile_qp += _t() - _t0_scp

                    # Update for linearization error recording
                    z_nom = z_nom2
                    a_k = a_k2
                    active_idx = active_idx2
                    cos_theta_nom = cos_theta_nom2

            # Extract nominal MPC solution
            self._prev_z_sol = z_sol
            self._prev_u_sol = u_sol

            # Save trajectory
            self._z_opt = z_sol.copy()
            self._v_opt = u_sol.copy()

            # First control
            u_opt = u_sol[:, 0]
            # Track consecutive MPC first-step control change
            if len(self._v_history) > 0:
                self._u_diff_history.append(float(np.linalg.norm(u_opt - self._v_history[-1])))
            else:
                self._u_diff_history.append(0.0)
            self._v_history.append(u_opt.copy())
            self._z_err_norm_history.append(0.0)  # At solve step, z_ref ≈ z_curr
            self._v_plan_history.append(u_sol.copy())  # (nu, N) planned v trajectory


            # Linearization error recording
            if self._use_koz and self.record_lin_error:
                self._record_linearization_error(
                    z_nom, a_k, active_idx, cos_theta_nom,
                )

        except RuntimeError as e:
            self._profile_qp = _t() - _t0
            qp_wall = self._profile_qp
            self.solve_failures += 1
            # On failure, use shifted control from previous solution
            if self._prev_u_sol is not None:
                v_shifted = self._prev_u_sol[:, 1]
                u_opt = v_shifted
                self._v_history.append(v_shifted.copy())

                self._prev_u_sol = np.column_stack([
                    self._prev_u_sol[:, 1:],
                    self._prev_u_sol[:, -1]
                ])

                self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                    f"[{self.ModelTag}] MPC solve failed: {e}. Using shifted control.")
            else:
                u_opt = np.zeros(self.nu)
                self._v_history.append(np.zeros(self.nu))
                self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                    f"[{self.ModelTag}] MPC solve failed: {e}. No previous solution, using zero control.")

        t_end = time.perf_counter()
        total_time = t_end - t_start
        gc.enable()  # re-enable GC after timing-critical section

        self.wall_times.append(float(qp_wall))
        self.total_times.append(total_time)

        # Accumulate profiling data
        if not hasattr(self, '_profile_accum'):
            self._profile_accum = {'mpc_init': [], 'QF_jac': [], 'koz': [], 'qp': [],
                                   'mpc_init/get_ctx': [], 'mpc_init/get_AB': [],
                                   'mpc_init/encode': [], 'mpc_init/to_numpy': []}
        self._profile_accum['mpc_init'].append(self._profile_mpc_init)
        for _k in ('get_ctx', 'get_AB', 'encode', 'to_numpy'):
            self._profile_accum[f'mpc_init/{_k}'].append(
                self._profile_mpc_init_detail.get(_k, 0.0))
        self._profile_accum['QF_jac'].append(self._profile_QF_jac)
        self._profile_accum['koz'].append(self._profile_koz)
        self._profile_accum['qp'].append(self._profile_qp)
        if self._profile_ctx_step_ms is not None:
            self._profile_accum.setdefault('ctx_step', []).append(
                self._profile_ctx_step_ms / 1000.0)  # store in seconds like others

        self._prev_step_data = (z_curr, u_opt, A, B)

        # Q-weighted latent error (outside timing)
        z_err = z_curr - z_goal
        self._z_cost_Q_history.append(float(z_err @ Q_lat_seq[0] @ z_err))

        return u_opt.copy()

    def _compute_warm_start(self, z_curr: np.ndarray, z_goal: np.ndarray,
                             x_curr: np.ndarray | None = None,
                             x_goal: np.ndarray | None = None) -> None:
        """Compute warm-start arrays for the QP solver."""
        if self._prev_z_sol is not None and self._prev_u_sol is not None:
            # Warm start: shift previous solution
            self._z_warm = np.zeros((self.nz, self.horizon + 1))
            self._u_warm = np.zeros((self.nu, self.horizon))

            self._z_warm[:, :-1] = self._prev_z_sol[:, 1:]
            self._z_warm[:, -1] = self._prev_z_sol[:, -1]
            self._u_warm[:, :-1] = self._prev_u_sol[:, 1:]
            self._u_warm[:, -1] = self._prev_u_sol[:, -1]
        else:
            # Cold start: SLERP in physical space, then encode to latent
            alphas = np.linspace(0.0, 1.0, self.horizon + 1)
            if x_curr is not None and x_goal is not None:
                q_curr, omega_curr = x_curr[:4], x_curr[4:]
                q_goal_ws, omega_goal = x_goal[:4], x_goal[4:]
                z_warm = np.zeros((self.nz, self.horizon + 1))
                for i, a in enumerate(alphas):
                    q_i = slerp(q_curr, q_goal_ws, a)
                    omega_i = (1.0 - a) * omega_curr + a * omega_goal
                    x_i = np.concatenate([q_i, omega_i])
                    x_torch = torch.from_numpy(x_i).float().to(self.model.device)
                    with torch.no_grad():
                        z_i, _ = self.model.encode(x_torch.unsqueeze(0))
                    z_warm[:, i] = z_i.squeeze(0).cpu().numpy().astype(np.float64)
                self._z_warm = z_warm
            else:
                # Fallback: linear interpolation in latent space
                self._z_warm = (1 - alphas) * z_curr.reshape(-1, 1) + alphas * z_goal.reshape(-1, 1)
            self._u_warm = np.zeros((self.nu, self.horizon))

    def _compute_P_seq(self, A: np.ndarray, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        P_seq = np.zeros((self.horizon + 1, self.nz, self.nz), dtype=np.float64)
        P_seq[0] = P
        for k in range(self.horizon):
            P_seq[k + 1] = A @ P_seq[k] @ A.T + W
        return P_seq

    def _record_linearization_error(
        self,
        z_nom: np.ndarray,
        a_k: np.ndarray,
        active_idx: tuple,
        cos_theta_nom: np.ndarray,
    ) -> None:
        """Compute and store linearization error between nonlinear and linearized KOZ.

        Computes |g(z_opt) - g_tilde(z_opt)| only for **active** constraints
        (those within the KOZ buffer).  Inactive constraints have no linearization
        applied in the QP, so their raw |g(z_opt) - g(z_nom)| is trajectory
        divergence, not linearization error — including them would inflate the
        margin statistics.

        Args:
            z_nom: (H, D) linearization point.
            a_k: (N_act, D) linearized constraint coefficient vectors.
            active_idx: Tuple of (time, boresight, KOZ) index arrays for active constraints.
            cos_theta_nom: (H, L, M) nonlinear cosine at z_nom.
        """
        H = self.horizon
        L = self.num_bs
        M = self.num_koz

        # z_opt: optimal trajectory from MPC, columns 1..H (skip initial z0)
        z_opt_np = self._prev_z_sol[:, 1:].T  # (H, D)
        z_opt_torch = torch.from_numpy(z_opt_np).to(
            self.model.device, dtype=self.model.dtype
        )

        # 1. Nonlinear cos_theta at z_opt
        bs_N_opt, _ = self.bs_handler.compute(z_opt_torch)  # (H, L, 3)
        cos_theta_actual = np.einsum(
            'hli, mi -> hlm', bs_N_opt, self.koz_handler.koz_dirs
        )  # (H, L, M)

        # 2. Linearized cos_theta at z_opt
        #    g_tilde(z_opt) = g(z_nom) + a_k^T * (z_opt - z_nom)
        cos_theta_linear = cos_theta_nom.copy()  # (H, L, M)
        n_active = len(active_idx[0])
        if n_active > 0:
            dz = z_opt_np[active_idx[0]] - z_nom[active_idx[0]]  # (N_act, D)
            linear_correction = np.sum(a_k * dz, axis=1)  # (N_act,)
            cos_theta_linear[active_idx] += linear_correction

        # 3. Signed error — only for active constraints
        #    Positive = actual cos > linearized cos = closer to KOZ than QP thinks (dangerous)
        #    Negative = actual cos < linearized cos = farther from KOZ (safe)
        lin_error = cos_theta_actual - cos_theta_linear  # (H, L, M)
        active_mask = np.zeros((H, L, M), dtype=bool)
        active_mask[active_idx] = True
        lin_error[~active_mask] = 0.0
        self._lin_errors.append(lin_error)

        # 4. Latent deviation (for threshold calibration)
        self._lin_devs.append(float(np.max(np.abs(z_opt_np - z_nom))))

    def get_lin_errors(self) -> list[np.ndarray]:
        """Return list of (H, L, M) linearization error arrays, one per MPC solve."""
        return self._lin_errors

    def get_lin_devs(self) -> list[float]:
        """Return ||z_opt - z_nom||_inf per MPC solve."""
        return self._lin_devs

    def get_W_history(self) -> np.ndarray:
        return np.array(self._W_history)

    def get_s_history(self) -> np.ndarray:
        """Return scalar W-scale factor at each MPC solve (for diagnostics)."""
        return np.array(self._s_history)

    def get_z_history(self) -> np.ndarray:
        return np.array(self._z_history)

    def get_ctx_history(self) -> list:
        return self._ctx_history


    def get_v_history(self) -> np.ndarray:
        """Return (N_mpc, nu) array of v (MPC decision variable) at each MPC solve."""
        return np.array(self._v_history) if self._v_history else np.empty((0, self.nu))

    def get_z_goal_history(self) -> np.ndarray:
        """Return (N_mpc, nz) array of z_goal at each MPC solve."""
        return np.array(self._z_goal_history) if self._z_goal_history else np.empty((0, self.nz))

    def get_z_err_norm_history(self) -> np.ndarray:
        """Return (N_ctrl,) array of ||z_curr - z_ref|| at each inter-solve step."""
        return np.array(self._z_err_norm_history)

    def get_ctx_diff_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||ctx_k - ctx_{k-1}|| between consecutive MPC solves."""
        return np.array(self._ctx_diff_history)

    def get_ctx_step_time_history(self) -> np.ndarray:
        """Return (N_solve-1,) array of incremental ctx step time [ms] per MPC solve.
        Empty if use_incremental_ctx is False."""
        return np.array(self._ctx_step_time_history)

    def get_A_diff_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||A_k - A_{k-1}||_F between consecutive MPC solves."""
        return np.array(self._A_diff_history)

    def get_A_rel_diff_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||A_k - A_{k-1}||_F / ||A_k||_F (relative change)."""
        return np.array(self._A_rel_diff_history)

    def get_A_norm_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||A_k||_F at each MPC solve."""
        return np.array(self._A_norm_history)

    def get_B_diff_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||B_k - B_{k-1}||_F between consecutive MPC solves."""
        return np.array(self._B_diff_history)

    def get_u_diff_history(self) -> np.ndarray:
        """Return (N_solve,) array of ||v_k[0] - v_{k-1}[0]|| between consecutive MPC first-step controls."""
        return np.array(self._u_diff_history)

    def get_B_sv_history(self) -> np.ndarray:
        """Return (N_solve, 3) array of (σ_min, σ_max, cond) of B at each MPC solve."""
        return np.array(self._B_sv_history) if self._B_sv_history else np.empty((0, 3))

    def get_z_pred_err_history(self) -> np.ndarray:
        """Return (N_solve,) one-step prediction error ||z_curr - z_predicted||."""
        return np.array(self._z_pred_err_history)

    def get_q_goal_err_deg_history(self) -> np.ndarray:
        """Return (N_solve,) geodesic angle to q_goal [deg]."""
        return np.array(self._q_goal_err_deg_history)

    def get_v_plan_history(self) -> list[np.ndarray]:
        """Return list of (nu, N) planned v trajectories at each MPC solve."""
        return self._v_plan_history

    def get_z_cost_Q_history(self) -> np.ndarray:
        """Return (N_mpc,) array of ||z_err||_{Q_latent}^2 at each MPC solve."""
        return np.array(self._z_cost_Q_history)

    def get_tighten_history(self) -> np.ndarray:
        """Return (N_solve, horizon) CC tightening per step in cosine units."""
        if not self._tighten_history:
            return np.empty((0, self.horizon))
        return np.array(self._tighten_history)

    def get_wall_time_stats(self) -> dict:
        if not self.wall_times:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
        
        return {
            'mean': np.mean(self.wall_times),
            'max': np.max(self.wall_times),
            'min': np.min(self.wall_times),
            'std': float(np.std(self.wall_times)),
            'count': len(self.wall_times)
        }
    
    def get_total_time_stats(self) -> dict:
        if not self.total_times:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}

        return {
            'mean': np.mean(self.total_times),
            'max': np.max(self.total_times),
            'min': np.min(self.total_times),
            'std': float(np.std(self.total_times)),
            'count': len(self.total_times)
        }

    def get_profile_summary(self, warmup_skip: int = 3) -> str:
        """Return formatted profiling summary of _solve() sections.

        Args:
            warmup_skip: Number of initial calls to exclude (CUDA JIT warmup).
        """
        if not hasattr(self, '_profile_accum') or not self._profile_accum.get('qp'):
            return "No profiling data."
        n_total = len(self._profile_accum['qp'])
        s = min(warmup_skip, n_total - 1)  # keep at least 1 sample
        n = n_total - s
        mpc_keys = ['mpc_init', 'QF_jac', 'koz', 'qp']
        mpc_init_sub_keys = ['mpc_init/get_ctx', 'mpc_init/get_AB',
                             'mpc_init/encode', 'mpc_init/to_numpy']

        # Header
        lines = [f"_solve() profiling ({n_total} calls, first {s} skipped → {n} calls, ms):"]
        lines.append(f"  {'':20s}  {'mean':>7s}  {'std':>7s}  {'WCET':>7s}  {'min':>7s}")
        lines.append(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

        # Per-section stats (skip warmup)
        for key in mpc_keys:
            vals = np.array(self._profile_accum.get(key, [0.0]))[s:] * 1000
            wcet_idx = int(np.argmax(vals)) + s  # original call index
            lines.append(
                f"  {key:20s}  {np.mean(vals):7.2f}  {np.std(vals):7.2f}"
                f"  {np.max(vals):7.2f}  {np.min(vals):7.2f}  (WCET@step {wcet_idx})"
            )
            # mpc_init sub-breakdown
            if key == 'mpc_init':
                for sub in mpc_init_sub_keys:
                    sub_data = self._profile_accum.get(sub)
                    if sub_data and len(sub_data) > s:
                        sv = np.array(sub_data)[s:] * 1000
                        label = '  ' + sub.split('/')[-1]
                        lines.append(
                            f"  {label:20s}  {np.mean(sv):7.2f}  {np.std(sv):7.2f}"
                            f"  {np.max(sv):7.2f}  {np.min(sv):7.2f}"
                        )

        # ctx_step (incremental context encoding, if enabled)
        if 'ctx_step' in self._profile_accum and len(self._profile_accum['ctx_step']) > 0:
            ctx_vals = np.array(self._profile_accum['ctx_step']) * 1000  # no warmup skip — first entry already excludes priming
            lines.append(
                f"  {'ctx_step':20s}  {np.mean(ctx_vals):7.2f}  {np.std(ctx_vals):7.2f}"
                f"  {np.max(ctx_vals):7.2f}  {np.min(ctx_vals):7.2f}  ({len(ctx_vals)} calls)"
            )

        # Other (unaccounted)
        mpc_per_call = np.zeros(n)
        for key in mpc_keys:
            mpc_per_call += np.array(self._profile_accum[key])[s:] * 1000
        total_arr = np.array(self.total_times)[s:] * 1000
        other_arr = total_arr - mpc_per_call
        lines.append(
            f"  {'other':20s}  {np.mean(other_arr):7.2f}  {np.std(other_arr):7.2f}"
            f"  {np.max(other_arr):7.2f}  {np.min(other_arr):7.2f}"
        )
        lines.append(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

        # MPC only (sum of mpc_keys per call, excl. intersolve + other)
        lines.append(
            f"  {'MPC only':20s}  {np.mean(mpc_per_call):7.2f}  {np.std(mpc_per_call):7.2f}"
            f"  {np.max(mpc_per_call):7.2f}  {np.min(mpc_per_call):7.2f}"
        )
        lines.append(
            f"  {'TOTAL':20s}  {np.mean(total_arr):7.2f}  {np.std(total_arr):7.2f}"
            f"  {np.max(total_arr):7.2f}  {np.min(total_arr):7.2f}"
        )
        lines.append(f"  (first {s} call(s) skipped for CUDA warmup)")
        return "\n".join(lines)