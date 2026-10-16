import time
import logging
import numpy as np
import casadi as ca
from omegaconf import DictConfig

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.architecture.bskLogging import BasiliskError
from Basilisk.utilities import RigidBodyKinematics as rbk

from sim.koz import QuadKOZ
from sim.utils.math_utils import quat_error_ca

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NMPC")

class NMPC(sysModel.SysModel):
    """
    CasADi-based Nonlinear MPC for spacecraft attitude control with constraints.
    - Quaternion-based attitude dynamics
    - Keepout Zone (KOZ) constraints as soft constraints
    - Torque bounds
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
                 integrator_steps: int,
                 du_weight: float = 0.0,
                 **kwargs):
        """
        Args:
            horizon: MPC prediction horizon steps
            mpc_dt: MPC prediction timestep [s]
            inertia: Spacecraft inertia tensor [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz] [kg.m^2]
            Q_diag: State cost diagonal [6] (quat_err[3], omega_err[3])
            R_diag: Control cost diagonal [3] (torque)
            F_diag: Terminal state cost diagonal [6]
            rho: Soft constraint penalty weight for KOZ violation
            max_iter: Maximum IPOPT iterations
            constraint_cfg: Constraint configuration (torque bounds, KOZ list, boresight)
        """
        super(NMPC, self).__init__()
        self.ModelTag = "NMPC"
        
        # Validate inputs
        if mpc_dt <= 0:
            raise ValueError(f"mpc_dt must be positive: {mpc_dt}")
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2 for proper MPC operation: {horizon}")
        
        self.horizon = horizon
        self.mpc_dt = mpc_dt
        self.max_iter = max_iter
        self.rho = rho
        self.integrator_steps = integrator_steps
        self.du_weight = du_weight
        # Inertia matrix setup
        inertia_arr = np.array(inertia)
        if inertia_arr.size == 9:
            self.J = inertia_arr.reshape(3, 3)
        else:
            raise ValueError(f"Inertia must be 9 elements, got {inertia_arr.size}")
        self.J_inv = np.linalg.inv(self.J)
        
        # Cost matrices (for 6D error: quat_err[3] + omega_err[3])
        self.Q = ca.diag(Q_diag)
        self.R = ca.diag(R_diag)
        self.F = ca.diag(F_diag)

        # Constraint parameters
        self.u_max = constraint_cfg.u_max
        self.u_min = constraint_cfg.u_min
        # KOZ (Keepout Zones) quadratic matrices for all (boresight, KOZ) pairs
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

        # State: [q0, q1, q2, q3, w1, w2, w3]
        # Control: [u1, u2, u3]
        self.nx = 7
        self.nu = 3
        
        # Solver (initialized in Reset)
        self.opti = None
        self.solver = None
        
        # Warm start storage
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        
        # Timing statistics
        self.total_times: list[float] = []
        self.wall_times: list[float] = []

        # Solver status tracking
        self.solve_failures: int = 0
        self.solver_statuses: list[str] = []  # IPOPT return_status per call
        
        # Input messages
        # - attNavInMsg: absolute spacecraft attitude/rate from simpleNav (B/N, w_BN_B)
        # - attRefInMsg: reference attitude from guidance (R/N)
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.attRefInMsg = messaging.AttRefMsgReader()
        
        # Output messages
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        
        logger.info(f"[NMPC] Initialized. horizon={horizon}, mpc_dt={mpc_dt}s")

    def Reset(self, CurrentSimNanos: int):
        """
        Initialize the NMPC controller.
        
        Args:
            CurrentSimNanos: Current simulation time [ns]
        """
        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, 
            f"[{self.ModelTag}] Resetting: mpc_dt={self.mpc_dt}s, horizon={self.horizon}, J={np.diag(self.J).tolist()} kg.m^2")
        
        if not self.attNavInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attNavInMsg not linked!")
        if not self.attRefInMsg.isLinked():
            raise BasiliskError(f"[{self.ModelTag}] attRefInMsg not linked!")
        
        # Build OCP
        self._setup_ocp()
        
        # Initialize warm start storage
        self._prev_x_sol = None
        self._prev_u_sol = None
        self._u_curr = np.zeros(self.nu)
        self.total_times = []
        self.wall_times = []
        
        # Write zero torque initially
        zero_torque = messaging.CmdTorqueBodyMsgPayload()
        zero_torque.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(zero_torque, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos: int):
        """
        Main control loop - called at mpc_dt intervals.
        
        Args:
            CurrentSimNanos: Current simulation time [ns]
        """
        # Read absolute navigation attitude state (not relative attitude error)
        nav_msg = self.attNavInMsg()
        sigma_BN = np.array(nav_msg.sigma_BN)
        omega_BN_B = np.array(nav_msg.omega_BN_B)
        
        # Convert MRP to quaternion
        q_curr = rbk.MRP2EP(sigma_BN)
        x_curr = np.concatenate([q_curr, omega_BN_B])
        
        # Read reference attitude
        ref_msg = self.attRefInMsg()
        sigma_RN = np.array(ref_msg.sigma_RN)
        q_goal = rbk.MRP2EP(sigma_RN)
        
        # Reference angular velocity (Inertial -> Body)
        # Assuming Regulation (omega_goal = 0) for simplicity. 
        # For tracking, compute: omega_RN_B = dcm_BN @ omega_RN_N
        omega_goal = np.zeros(3)

        # Build goal trajectory
        state_goal = np.concatenate([q_goal, omega_goal])
        x_goal = np.tile(state_goal.reshape(-1, 1), (1, self.horizon + 1))
        
        # Solve MPC
        torque = self._solve(x_curr, x_goal)
        
        # Write output message
        out_msg = messaging.CmdTorqueBodyMsgPayload()
        out_msg.torqueRequestBody = torque.tolist()
        self.cmdTorqueOutMsg.write(out_msg, CurrentSimNanos, self.moduleID)


    def _setup_ocp(self) -> None:
        """
        Set up the CasADi optimal control problem.
        """
        # CasADi Opti Stack
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)
        
        self._use_koz = (self.num_koz > 0) and (self.num_bs > 0)
        if self._use_koz:
            # CasADi Opti supports up to 2D variables; flatten (boresight, koz) pair index.
            self.slack = self.opti.variable(self.num_bs * self.num_koz, self.horizon)
        else:
            self.slack = None
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] num_koz={self.num_koz}, num_bs={self.num_bs}: KOZ constraints disabled.")

        # Parameters
        self.x_curr = self.opti.parameter(self.nx)
        self.x_goal = self.opti.parameter(self.nx, self.horizon + 1)
        if self.du_weight > 0:
            self.u_prev_p = self.opti.parameter(self.nu)

        # Cost function
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

            x_next = self._symbolic_rk4_step(
                self.X[:, k], self.U[:, k],
                self.mpc_dt, self.integrator_steps)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

            self.opti.subject_to(self.opti.bounded(
                self.u_min, self.U[:, k], self.u_max))

        # Terminal cost
        q_err_f = quat_error_ca(self.X[:4, self.horizon], self.x_goal[:4, self.horizon])
        w_err_f = self.X[4:, self.horizon] - self.x_goal[4:, self.horizon]
        x_err_f = ca.vertcat(q_err_f, w_err_f)
        cost += ca.mtimes([x_err_f.T, self.F, x_err_f])
   
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

        # Solver options
        p_opts = {
            'expand': True,
            'print_time': False,
            'record_time': True,
            'verbose': False,
        }
        s_opts = {
            'print_level': 0,
            'max_iter': self.max_iter,
            'sb': 'yes',
            'print_timing_statistics': 'no',
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
        }
        self.opti.solver('ipopt', p_opts, s_opts)
        
        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, 
            f"[{self.ModelTag}] OCP setup complete. num_koz={self.num_koz}, num_bs={self.num_bs}")

    def _solve(self, x_curr: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        """
        Solve the MPC problem.
        
        Args:
            x_curr: Current state [q0, q1, q2, q3, w1, w2, w3]
            x_goal: Goal trajectory [N+1, 7]
        Returns:
            u: Optimal control torque [3]
        """
        t_start = time.perf_counter()
        # Set parameters
        self.opti.set_value(self.x_curr, x_curr)
        self.opti.set_value(self.x_goal, x_goal)
        if self.du_weight > 0:
            self.opti.set_value(self.u_prev_p, self._u_curr)

        # Initial guess (warm start if available, otherwise cold start)
        self._set_initial_guess(x_curr)

        # Solve
        try:
            sol = self.opti.solve()

            # Extract solution
            self._prev_x_sol = sol.value(self.X)
            self._prev_u_sol = sol.value(self.U)
            self._u_curr = self._prev_u_sol[:, 0]

            # Record timing from IPOPT stats
            stats = sol.stats()

        except RuntimeError as e:
            self.solve_failures += 1
            stats = self.opti.debug.stats()
            return_status = str(stats.get('return_status', 'unknown'))

            # Max iter reached: last iterate is often near-optimal, try to use it
            if return_status in ('Maximum_Iterations_Exceeded', 'Solved_To_Acceptable_Level'):
                try:
                    debug_x = self.opti.debug.value(self.X)
                    debug_u = self.opti.debug.value(self.U)

                    # Clip torques to actuator bounds
                    debug_u = np.clip(debug_u, self.u_min, self.u_max)

                    self._prev_x_sol = debug_x
                    self._prev_u_sol = debug_u
                    self._u_curr = debug_u[:, 0]

                    self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                        f"[{self.ModelTag}] Solver {return_status}: using last iterate (clipped).")
                except Exception:
                    self._fallback_control(e)
            else:
                self._fallback_control(e)

        t_end = time.perf_counter()
        total_time = t_end - t_start
        self.wall_times.append(float(stats['t_wall_total']))
        self.total_times.append(total_time)
        self.solver_statuses.append(str(stats.get('return_status', 'unknown')))

        return self._u_curr.copy()

    def _fallback_control(self, error: Exception) -> None:
        """Fallback to shifted previous solution when solver truly fails."""
        if self._prev_u_sol is not None:
            self._u_curr = self._prev_u_sol[:, min(1, self._prev_u_sol.shape[1] - 1)]
            self._prev_u_sol = np.column_stack([
                self._prev_u_sol[:, 1:],
                self._prev_u_sol[:, -1]
            ])
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] MPC solve failed: {error}. Using shifted previous solution.")
        else:
            self.bskLogger.bskLog(bskLogging.BSK_WARNING,
                f"[{self.ModelTag}] MPC solve failed: {error}. No previous solution, using fallback (zero/last).")

    def _set_initial_guess(self, x_curr: np.ndarray) -> None:
        # Warm start
        if self._prev_x_sol is not None and self._prev_u_sol is not None:
            # Shift warm start (use previous solution shifted by one step)
            x_warm = np.zeros((self.nx, self.horizon + 1))
            u_warm = np.zeros((self.nu, self.horizon))

            x_warm[:, :-1] = self._prev_x_sol[:, 1:]
            x_warm[:, -1]  = self._prev_x_sol[:, -1]
            u_warm[:, :-1] = self._prev_u_sol[:, 1:]
            u_warm[:, -1]  = self._prev_u_sol[:, -1]

            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
            if self._use_koz:
                self.opti.set_initial(self.slack, np.zeros((self.num_bs * self.num_koz, self.horizon)))
        else:
            # Cold start
            x_warm = np.tile(x_curr.reshape(-1, 1), (1, self.horizon + 1))
            u_warm = np.zeros((self.nu, self.horizon))
            self.opti.set_initial(self.X, x_warm)
            self.opti.set_initial(self.U, u_warm)
            if self._use_koz:
                self.opti.set_initial(self.slack, np.zeros((self.num_bs * self.num_koz, self.horizon)))

    def _symbolic_rk4_step(self, x, u, dt, substeps):
        """Manual symbolic RK4 — inlined into NLP graph for efficient AD.

        Args:
            x:        (7,) MX state.
            u:        (3,) MX control.
            dt:       float total step size.
            substeps: int number of RK4 sub-steps.
        Returns:
            x_next: (7,) MX propagated state (quaternion normalized).
        """
        h = dt / substeps
        xk = x
        for _ in range(substeps):
            k1 = self._attitude_dynamics(xk, u)
            k2 = self._attitude_dynamics(xk + 0.5 * h * k1, u)
            k3 = self._attitude_dynamics(xk + 0.5 * h * k2, u)
            k4 = self._attitude_dynamics(xk + h * k3, u)
            xk = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Normalize quaternion
        q = xk[:4]
        q_norm = ca.fmax(ca.norm_2(q), 1e-8)
        return ca.vertcat(q / q_norm, xk[4:])

    def _attitude_dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """
        Continuous time dynamics: x_dot = f(x, u)
        x = [q0, q1, q2, q3, w1, w2, w3]
        """
        q = x[:4]
        w = x[4:]
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        # Quaternion kinematics: q_dot = 0.5 * Ω(w) * q
        q_dot = 0.5 * ca.vertcat(
            -q1*w[0] - q2*w[1] - q3*w[2],
             q0*w[0] - q3*w[1] + q2*w[2],
             q3*w[0] + q0*w[1] - q1*w[2],
            -q2*w[0] + q1*w[1] + q0*w[2]
        )
        
        # Angular velocity dynamics: J * w_dot = -w x (J * w) + u
        Jw = ca.mtimes(ca.DM(self.J), w)
        cross_term = ca.cross(w, Jw)
        w_dot = ca.mtimes(ca.DM(self.J_inv), -cross_term + u)
        
        return ca.vertcat(q_dot, w_dot)

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