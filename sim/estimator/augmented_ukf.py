"""
13-state augmented UKF for joint attitude + inertia estimation.

State: x_aug = [q(4), omega(3), j(6)]  where j = [J11, J12, J13, J22, J23, J33]
       (row-major upper triangle of 3x3 symmetric inertia matrix)

- q, omega propagated via RK4 attitude dynamics using J(j)
- j evolves as a random walk (identity dynamics)
- Measurement: z = [q(4), omega(3)]  (direct, linear in state)

This module is independent of MPC — it only filters.
"""

import numpy as np
from src.utils.math_numpy import (
    attitude_dyn_dt, j_vec2mat, quat_norm,
)


class AugmentedUKF:
    """13-state UKF: quaternion (4) + angular velocity (3) + inertia (6)."""

    NX = 13          # augmented state dimension
    NX_STATE = 7     # q + omega
    NX_J = 6         # unique inertia elements
    NZ = 7           # measurement dimension (q + omega)

    def __init__(
        self,
        x0_state: np.ndarray,
        j_nom: np.ndarray,
        P0_state: np.ndarray,
        P0_J: np.ndarray,
        Q_state: np.ndarray,
        Q_j: np.ndarray,
        R_meas: np.ndarray,
        dt: float,
        integrator_steps: int,
        ut_alpha: float = 1e-3,
        ut_beta: float = 2.0,
        ut_kappa: float = 0.0,
    ):
        """
        Args:
            x0_state:  (7,)   initial [q, omega].
            j_nom:     (6,)   nominal inertia vector [J11,J12,J13,J22,J23,J33].
            P0_state:  (7,7)  initial state covariance.
            P0_J:      (6,6)  initial inertia covariance.
            Q_state:   (7,7)  state process noise.
            Q_j:       (6,6)  inertia random-walk noise.
            R_meas:    (7,7)  measurement noise.
            dt:        float  control period [s].
            integrator_steps: int  RK4 sub-steps.
            ut_alpha, ut_beta, ut_kappa: UT tuning parameters.
        """
        # Augmented state
        self._x = np.zeros(self.NX, dtype=np.float64)
        self._x[:self.NX_STATE] = x0_state.copy()
        self._x[self.NX_STATE:] = j_nom.copy()

        # Augmented covariance
        self._P = np.zeros((self.NX, self.NX), dtype=np.float64)
        self._P[:self.NX_STATE, :self.NX_STATE] = P0_state.copy()
        self._P[self.NX_STATE:, self.NX_STATE:] = P0_J.copy()

        # Noise matrices
        self._Q = np.zeros((self.NX, self.NX), dtype=np.float64)
        self._Q[:self.NX_STATE, :self.NX_STATE] = Q_state.copy()
        self._Q[self.NX_STATE:, self.NX_STATE:] = Q_j.copy()
        self._R = R_meas.copy()

        # Integration
        self._dt = dt
        self._integrator_steps = integrator_steps

        # UT constants (n=13, 2n+1=27 sigma points)
        n = self.NX
        lam = ut_alpha**2 * (n + ut_kappa) - n
        self._lam = lam
        self._n_pts = 2 * n + 1
        self._scale = n + lam

        self._Wm = np.full(self._n_pts, 0.5 / self._scale)
        self._Wc = np.full(self._n_pts, 0.5 / self._scale)
        self._Wm[0] = lam / self._scale
        self._Wc[0] = lam / self._scale + (1.0 - ut_alpha**2 + ut_beta)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray) -> None:
        """Predict step: propagate sigma points through augmented dynamics."""
        sigma_pts = self._generate_sigma_points()          # (27, 13)
        sigma_pts_prop = self._propagate_sigma_points(sigma_pts, u)
        self._x, self._P = self._ut_recover(sigma_pts_prop)
        self._P += self._Q
        self._P = 0.5 * (self._P + self._P.T)

    def update(self, z_meas: np.ndarray) -> None:
        """Update step with measurement z = [q(4), omega(3)].

        Quaternion hemisphere check: flip z_meas if in opposite hemisphere.
        """
        z = z_meas.copy()

        # Quaternion sign fix
        if np.dot(self._x[:4], z[:4]) < 0.0:
            z[:4] *= -1.0

        # Generate sigma points from predicted state
        sigma_pts = self._generate_sigma_points()          # (27, 13)

        # Transform through measurement model h(x) = x[:7]
        Z_sigma = sigma_pts[:, :self.NZ]                   # (27, 7)
        z_mean = self._Wm @ Z_sigma                        # (7,)

        # Normalize predicted measurement quaternion
        q_norm = np.linalg.norm(z_mean[:4])
        if q_norm > 1e-8:
            z_mean[:4] /= q_norm

        # Innovation covariance Pzz and cross-covariance Pxz
        Z_diff = Z_sigma - z_mean                          # (27, 7)
        X_diff = sigma_pts - self._x                       # (27, 13)

        Pzz = (Z_diff.T * self._Wc) @ Z_diff + self._R    # (7, 7)
        Pxz = (X_diff.T * self._Wc) @ Z_diff               # (13, 7)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)                       # (13, 7)

        # State update
        innovation = z - z_mean                             # (7,)
        self._x = self._x + K @ innovation

        # Covariance update (Joseph form is more stable but standard is fine here)
        self._P = self._P - K @ Pzz @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        # Normalize quaternion
        q_n = np.linalg.norm(self._x[:4])
        if q_n > 1e-8:
            self._x[:4] /= q_n

    def reset(
        self,
        x0_state: np.ndarray,
        j_nom: np.ndarray,
        P0_state: np.ndarray,
        P0_J: np.ndarray,
    ) -> None:
        """Re-initialize estimator to given state and covariance."""
        self._x[:self.NX_STATE] = x0_state.copy()
        self._x[self.NX_STATE:] = j_nom.copy()
        self._P[:] = 0.0
        self._P[:self.NX_STATE, :self.NX_STATE] = P0_state.copy()
        self._P[self.NX_STATE:, self.NX_STATE:] = P0_J.copy()

    def get_state_mean(self) -> np.ndarray:
        """Full 13-element augmented state."""
        return self._x.copy()

    def get_J_hat(self) -> np.ndarray:
        """Estimated 3×3 inertia matrix."""
        return j_vec2mat(self._x[self.NX_STATE:])

    def get_j_vec(self) -> np.ndarray:
        """Estimated 6-element inertia vector."""
        return self._x[self.NX_STATE:].copy()

    def get_P_state(self) -> np.ndarray:
        """7×7 state covariance (q + omega block)."""
        return self._P[:self.NX_STATE, :self.NX_STATE].copy()

    def get_P_J(self) -> np.ndarray:
        """6×6 inertia covariance block."""
        return self._P[self.NX_STATE:, self.NX_STATE:].copy()

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate 2n+1 = 27 sigma points from current (x, P)."""
        n = self.NX
        S = _safe_cholesky(self._scale * self._P)

        pts = np.empty((self._n_pts, n), dtype=np.float64)
        pts[0] = self._x
        pts[1:n + 1] = self._x + S.T
        pts[n + 1:] = self._x - S.T

        # Normalize quaternion part of every sigma point
        for i in range(self._n_pts):
            q_n = np.linalg.norm(pts[i, :4])
            if q_n > 1e-8:
                pts[i, :4] /= q_n

        return pts

    def _propagate_sigma_points(
        self, pts: np.ndarray, u: np.ndarray,
    ) -> np.ndarray:
        """Propagate each sigma point through augmented dynamics.

        - q, omega: RK4 attitude dynamics with per-point J(j)
        - j: identity (random walk)
        """
        pts_prop = np.empty_like(pts)
        for i in range(self._n_pts):
            state_i = pts[i, :self.NX_STATE]       # (7,)
            j_i = pts[i, self.NX_STATE:]            # (6,)

            J_i = j_vec2mat(j_i)

            # Guard against non-positive-definite J from extreme sigma points
            try:
                eigvals = np.linalg.eigvalsh(J_i)
                if np.any(eigvals < 1e-4):
                    eigvals = np.maximum(eigvals, 1e-4)
                    V = np.linalg.eigh(J_i)[1]
                    J_i = V @ np.diag(eigvals) @ V.T
                J_i_inv = np.linalg.inv(J_i)
            except np.linalg.LinAlgError:
                # Fallback: skip this sigma point, copy mean propagation
                J_i = j_vec2mat(self._x[self.NX_STATE:])
                J_i_inv = np.linalg.inv(J_i)

            state_next = attitude_dyn_dt(
                state_i, u, self._dt, J_i, J_i_inv,
                steps=self._integrator_steps,
            )
            pts_prop[i, :self.NX_STATE] = state_next
            pts_prop[i, self.NX_STATE:] = j_i       # random walk

        return pts_prop

    def _ut_recover(
        self, pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover mean and covariance from sigma points."""
        x_mean = self._Wm @ pts                     # (13,)

        # Normalize quaternion mean
        q_n = np.linalg.norm(x_mean[:4])
        if q_n > 1e-8:
            x_mean[:4] /= q_n

        diff = pts - x_mean                          # (27, 13)
        P = (diff.T * self._Wc) @ diff               # (13, 13)
        P = 0.5 * (P + P.T)

        return x_mean, P


# ======================================================================
#  Module-level utility
# ======================================================================

def _safe_cholesky(A: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Cholesky with fallback jitter."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
