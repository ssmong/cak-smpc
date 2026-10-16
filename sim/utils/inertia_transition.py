"""Time-varying inertia transition utilities.

Supports abrupt (step) inertia change and physics-based 2-link robot arm
model with parallel axis theorem.
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _minimum_jerk(alpha: float) -> float:
    """Minimum-jerk trajectory: zero velocity & acceleration at endpoints."""
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _point_mass_inertia(mass: float, r: np.ndarray) -> np.ndarray:
    """Inertia of a point mass at *r* about the origin (parallel axis theorem).

    J = m (|r|^2 I_3 - r (x) r).  Returns flattened (9,) row-major array.
    """
    r_sq = np.dot(r, r)
    J = mass * (r_sq * np.eye(3) - np.outer(r, r))
    return J.ravel()


# ---------------------------------------------------------------------------
#  InertiaTransition
# ---------------------------------------------------------------------------

class InertiaTransition:
    """Manages gradual (or abrupt) inertia change during simulation.

    Parameters
    ----------
    pre_inertia : array-like, shape (9,)
        Flattened 3x3 inertia tensor before the change.
        For ``robot_arm``, this is the base spacecraft inertia (no arm mass).
    post_inertia : array-like, shape (9,)
        Flattened 3x3 inertia tensor after the change.
        Ignored for ``robot_arm`` (auto-computed from geometry).
    inertia_change_cfg : dict-like
        The ``sc.inertia_change`` config block.  Expected keys::

            enabled: bool
            time_sec: float
            transition:
                type: str           # "step" or "robot_arm"
                duration_sec: float
                params: dict        # robot_arm params
    """

    def __init__(
        self,
        pre_inertia: np.ndarray,
        post_inertia: np.ndarray,
        inertia_change_cfg: Any,
    ) -> None:
        self._pre = np.asarray(pre_inertia, dtype=float).ravel()
        self._post = np.asarray(post_inertia, dtype=float).ravel()

        cfg = (
            dict(inertia_change_cfg)
            if not isinstance(inertia_change_cfg, dict)
            else inertia_change_cfg
        )

        self._enabled: bool = bool(cfg.get("enabled", False))
        self._t_start: float = float(cfg.get("time_sec", 0.0))

        trans = cfg.get("transition", {})
        if trans is None:
            trans = {}
        trans = dict(trans) if not isinstance(trans, dict) else trans

        self._type: str = str(trans.get("type", "step"))
        self._duration: float = float(trans.get("duration_sec", 0.0))
        params = trans.get("params", {})
        if params is None:
            params = {}
        self._params: dict = dict(params) if not isinstance(params, dict) else params

        if self._type == "step":
            self._duration = 0.0

        if self._type not in ("step", "robot_arm"):
            raise ValueError(
                f"Unknown transition type '{self._type}'. "
                f"Available: ['step', 'robot_arm']"
            )
        if self._duration < 0.0:
            raise ValueError(
                f"transition.duration_sec must be >= 0, got {self._duration}"
            )

        self._t_end: float = self._t_start + self._duration
        self._complete: bool = False

        if self._type == "robot_arm":
            self._init_robot_arm()

    # ------------------------------------------------------------------
    #  2-link robot arm
    # ------------------------------------------------------------------

    def _init_robot_arm(self) -> None:
        """Parse 2-link arm parameters and pre-compute swing-plane basis.

        Expected params::

            mass_1, arm_length_1, theta_1_0_deg, theta_1_f_deg,
            mass_2, arm_length_2, theta_2_0_deg, theta_2_f_deg,
            delay_sec,          # joint-2 starts this many seconds after joint-1
            pivot: [x,y,z],
            rotation_axis: [x,y,z],
            arm_dir_init: [x,y,z] | null
        """
        p = self._params

        # Link 1
        self._m1: float = float(p.get("mass_1", 5.0))
        self._L1: float = float(p.get("arm_length_1", 1.0))
        self._th1_0: float = np.deg2rad(float(p.get("theta_1_0_deg", 0.0)))
        self._th1_f: float = np.deg2rad(float(p.get("theta_1_f_deg", 120.0)))

        # Link 2
        self._m2: float = float(p.get("mass_2", 8.0))
        self._L2: float = float(p.get("arm_length_2", 1.2))
        self._th2_0: float = np.deg2rad(float(p.get("theta_2_0_deg", 0.0)))
        self._th2_f: float = np.deg2rad(float(p.get("theta_2_f_deg", -90.0)))

        # Delay for joint 2
        self._delay: float = float(p.get("delay_sec", 0.0))

        # Pivot & swing plane
        self._pivot = np.asarray(p.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
        axis = np.asarray(p.get("rotation_axis", [0, 0, 1]), dtype=float)
        axis = axis / np.linalg.norm(axis)
        self._axis = axis

        arm_dir_init = p.get("arm_dir_init", None)
        if arm_dir_init is not None:
            e1 = np.asarray(arm_dir_init, dtype=float)
            e1 = e1 - np.dot(e1, axis) * axis
            e1 /= np.linalg.norm(e1)
        else:
            ref = (
                np.array([1.0, 0.0, 0.0])
                if abs(axis[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            e1 = ref - np.dot(ref, axis) * axis
            e1 /= np.linalg.norm(e1)

        self._e1 = e1
        self._e2 = np.cross(axis, e1)

        # Pre-compute initial arm contribution (alpha=0) for relative change
        r1_0, r2_0 = self._arm_positions(self._th1_0, self._th1_0 + self._th2_0)
        self._arm_inertia_0 = (
            _point_mass_inertia(self._m1, r1_0)
            + _point_mass_inertia(self._m2, r2_0)
        )

        # Auto-compute post_inertia from final configuration (relative)
        r1_f, r2_f = self._arm_positions(self._th1_f, self._th1_f + self._th2_f)
        arm_inertia_f = (
            _point_mass_inertia(self._m1, r1_f)
            + _point_mass_inertia(self._m2, r2_f)
        )
        self._post = self._pre + (arm_inertia_f - self._arm_inertia_0)

    def _arm_positions(
        self, theta1: float, theta12: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (r1, r2) positions of the two point masses.

        Parameters
        ----------
        theta1 : float
            Absolute angle of link 1 in the swing plane.
        theta12 : float
            Absolute angle of link 2 tip = theta1 + theta2_relative.
        """
        dir1 = np.cos(theta1) * self._e1 + np.sin(theta1) * self._e2
        r1 = self._pivot + self._L1 * dir1

        dir2 = np.cos(theta12) * self._e1 + np.sin(theta12) * self._e2
        r2 = r1 + self._L2 * dir2
        return r1, r2

    def _joint_angles(self, alpha: float) -> tuple[float, float]:
        """Return (theta1, theta1+theta2) at normalised progress *alpha*.

        Joint 1 uses full alpha range.  Joint 2 accounts for ``delay_sec``:
        its own progress alpha2 is shifted so it starts later and still
        finishes at alpha=1.
        """
        s1 = _minimum_jerk(alpha)
        th1 = self._th1_0 + s1 * (self._th1_f - self._th1_0)

        if self._delay > 0.0 and self._duration > 0.0:
            delay_frac = self._delay / self._duration
            if alpha <= delay_frac:
                alpha2 = 0.0
            else:
                alpha2 = min((alpha - delay_frac) / (1.0 - delay_frac), 1.0)
        else:
            alpha2 = alpha

        s2 = _minimum_jerk(alpha2)
        th2_rel = self._th2_0 + s2 * (self._th2_f - self._th2_0)
        return th1, th1 + th2_rel

    def _compute_robot_arm(self, alpha: float) -> np.ndarray:
        th1, th12 = self._joint_angles(alpha)
        r1, r2 = self._arm_positions(th1, th12)
        arm_inertia = (
            _point_mass_inertia(self._m1, r1)
            + _point_mass_inertia(self._m2, r2)
        )
        return self._pre + (arm_inertia - self._arm_inertia_0)

    # ------------------------------------------------------------------
    #  Stateless computation
    # ------------------------------------------------------------------

    def compute_inertia(self, time_sec: float) -> np.ndarray:
        """Return (9,) inertia tensor at *time_sec* (no side effects)."""
        if not self._enabled or time_sec < self._t_start:
            return self._pre.copy()

        if self._type == "step":
            return self._post.copy()

        # robot_arm
        if self._duration <= 0.0:
            alpha = 1.0
        else:
            alpha = min((time_sec - self._t_start) / self._duration, 1.0)
        return self._compute_robot_arm(alpha)

    # ------------------------------------------------------------------
    #  Profile evaluation
    # ------------------------------------------------------------------

    def evaluate_profile(
        self, sim_time: float, dt: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate full inertia profile over ``[0, sim_time]``.

        Returns
        -------
        t : ndarray, shape (N,)
            Time vector [s].
        J_upper : ndarray, shape (N, 6)
            ``[Jxx, Jyy, Jzz, Jxy, Jyz, Jzx]`` per row.
        """
        t = np.arange(0.0, sim_time + dt / 2, dt)
        J_flat = np.array([self.compute_inertia(ti) for ti in t])
        J_upper = np.column_stack([
            J_flat[:, 0],  # Jxx (0,0)
            J_flat[:, 4],  # Jyy (1,1)
            J_flat[:, 8],  # Jzz (2,2)
            J_flat[:, 1],  # Jxy (0,1)
            J_flat[:, 5],  # Jyz (1,2)
            J_flat[:, 2],  # Jzx (0,2)
        ])
        return t, J_upper

    # ------------------------------------------------------------------
    #  Stateful API (simulation loop)
    # ------------------------------------------------------------------

    def get_inertia(self, time_sec: float) -> np.ndarray | None:
        """Return inertia at *time_sec*, or ``None`` if unchanged."""
        if not self._enabled or self._complete:
            return None

        if time_sec < self._t_start:
            return None

        if self._type == "step":
            self._complete = True
            return self._post.copy()

        # robot_arm
        if self._duration <= 0.0:
            alpha = 1.0
        else:
            alpha = min((time_sec - self._t_start) / self._duration, 1.0)

        inertia = self._compute_robot_arm(alpha)
        if alpha >= 1.0:
            self._complete = True
        return inertia

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def pre_inertia(self) -> np.ndarray:
        """Base (pre-transition) inertia, shape (9,)."""
        return self._pre.copy()

    @property
    def post_inertia(self) -> np.ndarray:
        """Final (post-transition) inertia, shape (9,).

        For physics models this is auto-computed from the final geometry.
        """
        return self._post.copy()

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def transition_start_sec(self) -> float | None:
        return self._t_start if self._enabled else None

    @property
    def transition_end_sec(self) -> float | None:
        return self._t_end if self._enabled else None

    @property
    def transition_type(self) -> str:
        return self._type

    @property
    def duration_sec(self) -> float:
        return self._duration
