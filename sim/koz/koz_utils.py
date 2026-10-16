import logging
import numpy as np
from typing import List, Optional

from scipy.spatial.transform import Rotation

from sim.koz.koz import KeepoutZone
from sim.utils.math_utils import random_quat, quat_error_rad

log = logging.getLogger(__name__)

def quat_min_koz_gap_deg(
    q: np.ndarray,
    boresight_B_list: np.ndarray,
    koz_object: List[KeepoutZone],
) -> float:
    """Return the smallest KOZ clearance angle [deg] across all boresights/KOZs.

    clearance = separation_angle(boresight, koz_axis) - koz_half_angle
    """
    q_norm = q / np.linalg.norm(q)
    w, x, y, z = q_norm
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ])

    min_gap_deg = np.inf
    for boresight_B in boresight_B_list:
        boresight_N = R @ boresight_B
        for koz in koz_object:
            angle_rad = np.arccos(np.clip(np.dot(boresight_N, koz.direction), -1, 1))
            sep_deg = np.degrees(angle_rad)
            gap_deg = sep_deg - koz.angle_deg
            if gap_deg < min_gap_deg:
                min_gap_deg = gap_deg
    return float(min_gap_deg)


def is_quat_safe(
    q: np.ndarray,
    boresight_B_list: np.ndarray,
    koz_object: List[KeepoutZone],
    safety_margin_deg: float = 10.0,
    margin_cap_deg: Optional[float] = None,
    koz_proximity_deg: Optional[float] = None,
) -> bool:
    min_gap_deg = quat_min_koz_gap_deg(q, boresight_B_list, koz_object)
    if min_gap_deg <= safety_margin_deg:
        return False
    if margin_cap_deg is not None and min_gap_deg >= margin_cap_deg:
        return False
    if koz_proximity_deg is not None and min_gap_deg > koz_proximity_deg:
        return False
    return True


def generate_safe_quat(
    boresight_B_list: np.ndarray,
    koz_object: List[KeepoutZone],
    max_attempts: int = 1000,
    safety_margin_deg: float = 10.0,
    margin_cap_deg: Optional[float] = None,
    koz_proximity_deg: Optional[float] = None,
) -> np.ndarray:
    for _ in range(max_attempts):
        q = random_quat()
        if is_quat_safe(q, boresight_B_list, koz_object, safety_margin_deg, margin_cap_deg, koz_proximity_deg):
            return q
    for reduced_margin in [5.0, 2.0, 0.0]:
        for _ in range(max_attempts // 2):
            q = random_quat()
            if is_quat_safe(q, boresight_B_list, koz_object, reduced_margin, margin_cap_deg, koz_proximity_deg):
                log.warning(
                    f"Fallback to quaternion with reduced margin: {reduced_margin} degrees"
                )
                return q
    q_I = np.array([1.0, 0.0, 0.0, 0.0])
    if is_quat_safe(q_I, boresight_B_list, koz_object, 0.0, margin_cap_deg, koz_proximity_deg):
        log.warning("Fallback to identity quaternion")
        return q_I
    raise RuntimeError(f"Failed to generate KOZ-safe quaternion after {max_attempts} attempts")


def generate_quat_pair(
    boresight_list: List[dict],
    koz_object: List[KeepoutZone],
    min_angle_deg: float = 120.0,
    max_angle_deg: float = 180.0,
    max_attempts: int = 10000,
    safety_margin_deg: float = 10.0,
    margin_cap_deg: Optional[float] = None,
    safety_margin_deg_init: Optional[float] = None,
    safety_margin_deg_goal: Optional[float] = None,
    koz_proximity_deg_init: Optional[float] = None,
    koz_proximity_deg_goal: Optional[float] = None,
    slerp_min_gap_deg: Optional[float] = None,
    slerp_max_gap_deg: Optional[float] = None,
    slerp_n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    boresight_B_list = np.array([bs['vec'] for bs in boresight_list], dtype=float)
    boresight_B_list = boresight_B_list / np.linalg.norm(boresight_B_list, axis=1, keepdims=True)

    margin_init = safety_margin_deg_init if safety_margin_deg_init is not None else safety_margin_deg
    margin_goal = safety_margin_deg_goal if safety_margin_deg_goal is not None else safety_margin_deg

    check_slerp = slerp_min_gap_deg is not None or slerp_max_gap_deg is not None

    log.info(f"Safety margin: init={margin_init}°, goal={margin_goal}°")
    if koz_proximity_deg_init is not None or koz_proximity_deg_goal is not None:
        log.info(f"KOZ proximity: init={koz_proximity_deg_init}°, goal={koz_proximity_deg_goal}°")
    if check_slerp:
        log.info(f"SLERP gap constraint: [{slerp_min_gap_deg}, {slerp_max_gap_deg}]°")

    log.info("Generating q_init...")
    q_init = generate_safe_quat(
        boresight_B_list,
        koz_object,
        safety_margin_deg=margin_init,
        margin_cap_deg=margin_cap_deg,
        koz_proximity_deg=koz_proximity_deg_init,
    )
    log.info(f"q_init generated: {[round(x, 4) for x in q_init]}")
    log.info(f"  min KOZ gap: {quat_min_koz_gap_deg(q_init, boresight_B_list, koz_object):.2f}°")

    min_angle_rad = np.deg2rad(min_angle_deg)
    max_angle_rad = np.deg2rad(max_angle_deg)
    log.info(f"Generating q_goal... (angle diff: [{min_angle_deg}, {max_angle_deg}] degrees)")

    for attempt in range(max_attempts):
        q_goal = generate_safe_quat(
            boresight_B_list,
            koz_object,
            safety_margin_deg=margin_goal,
            margin_cap_deg=margin_cap_deg,
            koz_proximity_deg=koz_proximity_deg_goal,
        )
        angle_diff_rad = quat_error_rad(q_init, q_goal)
        angle_diff_deg = np.rad2deg(angle_diff_rad)

        if angle_diff_rad < min_angle_rad or angle_diff_rad > max_angle_rad:
            if (attempt + 1) % 100 == 0:
                log.debug(f"Attempt {attempt + 1}/{max_attempts}... angle diff: {angle_diff_deg:.2f}°")
            continue

        if check_slerp:
            per_bs_gaps = slerp_min_koz_gap_per_boresight(
                q_init, q_goal, boresight_B_list, koz_object, n_samples=slerp_n_samples,
            )
            # Every boresight must individually satisfy the constraint
            ok = True
            for gap in per_bs_gaps:
                if slerp_min_gap_deg is not None and gap < slerp_min_gap_deg:
                    ok = False
                    break
                if slerp_max_gap_deg is not None and gap > slerp_max_gap_deg:
                    ok = False
                    break
            if not ok:
                if (attempt + 1) % 100 == 0:
                    gaps_str = ", ".join(f"{g:.2f}" for g in per_bs_gaps)
                    log.debug(f"Attempt {attempt + 1}/{max_attempts}... per-boresight SLERP gaps: [{gaps_str}]°")
                continue
            gaps_str = ", ".join(f"{g:.2f}" for g in per_bs_gaps)
            log.info(f"SLERP min KOZ gap per boresight: [{gaps_str}]°")

        log.info(f"q_goal generated (attempt {attempt + 1}): {[round(x, 4) for x in q_goal]}")
        log.info(f"Angle difference: {angle_diff_deg:.2f} degrees")
        return q_init, q_goal

    raise RuntimeError(
        f"Failed to generate q_goal: Could not find a safe quaternion that differs from q_init "
        f"by at least {min_angle_deg} degrees after {max_attempts} attempts."
    )


def slerp_min_koz_gap_per_boresight(
    q_init: np.ndarray,
    q_goal: np.ndarray,
    boresight_B_list: np.ndarray,
    koz_object: List[KeepoutZone],
    n_samples: int = 50,
) -> np.ndarray:
    """Return per-boresight minimum KOZ gap [deg] along the SLERP path.

    Returns:
        (N_boresight,) array where each element is the minimum KOZ gap
        for that boresight across the entire SLERP path.
    """
    r_init = Rotation.from_quat([q_init[1], q_init[2], q_init[3], q_init[0]])  # scalar-last
    r_goal = Rotation.from_quat([q_goal[1], q_goal[2], q_goal[3], q_goal[0]])

    from scipy.spatial.transform import Slerp
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_init, r_goal]))
    ts = np.linspace(0.0, 1.0, n_samples)

    n_bs = len(boresight_B_list)
    min_gaps = np.full(n_bs, np.inf)

    for t in ts:
        q_xyzw = slerp(t).as_quat()  # scalar-last
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        q_norm = q_wxyz / np.linalg.norm(q_wxyz)
        w, x, y, z = q_norm
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
        ])
        for i, bs_B in enumerate(boresight_B_list):
            bs_N = R @ bs_B
            for koz in koz_object:
                angle_deg = np.degrees(np.arccos(np.clip(np.dot(bs_N, koz.direction), -1, 1)))
                gap = angle_deg - koz.angle_deg
                if gap < min_gaps[i]:
                    min_gaps[i] = gap

    return min_gaps


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2.  Convention: scalar-first [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def perturb_quaternion(q_base: np.ndarray, max_angle_deg: float) -> np.ndarray:
    """Apply a small random rotation to a base quaternion.

    Samples a random rotation axis uniformly on the unit sphere and a
    rotation angle uniformly in [0, max_angle_deg], then composes with
    *q_base*.

    Args:
        q_base: (4,) base quaternion [w, x, y, z] (scalar-first).
        max_angle_deg: Maximum perturbation angle [deg].

    Returns:
        (4,) perturbed quaternion, unit-normalized.
    """
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle_rad = np.deg2rad(np.random.uniform(0.0, max_angle_deg))

    # scipy uses scalar-last [x, y, z, w]
    delta_q_xyzw = Rotation.from_rotvec(axis * angle_rad).as_quat()
    delta_q = np.array([delta_q_xyzw[3], delta_q_xyzw[0],
                        delta_q_xyzw[1], delta_q_xyzw[2]])

    q_perturbed = _quat_multiply(delta_q, q_base)
    q_perturbed /= np.linalg.norm(q_perturbed)
    return q_perturbed
