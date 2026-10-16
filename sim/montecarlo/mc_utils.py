import numpy as np
from scipy.spatial.transform import Rotation


def random_inertia(inertia_range: list[float], max_angle_deg: float) -> np.ndarray:
    """Generate a physically valid random inertia matrix.
    
    Samples principal moments from uniform distribution, ensures triangle 
    inequality, then rotates the diagonal inertia by a random rotation.
    
    Args:
        inertia_range: [min, max] range for principal moments of inertia.
        max_angle_deg: Maximum rotation angle (deg) for each Euler axis.
        
    Returns:
        3x3 symmetric positive-definite inertia matrix.
    """
    if len(inertia_range) != 2:
        raise ValueError(f"inertia_range must have length 2, got {len(inertia_range)}")

    i_min, i_max = inertia_range
    if i_min <= 0.0 or i_max <= 0.0 or i_min >= i_max:
        raise ValueError(
            "inertia_range must satisfy 0 < min < max; "
            f"got min={i_min}, max={i_max}"
        )

    if not (0.0 <= max_angle_deg <= 180.0):
        raise ValueError(
            "max_angle_deg must be within [0, 180] for valid rotation sampling; "
            f"got {max_angle_deg}"
        )

    # Keep principal moments in sampled order (do not sort), then enforce the
    # physical triangle inequality for all three axes to avoid axis bias.
    while True:
        principal_I = np.random.uniform(i_min, i_max, 3)
        if np.all(principal_I < (principal_I.sum() - principal_I)):
            break

    if np.isclose(max_angle_deg, 180.0):
        # Haar-uniform on SO(3): best choice for orientation-isotropic dataset.
        R = Rotation.random().as_matrix()
    else:
        # Isotropic capped rotation: random axis on sphere + angle in [0, max].
        axis = np.random.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            raise RuntimeError("failed to sample non-zero axis for inertia rotation")
        axis /= axis_norm
        angle_rad = np.deg2rad(np.random.uniform(0.0, max_angle_deg))
        R = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    
    I_diag = np.diag(principal_I)
    I_final = R @ I_diag @ R.T    
        
    return I_final

def perturbed_inertia(
    nominal: np.ndarray, pct: float, off_diag_pct: float | None = None,
) -> np.ndarray:
    """Generate a perturbed inertia matrix around a nominal value.

    Perturbs diagonal elements by ±pct%, then samples off-diagonal elements
    as ±off_diag_pct% of the geometric mean of the corresponding diagonal elements.

    Args:
        nominal: 3x3 nominal inertia matrix.
        pct: Perturbation percentage for diagonal (e.g. 10 → ±10%).
        off_diag_pct: Perturbation percentage for off-diagonal. Defaults to pct.

    Returns:
        3x3 symmetric positive-definite perturbed inertia matrix.
    """
    diag_nom = np.diag(nominal)
    frac = pct / 100.0
    frac_od = (off_diag_pct if off_diag_pct is not None else pct) / 100.0
    max_tries = 1000

    for _ in range(max_tries):
        # Perturb diagonal elements
        diag_new = diag_nom * (1.0 + np.random.uniform(-frac, frac, size=3))

        # Sample off-diagonal as ±off_diag_pct% of geometric mean of neighboring diag
        J12 = np.random.uniform(-frac_od, frac_od) * np.sqrt(diag_new[0] * diag_new[1])
        J13 = np.random.uniform(-frac_od, frac_od) * np.sqrt(diag_new[0] * diag_new[2])
        J23 = np.random.uniform(-frac_od, frac_od) * np.sqrt(diag_new[1] * diag_new[2])

        I = np.array([
            [diag_new[0], J12, J13],
            [J12, diag_new[1], J23],
            [J13, J23, diag_new[2]],
        ], dtype=np.float64)

        # Check positive-definiteness
        eigvals = np.linalg.eigvalsh(I)
        if np.all(eigvals > 0):
            return I

    # Fallback: return nominal with perturbed diagonal only
    return np.diag(diag_new)


def random_omega(omega_max: float) -> np.ndarray:
    return np.random.uniform(-omega_max, omega_max, size=3)


def is_data_valid(data: dict) -> bool:
    """필수 메시지가 모두 있고, 비어있지 않고, NaN도 없는지 체크."""
    if "messages" not in data:
        return False
    msgs = data["messages"]

    required_msg_keys = [
        "SimpleNavigation.attOutMsg.sigma_BN",
        "SimpleNavigation.attOutMsg.omega_BN_B",
        "randomCtrl.cmdTorqueOutMsg.torqueRequestBody",
    ]

    for key in required_msg_keys:
        arr = msgs.get(key, None)
        if arr is None:
            return False
        if not isinstance(arr, np.ndarray):
            return False
        if arr.size == 0:
            return False
        if np.isnan(arr).any():
            return False

    return True

def mrp2quat_batch(sigma: np.ndarray) -> np.ndarray:
    """Convert MRP array (N, 3) to quaternion array (N, 4) [q0, q1, q2, q3].

    Works for both regular and shadow MRP sets.  When |σ| > 1 the resulting
    quaternion simply has q0 < 0, which is the valid antipodal representation.
    """
    sigma_sq = np.sum(sigma ** 2, axis=1, keepdims=True)
    denom = 1.0 + sigma_sq
    q0 = (1.0 - sigma_sq) / denom
    q_vec = 2.0 * sigma / denom
    return np.hstack([q0, q_vec])


def unwrap_quaternions(quats: np.ndarray) -> np.ndarray:
    """Unwrap quaternion trajectory to ensure continuity (avoid sign flips).
    
    Quaternions q and -q represent the same rotation, but sign discontinuities
    cause issues for neural network training. This function ensures that 
    consecutive quaternions have positive inner product (dot(q_t, q_{t-1}) >= 0).
    
    Args:
        quats: (T, 4) array of quaternions [q0, q1, q2, q3]
        
    Returns:
        (T, 4) array of unwrapped quaternions (modified in-place and returned)
    """
    quats = quats.copy()
    # Align q[0] to q_w >= 0 hemisphere (identity side)
    if quats[0, 0] < 0:
        quats[0] *= -1.0
    # Sequential continuity: ensure dot(q_t, q_{t-1}) >= 0
    for t in range(1, len(quats)):
        if np.dot(quats[t - 1], quats[t]) < 0:
            quats[t] *= -1.0
    return quats