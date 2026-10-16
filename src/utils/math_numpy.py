import numpy as np

def quat_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / n).astype(np.float64)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation (shortest path on S³)."""
    q0, q1 = quat_norm(q0), quat_norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_norm((1.0 - t) * q0 + t * q1)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )

def quat2rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_norm(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(nv, np.clip(q[0], 1e-12, None))
    axis = v / nv
    return (angle * axis).astype(np.float64)

def quat_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    Quaternion error: q_err = q^* ⊗ q_ref, returns vector part only [3].
    """
    q_norm = quat_norm(q)
    q_ref_norm = quat_norm(q_ref)
    s1, v1 = q_norm[0], -q_norm[1:4]
    s2, v2 = q_ref_norm[0], q_ref_norm[1:4]
    q_err_v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return q_err_v.astype(np.float64)


def rotvec2quat(rv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rv / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def attitude_error_rotvec(q_nom: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Rotation-vector attitude error (exact, handles large angles)."""
    q_err = quat_mul(quat_norm(q_true), quat_conj(quat_norm(q_nom)))
    return quat2rotvec(q_err)


def attitude_error_rotvec_batch(q_nom: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Batch rotation-vector attitude error.

    Args:
        q_nom: (M, 4) nominal quaternions.
        q_true: (M, 4) true quaternions.
    Returns:
        rotvec: (M, 3) rotation vector errors.
    """
    # Normalize
    q_nom = q_nom / np.linalg.norm(q_nom, axis=1, keepdims=True).clip(1e-12)
    q_true = q_true / np.linalg.norm(q_true, axis=1, keepdims=True).clip(1e-12)

    # q_err = q_true * conj(q_nom)
    q_conj = q_nom.copy()
    q_conj[:, 1:] = -q_conj[:, 1:]

    w1, x1, y1, z1 = q_true[:, 0], q_true[:, 1], q_true[:, 2], q_true[:, 3]
    w2, x2, y2, z2 = q_conj[:, 0], q_conj[:, 1], q_conj[:, 2], q_conj[:, 3]
    q_err = np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

    # Ensure positive scalar part
    sign = np.where(q_err[:, 0] < 0, -1.0, 1.0)[:, None]
    q_err = q_err * sign

    # quat2rotvec batch
    v = q_err[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(nv, q_err[:, 0].clip(1e-12))

    rotvec = np.zeros_like(v)
    mask = nv > 1e-12
    rotvec[mask] = (angle[mask] / nv[mask])[:, None] * v[mask]
    return rotvec


def apply_rotvec_to_quat(q_nom: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    dq = rotvec2quat(dtheta)
    q_new = quat_mul(dq, quat_norm(q_nom))
    return quat_norm(q_new)


def quat_jacobian_wrt_rotvec(q_nom: np.ndarray, dtheta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian dq/d(dtheta) at current (q_nom, dtheta)."""
    jac = np.zeros((4, 3), dtype=np.float64)
    for i in range(3):
        dt = np.zeros(3, dtype=np.float64)
        dt[i] = eps
        qp = apply_rotvec_to_quat(q_nom, dtheta + dt)
        qm = apply_rotvec_to_quat(q_nom, dtheta - dt)
        jac[:, i] = (qp - qm) / (2.0 * eps)
    return jac


def attitude_dyn_ct(x: np.ndarray, u: np.ndarray, j: np.ndarray, j_inv: np.ndarray) -> np.ndarray:
    """Continuous-time attitude dynamics (quaternion + angular velocity)."""
    q0, q1, q2, q3 = x[:4]
    w = x[4:]
    q_dot = 0.5 * np.array(
        [
            -q1 * w[0] - q2 * w[1] - q3 * w[2],
            q0 * w[0] - q3 * w[1] + q2 * w[2],
            q3 * w[0] + q0 * w[1] - q1 * w[2],
            -q2 * w[0] + q1 * w[1] + q0 * w[2],
        ],
        dtype=np.float64,
    )
    jw = j @ w
    w_dot = j_inv @ (-np.cross(w, jw) + u)
    return np.concatenate([q_dot, w_dot])


def _omega_matrix(w: np.ndarray) -> np.ndarray:
    """4x4 Omega matrix for quaternion kinematics: qdot = 0.5 * Omega(w) @ q."""
    wx, wy, wz = w
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0],
    ], dtype=np.float64)


def _quat_w_matrix(q: np.ndarray) -> np.ndarray:
    """4x3 matrix such that qdot = 0.5 * Q(q) @ w."""
    q0, q1, q2, q3 = q
    return 0.5 * np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0],
    ], dtype=np.float64)


def attitude_dyn_ct_jac(
    x: np.ndarray, u: np.ndarray, j: np.ndarray, j_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous-time attitude dynamics with analytical Jacobians.

    Returns:
        xdot: (7,) state derivative.
        df_dx: (7, 7) Jacobian w.r.t. state.
        df_du: (7, 3) Jacobian w.r.t. control.
    """
    q = x[:4]
    w = x[4:]

    # --- xdot ---
    Om = _omega_matrix(w)
    q_dot = 0.5 * Om @ q
    jw = j @ w
    w_dot = j_inv @ (-np.cross(w, jw) + u)
    xdot = np.concatenate([q_dot, w_dot])

    # --- df/dx (7x7) ---
    df_dx = np.zeros((7, 7), dtype=np.float64)

    # dq_dot/dq = 0.5 * Omega(w)
    df_dx[:4, :4] = 0.5 * Om

    # dq_dot/dw: qdot = 0.5 * Q(q) @ w, so dq_dot/dw = Q(q) (already has 0.5)
    df_dx[:4, 4:] = _quat_w_matrix(q)

    # dw_dot/dq = 0
    # dw_dot/dw: w_dot = J_inv @ (u - w x Jw)
    #   d(w x Jw)/dw = [w]_x J + [Jw]_x  (where [v]_x is skew-symmetric)
    #   But: d(w x Jw)/dw = [w]_x J - [Jw]_x  ... let me be careful.
    #   (a x b) → d/da(a x b) = -[b]_x,  d/db(a x b) = [a]_x
    #   So d(w x Jw)/dw = -[Jw]_x + [w]_x @ J
    skew_jw = _skew(jw)
    skew_w = _skew(w)
    df_dx[4:, 4:] = j_inv @ (skew_jw - skew_w @ j)

    # --- df/du (7x3) ---
    df_du = np.zeros((7, 3), dtype=np.float64)
    df_du[4:, :] = j_inv

    return xdot, df_dx, df_du


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix [v]_x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def _attitude_dyn_dt_jac_single(
    x: np.ndarray, u: np.ndarray, dt: float, j: np.ndarray, j_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single RK4 step with analytical Jacobians (no quat normalization)."""
    I7 = np.eye(7, dtype=np.float64)

    k1, F1x, F1u = attitude_dyn_ct_jac(x, u, j, j_inv)

    x2 = x + 0.5 * dt * k1
    k2, F2x, F2u = attitude_dyn_ct_jac(x2, u, j, j_inv)
    dk2_dx = F2x @ (I7 + 0.5 * dt * F1x)
    dk2_du = F2u + F2x @ (0.5 * dt * F1u)

    x3 = x + 0.5 * dt * k2
    k3, F3x, F3u = attitude_dyn_ct_jac(x3, u, j, j_inv)
    dk3_dx = F3x @ (I7 + 0.5 * dt * dk2_dx)
    dk3_du = F3u + F3x @ (0.5 * dt * dk2_du)

    x4 = x + dt * k3
    k4, F4x, F4u = attitude_dyn_ct_jac(x4, u, j, j_inv)
    dk4_dx = F4x @ (I7 + dt * dk3_dx)
    dk4_du = F4u + F4x @ (dt * dk3_du)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    A = I7 + (dt / 6.0) * (F1x + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx)
    B = (dt / 6.0) * (F1u + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du)

    return x_next, A, B


def attitude_dyn_dt_jac(
    x: np.ndarray, u: np.ndarray, dt: float, j: np.ndarray, j_inv: np.ndarray,
    steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4 discrete-time attitude dynamics with analytical Jacobians.

    Differentiates through each RK4 stage using the chain rule.
    When *steps* > 1, chains sub-step Jacobians via the product rule.

    Returns:
        x_next: (7,) next state.
        A: (7, 7) Jacobian dx_next/dx.
        B: (7, 3) Jacobian dx_next/du.
    """
    sub_dt = dt / steps
    x_cur = x.copy()
    A_total = np.eye(7, dtype=np.float64)
    B_total = np.zeros((7, 3), dtype=np.float64)

    for _ in range(steps):
        x_cur, A_step, B_step = _attitude_dyn_dt_jac_single(
            x_cur, u, sub_dt, j, j_inv,
        )
        # Chain rule: B_total = A_step @ B_total + B_step
        B_total = A_step @ B_total + B_step
        A_total = A_step @ A_total

    x_cur[:4] = quat_norm(x_cur[:4])
    return x_cur, A_total, B_total


def attitude_dyn_dt(x: np.ndarray, u: np.ndarray, dt: float, j: np.ndarray, j_inv: np.ndarray, steps: int = 1) -> np.ndarray:
    """RK4 discrete-time attitude dynamics with sub-stepping."""
    sub_dt = dt / steps
    x_next = x.copy()
    for _ in range(steps):
        k1 = attitude_dyn_ct(x_next, u, j, j_inv)
        k2 = attitude_dyn_ct(x_next + 0.5 * sub_dt * k1, u, j, j_inv)
        k3 = attitude_dyn_ct(x_next + 0.5 * sub_dt * k2, u, j, j_inv)
        k4 = attitude_dyn_ct(x_next + sub_dt * k3, u, j, j_inv)
        x_next = x_next + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[:4] = quat_norm(x_next[:4])
    return x_next


# ---------------------------------------------------------------------------
#   Batch (vectorized) attitude dynamics for Monte Carlo propagation
# ---------------------------------------------------------------------------

def attitude_dyn_ct_batch(
    x: np.ndarray, u: np.ndarray, j: np.ndarray, j_inv: np.ndarray,
) -> np.ndarray:
    """Continuous-time attitude dynamics for a batch of M states.

    Args:
        x:     (M, 7)  states  [q0, q1, q2, q3, w1, w2, w3]
        u:     (3,) or (M, 3)  control torque
        j:     (3, 3)  inertia matrix
        j_inv: (3, 3)  inverse inertia matrix
    Returns:
        x_dot: (M, 7)
    """
    q0, q1, q2, q3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    w1, w2, w3 = x[:, 4], x[:, 5], x[:, 6]

    q_dot = np.empty((x.shape[0], 4))
    q_dot[:, 0] = 0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)
    q_dot[:, 1] = 0.5 * ( q0 * w1 - q3 * w2 + q2 * w3)
    q_dot[:, 2] = 0.5 * ( q3 * w1 + q0 * w2 - q1 * w3)
    q_dot[:, 3] = 0.5 * (-q2 * w1 + q1 * w2 + q0 * w3)

    w = x[:, 4:]                                      # (M, 3)
    Jw = (j @ w.T).T                                   # (M, 3)
    cross = np.cross(w, Jw)                            # (M, 3)
    u_2d = np.atleast_2d(u)                            # (1, 3) or (M, 3)
    w_dot = (j_inv @ (-cross + u_2d).T).T              # (M, 3)

    return np.concatenate([q_dot, w_dot], axis=1)      # (M, 7)


def attitude_dyn_dt_batch(
    x: np.ndarray, u: np.ndarray, dt: float,
    j: np.ndarray, j_inv: np.ndarray,
    steps: int = 1,
) -> np.ndarray:
    """RK4 discrete-time attitude dynamics for a batch of M states.

    Args:
        x:     (M, 7)  states
        u:     (3,) or (M, 3)  control torque
        dt:    float   total integration timestep [s]
        j:     (3, 3)  inertia matrix
        j_inv: (3, 3)  inverse inertia matrix
        steps: int     number of RK4 sub-steps (matches CasADi integrator fidelity)
    Returns:
        x_next: (M, 7) propagated states with normalized quaternions
    """
    sub_dt = dt / steps
    for _ in range(steps):
        k1 = attitude_dyn_ct_batch(x, u, j, j_inv)
        k2 = attitude_dyn_ct_batch(x + 0.5 * sub_dt * k1, u, j, j_inv)
        k3 = attitude_dyn_ct_batch(x + 0.5 * sub_dt * k2, u, j, j_inv)
        k4 = attitude_dyn_ct_batch(x + sub_dt * k3, u, j, j_inv)
        x = x + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Normalize quaternion part
    q_norms = np.linalg.norm(x[:, :4], axis=1, keepdims=True)
    np.maximum(q_norms, 1e-8, out=q_norms)
    x[:, :4] /= q_norms
    return x


# ---------------------------------------------------------------------------
#   Inertia 6-vector <-> 3x3 symmetric matrix
#   Order: [J11, J12, J13, J22, J23, J33]  (row-major upper triangle)
# ---------------------------------------------------------------------------

def j_vec2mat(j6: np.ndarray) -> np.ndarray:
    """[J11, J12, J13, J22, J23, J33] → 3×3 symmetric matrix."""
    return np.array([
        [j6[0], j6[1], j6[2]],
        [j6[1], j6[3], j6[4]],
        [j6[2], j6[4], j6[5]],
    ], dtype=np.float64)


def j_mat_to_vec(J: np.ndarray) -> np.ndarray:
    """3×3 symmetric → [J11, J12, J13, J22, J23, J33]."""
    return np.array([J[0, 0], J[0, 1], J[0, 2],
                     J[1, 1], J[1, 2], J[2, 2]], dtype=np.float64)