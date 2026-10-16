import logging
import numpy as np
import casadi as ca
from typing import List

log = logging.getLogger(__name__)

def quat_error_ca(q: ca.SX, q_ref: ca.SX) -> ca.SX:
    """
    Quaternion error: q_err = q_ref^* ⊗ q, returns vector part only [3].
    """
    s1, v1 = q_ref[0], -q_ref[1:4]
    s2, v2 = q[0], q[1:4]
    q_err_v = s1 * v2 + s2 * v1 + ca.cross(v1, v2)
    return q_err_v

def random_quats(n: int) -> np.ndarray:
    u = np.random.uniform(0, 1, (3, n))
    
    r1 = np.sqrt(1 - u[0])
    r2 = np.sqrt(u[0])
    t1 = 2 * np.pi * u[1]
    t2 = 2 * np.pi * u[2]

    return np.stack([
        r1 * np.sin(t1),
        r1 * np.cos(t1),
        r2 * np.sin(t2),
        r2 * np.cos(t2)
    ], axis=1)

def random_quat() -> np.ndarray:
    return random_quats(1)[0]

def quat_error_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1_norm, q2_norm))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot)
    return angle_rad

def quat2boresight_batch(q_traj: np.ndarray, boresight_B: np.ndarray) -> np.ndarray:
    """
    Compute the inertial-frame boresight trajectory from the quaternion trajectory.

    Args:
        q_traj: (N, 4) quaternion trajectory [w, x, y, z]
        boresight_B: (3,) body-frame boresight unit vector
        
    Returns:
        boresight_N_traj: (N, 3) inertial-frame boresight trajectory
    """
    boresight_B = np.asarray(boresight_B, dtype=float).flatten()
    boresight_B = boresight_B / np.linalg.norm(boresight_B)
    boresight_vecs = []
    for q in q_traj:
        q_norm = q / np.linalg.norm(q)
        w, x, y, z = q_norm
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        boresight_vecs.append(R @ boresight_B)
    return np.array(boresight_vecs)