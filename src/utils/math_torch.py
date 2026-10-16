import torch
import torch.nn.functional as F


def slerp_batch(q1: torch.Tensor, q2: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
    """Batch spherical linear interpolation on S³.

    Args:
        q1: (..., 4) unit quaternions (start).
        q2: (..., 4) unit quaternions (end).
        t:  (..., 1) or scalar interpolation parameter in [0, 1].

    Returns:
        (..., 4) unit quaternion at fraction *t* along the shortest great-circle arc.
    """
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    dot = (q1 * q2).sum(-1, keepdim=True)
    # Shortest path: flip q2 when dot < 0
    q2 = torch.where(dot < 0, -q2, q2)
    dot = dot.abs().clamp(0.0, 1.0)

    # theta = angle between q1, q2
    omega = torch.acos(dot.clamp(max=1.0 - 1e-6))
    sin_omega = torch.sin(omega)

    # For nearly identical quaternions (theta ≈ 0): fall back to lerp
    close = (sin_omega < 1e-6)
    safe_sin = torch.where(close, torch.ones_like(sin_omega), sin_omega)

    s1 = torch.where(close, 1.0 - t, torch.sin((1.0 - t) * omega) / safe_sin)
    s2 = torch.where(close, t, torch.sin(t * omega) / safe_sin)

    return F.normalize(s1 * q1 + s2 * q2, dim=-1)


def quat_angle_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / torch.clamp(q1.norm(dim=-1, keepdim=True), min=1e-8)
    q2 = q2 / torch.clamp(q2.norm(dim=-1, keepdim=True), min=1e-8)
    
    dot = (q1 * q2).sum(dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2 * torch.acos(dot)

def quat_mul(q1: torch.Tensor, 
                  q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def canonicalize_quat(x: torch.Tensor, quat_slice: slice = slice(0, 4)) -> torch.Tensor:
    """Canonicalize quaternion to q_w >= 0 hemisphere.

    Flips the sign of the quaternion component (indices given by *quat_slice*)
    whenever the scalar part (first element of the slice) is negative.
    Non-quaternion state components (e.g. angular velocity) are left unchanged.

    Uses stop-gradient on the sign to avoid gradient discontinuity at q_w = 0.

    Args:
        x: (..., state_dim) tensor containing quaternion + other state.
        quat_slice: slice identifying quaternion indices.  Default ``slice(0, 4)``.
    Returns:
        Tensor of same shape with quaternion in q_w >= 0 hemisphere.
    """
    q = x[..., quat_slice]                          # (..., 4)
    sign = (q[..., 0:1] >= 0).float() * 2.0 - 1.0  # (..., 1): +1 or -1
    q_canon = q * sign.detach()                      # stop gradient through sign
    out = x.clone()
    out[..., quat_slice] = q_canon
    return out