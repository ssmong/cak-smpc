import torch


def quat_quadratic(q: torch.Tensor) -> torch.Tensor:
    """Compute unique entries of q ⊗ q (quaternion outer product).

    Maps (..., 4) → (..., 10). Automatically sign-invariant: (-q)⊗(-q) = q⊗q.
    Rotation matrix entries are linear combinations of these features,
    so no rotation information is lost.

    Args:
        q: (..., 4) quaternion [w, x, y, z].
    Returns:
        (..., 10) unique entries [w², wx, wy, wz, x², xy, xz, y², yz, z²].
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        w*w, w*x, w*y, w*z,
        x*x, x*y, x*z,
        y*y, y*z,
        z*z,
    ], dim=-1)
