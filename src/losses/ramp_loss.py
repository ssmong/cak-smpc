"""
RAMP-Net loss functions: Physics Loss (Lp) + Data Loss (Ld).

Reference:
    Sanyal & Roy, "RAMP-Net: A Robust Adaptive MPC for Quadrotors via
    Physics-informed Neural Network", ICRA 2023, Section III-C, IV-B.

Physics Loss (Eq. 5, 10):
    Lp = (1/|P|) Σ ||dϕ/dt(x_k, u_k; θ) - (f_bar + f_hat)(ϕ(x_k, u_k, t; θ), u_k)||²

Data Loss (Eq. 7):
    Ld = (1/|D|) Σ ||ϕ_T(x_k, u_k; θ) - y_k||²
"""

import torch

from src.model.ramp_net import RampNet


def spacecraft_nominal_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    J_diag: torch.Tensor,
) -> torch.Tensor:
    """
    Compute nominal spacecraft attitude dynamics: x_dot = f_bar(x, u).

    Ported from sim/controller/nmpc.py:_attitude_dynamics (CasADi → PyTorch).

    State x = [q0, q1, q2, q3, w1, w2, w3]
    Control u = [τ1, τ2, τ3]

    Quaternion kinematics: q_dot = 0.5 * Ω(ω) * q
    Euler equation:        ω_dot = J⁻¹(-ω × Jω + u)

    Args:
        x: State tensor [B, 7]
        u: Control tensor [B, 3]
        J_diag: Diagonal inertia [3] or per-sample [B, 3] (Jx, Jy, Jz)

    Returns:
        x_dot: State derivative [B, 7]
    """
    q = x[:, :4]   # [B, 4]
    w = x[:, 4:]   # [B, 3]

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w1, w2, w3 = w[:, 0], w[:, 1], w[:, 2]

    # Quaternion kinematics: q_dot = 0.5 * Ω(ω) * q
    q0_dot = 0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)
    q1_dot = 0.5 * ( q0 * w1 - q3 * w2 + q2 * w3)
    q2_dot = 0.5 * ( q3 * w1 + q0 * w2 - q1 * w3)
    q3_dot = 0.5 * (-q2 * w1 + q1 * w2 + q0 * w3)

    q_dot = torch.stack([q0_dot, q1_dot, q2_dot, q3_dot], dim=-1)  # [B, 4]

    # Euler equation: J * w_dot = -w × (J * w) + u
    # Support both shared [3] and per-sample [B, 3] inertia
    if J_diag.dim() == 1:
        Jx, Jy, Jz = J_diag[0], J_diag[1], J_diag[2]
        Jw = torch.stack([Jx * w1, Jy * w2, Jz * w3], dim=-1)
        J_inv = torch.tensor([1.0 / Jx, 1.0 / Jy, 1.0 / Jz],
                             device=x.device, dtype=x.dtype)
    else:
        Jw = J_diag * w  # [B, 3]
        J_inv = 1.0 / J_diag  # [B, 3]

    # Cross product: w × Jw
    cross = torch.stack([
        w2 * Jw[:, 2] - w3 * Jw[:, 1],
        w3 * Jw[:, 0] - w1 * Jw[:, 2],
        w1 * Jw[:, 1] - w2 * Jw[:, 0],
    ], dim=-1)  # [B, 3]

    w_dot = J_inv * (-cross + u)  # [B, 3]

    return torch.cat([q_dot, w_dot], dim=-1)  # [B, 7]


def spacecraft_dynamics_with_uncertainty(
    x: torch.Tensor,
    u: torch.Tensor,
    J_diag: torch.Tensor,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Compute dynamics with parametric uncertainty: f_bar(x,u) + f_hat.

    Following the paper (Eq. 9-10, Section IV-B, V-A), f_hat is an additive
    noise vector sampled from N(0, σ²I) applied to the dynamics output:

        f(x,u) = f_bar(x,u) + f_hat,   f_hat ~ N(0, σ²I_n)

    This acts as a regularizer in the physics loss, preventing the PINN from
    overfitting to the exact nominal dynamics and enforcing robustness to
    parametric model mismatch.

    Reference: RAMP-Net Section IV-B (Eq. 9-10), Section V-A:
        "N(0, Σ), where Σ is a constant unit diagonal covariance matrix"

    Args:
        x: State tensor [B, 7]
        u: Control tensor [B, 3]
        J_diag: Nominal diagonal inertia [3]
        noise_std: Standard deviation σ for additive dynamics noise
        generator: Optional torch.Generator for deterministic noise within
                   L-BFGS closure calls. If None, uses global RNG.

    Returns:
        x_dot: State derivative with parametric uncertainty [B, 7]
    """
    # Nominal dynamics: f_bar(x, u)
    f_bar = spacecraft_nominal_dynamics(x, u, J_diag)

    # Additive parametric uncertainty: f_hat ~ N(0, σ²I_7)
    f_hat = torch.randn(
        f_bar.shape, device=f_bar.device, dtype=f_bar.dtype, generator=generator,
    ) * noise_std

    return f_bar + f_hat


def physics_loss(
    model: RampNet,
    x_phys: torch.Tensor,
    u_phys: torch.Tensor,
    J_diag: torch.Tensor,
    noise_std: float,
    dt: float,
    rng_seed: int | None = None,
) -> torch.Tensor:
    """
    Physics loss Lp (Eq. 10 in RAMP-Net).

    Lp = (1/|P|) Σ ||dϕ/dt - (f_bar + f_hat)||²

    Uses torch.autograd.grad to compute dϕ/dt w.r.t. the time variable t.

    Args:
        model: RampNet PINN model
        x_phys: Collocation point states [|P|, 7]
        u_phys: Collocation point controls [|P|, 3]
        J_diag: Nominal diagonal inertia [3]
        noise_std: Parametric uncertainty std
        dt: Time horizon T
        rng_seed: Optional seed for deterministic noise/time sampling within
                  a single L-BFGS step. Each epoch should use a different seed.
                  If None, uses global RNG (old behavior).

    Returns:
        Scalar physics loss
    """
    batch_size = x_phys.shape[0]

    # Create a local generator for deterministic sampling within L-BFGS closure.
    # This ensures all closure calls within one optimizer.step() produce
    # identical t and f_hat noise, satisfying L-BFGS's deterministic function
    # assumption. Noise diversity is preserved across epochs (different rng_seed
    # per epoch) and across samples (B independent draws per call).
    if rng_seed is not None:
        gen = torch.Generator(device=x_phys.device).manual_seed(rng_seed)
    else:
        gen = None

    # Create time variable with requires_grad for autograd
    # OLD: L-BFGS closure 반복 호출 시 매번 다른 t가 생성됨
    # t = torch.rand(batch_size, 1, device=x_phys.device, dtype=x_phys.dtype) * dt
    t = torch.rand(
        batch_size, 1, device=x_phys.device, dtype=x_phys.dtype, generator=gen,
    ) * dt
    t.requires_grad_(True)

    # Forward pass through PINN
    x_pred = model(x_phys, u_phys, t)  # [B, 7]

    # Compute dϕ/dt using autograd (paper's approach).
    # We need per-output-dim gradients ∂ϕ_i/∂t for each of 7 output dims.
    dphi_dt_list = []
    for i in range(model.state_dim):
        grad_i = torch.autograd.grad(
            outputs=x_pred[:, i].sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, 1]
        dphi_dt_list.append(grad_i)
    dphi_dt = torch.cat(dphi_dt_list, dim=-1)  # [B, 7]

    # Compute target: f_bar + f_hat (with parametric uncertainty)
    # Evaluate dynamics at the PINN's predicted state x_pred (= ϕ(x₀, u, t; θ)),
    # not at the initial/collocation state x_phys.
    # This follows the standard PINN ODE residual: ∂ϕ/∂t = f(ϕ, u)
    # consistent with the paper's Eq. (5): L_p = MSE(ẋ(t), f(x(t), u(t)))
    # where x(t) is the state at time t, not the initial state x(0).
    # OLD: f를 초기 상태 x_phys에서 평가 (t=0에서만 정확, t>0에서는 ODE 잔차 불일치)
    # f_target = spacecraft_dynamics_with_uncertainty(
    #     x_phys, u_phys, J_diag, noise_std, generator=gen,
    # )
    f_target = spacecraft_dynamics_with_uncertainty(
        x_pred, u_phys, J_diag, noise_std, generator=gen,
    )

    # MSE loss
    return torch.mean((dphi_dt - f_target) ** 2)


def data_loss(
    model: RampNet,
    x_data: torch.Tensor,
    u_data: torch.Tensor,
    y_data: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Data loss Ld (Eq. 7 in RAMP-Net).

    Ld = (1/|D|) Σ ||ϕ_T(x_k, u_k; θ) - y_k||²

    Args:
        model: RampNet PINN model
        x_data: Input states [|D|, 7]
        u_data: Input controls [|D|, 3]
        y_data: Ground truth next states [|D|, 7]
        dt: Time horizon T (= mpc_dt)

    Returns:
        Scalar data loss
    """
    batch_size = x_data.shape[0]
    t = torch.full(
        (batch_size, 1), dt, device=x_data.device, dtype=x_data.dtype,
    )
    x_pred = model(x_data, u_data, t)  # [B, 7]

    return torch.mean((x_pred - y_data) ** 2)


def ramp_loss(
    model: RampNet,
    x_phys: torch.Tensor,
    u_phys: torch.Tensor,
    x_data: torch.Tensor,
    u_data: torch.Tensor,
    y_data: torch.Tensor,
    J_diag: torch.Tensor,
    noise_std: float,
    dt: float,
    rng_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Total RAMP-Net composite loss: L = Lp + Ld.

    Both loss weights are 1.0 following the paper (Section V-A).

    Args:
        rng_seed: Optional seed for deterministic physics loss within L-BFGS closure.

    Returns:
        Dictionary with 'physics', 'data', 'total' loss values.
    """
    Lp = physics_loss(model, x_phys, u_phys, J_diag, noise_std, dt, rng_seed=rng_seed)
    Ld = data_loss(model, x_data, u_data, y_data, dt)

    return {
        "physics": Lp,
        "data": Ld,
        "total": Lp + Ld,
    }
