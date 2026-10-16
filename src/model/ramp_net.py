"""
RAMP-Net: Physics-informed Neural Network for spacecraft attitude dynamics.

Reference:
    Sanyal & Roy, "RAMP-Net: A Robust Adaptive MPC for Quadrotors via
    Physics-informed Neural Network", ICRA 2023.

Architecture: 4-layer MLP with 128 neurons per layer, tanh activation.
Input:  [x(7), u(3), t(1)] = 11D
Output: x(t) = 7D  (predicted state at time t)
"""

import torch
import torch.nn as nn


class RampNet(nn.Module):

    def __init__(
        self,
        state_dim: int = 7,
        control_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.input_dim = state_dim + control_dim + 1  # +1 for time variable t

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        layers = []
        in_dim = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

        # Xavier initialization (common for PINN with tanh)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict state at time t given initial state x0 and control u.

        Args:
            x0: Initial state [B, state_dim]  (q0,q1,q2,q3, w1,w2,w3)
            u:  Control input [B, control_dim] (torque)
            t:  Time variable [B, 1]  (must have requires_grad=True for physics loss)

        Returns:
            x_t: Predicted state at time t [B, state_dim]
        """
        inp = torch.cat([x0, u, t], dim=-1)  # [B, 11]
        return self.net(inp)  # [B, 7]

    def predict_next_state(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Convenience method for inference: predict x[k+1] = phi_T(x[k], u[k]).

        Args:
            x0: Current state [B, state_dim]
            u:  Control input [B, control_dim]
            dt: Time horizon T (= mpc_dt)

        Returns:
            x_next: Predicted next state [B, state_dim]
        """
        batch_size = x0.shape[0]
        t = torch.full((batch_size, 1), dt, device=x0.device, dtype=x0.dtype)
        with torch.no_grad():
            return self.forward(x0, u, t)
