"""Pure PyTorch Mamba block — drop-in replacement for mamba_ssm.Mamba.

Weight-compatible with trained mamba_ssm checkpoints (identical state_dict keys
and shapes). No custom CUDA kernels required, works on any PyTorch-supported GPU
including Jetson Orin Nano (aarch64).

The selective scan uses a sequential loop over L (vectorized over B, D, N dims),
matching mamba_ssm's own `selective_scan_ref`. For our typical L=50, the loop
overhead is negligible compared to per-step GPU tensor ops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InferenceParams:
    """Minimal replacement for mamba_ssm.utils.generation.InferenceParams."""
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None


def selective_scan_ref(
    u: torch.Tensor,       # (B, D, L)
    delta: torch.Tensor,   # (B, D, L)
    A: torch.Tensor,       # (D, N)
    B: torch.Tensor,       # (B, N, L)
    C: torch.Tensor,       # (B, N, L)
    D: torch.Tensor | None = None,  # (D,)
    z: torch.Tensor | None = None,  # (B, D, L)
    delta_bias: torch.Tensor | None = None,  # (D,)
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Reference selective scan — sequential over L, vectorized over B/D/N."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()

    # Discretize: deltaA = exp(delta * A), deltaB_u = delta * B * u
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    # Sequential scan
    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, D, L)

    out = y if D is None else y + u * D.unsqueeze(0).unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype_in)


class MambaPure(nn.Module):
    """Pure PyTorch Mamba block with identical state_dict to mamba_ssm.Mamba."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,  # ignored, kept for API compat
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Projections (exact names match mamba_ssm)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias, **factory,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory)

        # SSM parameters (nn.Parameter, not modules)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

        # dt_proj special initialization (match mamba_ssm)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, hidden_states: torch.Tensor,
                inference_params: InferenceParams | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, d_model)
            inference_params: optional, for incremental decoding
        Returns:
            (B, L, d_model)
        """
        is_step = (inference_params is not None
                   and inference_params.seqlen_offset > 0
                   and hidden_states.shape[1] == 1)

        if is_step:
            return self._step(hidden_states, inference_params)

        batch, seqlen, _ = hidden_states.shape

        # 1. Input projection
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)       # each (B, L, d_inner)

        # 2. Causal convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :seqlen]  # causal: trim padding
        x = F.silu(x)

        # 3. SSM projections
        x_t = x.transpose(1, 2)  # (B, L, d_inner)
        x_dbl = self.x_proj(x_t)  # (B, L, dt_rank + 2*d_state)
        dt, B_proj, C_proj = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 4. dt projection
        dt = self.dt_proj.weight @ dt.transpose(1, 2)  # (d_inner, dt_rank) @ (B, dt_rank, L) -> (B, d_inner, L)

        # 5. Reshape B, C for scan
        B_proj = B_proj.transpose(1, 2)  # (B, d_state, L)
        C_proj = C_proj.transpose(1, 2)  # (B, d_state, L)

        # 6. Selective scan
        A = -torch.exp(self.A_log.float())
        z_t = z.transpose(1, 2)  # (B, d_inner, L)
        y = selective_scan_ref(
            x, dt, A, B_proj, C_proj,
            D=self.D.float(),
            z=z_t,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # 7. Output projection
        y = y.transpose(1, 2)  # (B, L, d_inner)
        out = self.out_proj(y)

        # Cache conv state for incremental inference
        if inference_params is not None:
            xz_for_cache = self.in_proj(hidden_states)
            x_cache = xz_for_cache[..., :self.d_inner].transpose(1, 2)  # (B, d_inner, L)
            conv_state = F.pad(x_cache, (self.d_conv - x_cache.shape[-1], 0))
            ssm_state = torch.zeros(
                batch, self.d_inner, self.d_state,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)

        return out

    def _step(self, hidden_states: torch.Tensor,
              inference_params: InferenceParams) -> torch.Tensor:
        """Single-token incremental step."""
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]

        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Conv step: shift state
        conv_state.copy_(torch.roll(conv_state, -1, dims=-1))
        conv_state[:, :, -1] = x
        x = (conv_state * self.conv1d.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = F.silu(x)

        # SSM projections
        x_db = self.x_proj(x)
        dt, B_val, C_val = x_db.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.linear(dt, self.dt_proj.weight)
        dt = F.softplus(dt + self.dt_proj.bias)

        A = -torch.exp(self.A_log.float())
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B_val.float())
        ssm_state.copy_(ssm_state * dA + dB * x.unsqueeze(-1).float())
        y = torch.einsum("bdn,bn->bd", ssm_state.float(), C_val.float())
        y = y + self.D.float() * x.float()
        y = (y * F.silu(z.float())).to(hidden_states.dtype)

        out = self.out_proj(y)
        return out.unsqueeze(1)
