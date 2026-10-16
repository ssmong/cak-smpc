import torch
import torch.nn as nn
import torch.nn.functional as F


class _PreNormResBlock(nn.Module):
    """Pre-norm 2-layer residual block: h -> h + Linear(GELU(Linear(LayerNorm(h))))"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_ff = self.norm(h)
        h_ff = F.gelu(self.linear1(h_ff))
        h_ff = self.linear2(h_ff)
        return h + h_ff


class VAEEncoder(nn.Module):
    """VAE Encoder: state [q, omega] -> (mean, log_var).
    Pre-norm residual with depth 2-layer blocks.
    """
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int,
                 depth: int):
        super().__init__()
        self.x_dim = state_dim
        self.h_dim = hidden_dim
        self.l_dim = latent_dim

        self.proj = nn.Linear(state_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([_PreNormResBlock(hidden_dim) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.proj(x))
        for block in self.res_blocks:
            h = block(h)
        h = self.output_norm(h)
        return self.mean_head(h), self.logvar_head(h)


class VAEDecoder(nn.Module):
    """VAE Decoder: latent state z -> physical state [q, omega].
    Single trunk (proj -> res_blocks -> output). Quat (first 4 dims) is L2-normalized.
    """
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int,
                 depth: int):
        super().__init__()
        self.x_dim = state_dim
        self.h_dim = hidden_dim
        self.l_dim = latent_dim

        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([_PreNormResBlock(hidden_dim) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, state_dim)

    def forward(self, z: torch.Tensor,
                return_raw_qnorm: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.proj(z))
        for block in self.res_blocks:
            h = block(h)
        h = self.output_norm(h)
        out = self.output(h)

        q_raw = out[..., :4]
        q = F.normalize(q_raw, p=2, dim=-1, eps=1e-8)
        x_out = torch.cat([q, out[..., 4:]], dim=-1)

        if return_raw_qnorm:
            return x_out, q_raw.norm(p=2, dim=-1)
        return x_out
