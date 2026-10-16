import logging
import torch
import torch.nn as nn

from src.model.modules import (
    Koopman,
    ContextEncoder,
    VAEEncoder,
    VAEDecoder
)

logger = logging.getLogger(__name__)


class Party(nn.Module):
    def __init__(self, state_dim: int, control_dim: int, hv_dim: int, hk_dim: int, latent_dim: int,
                 vae_depth: int,
                 rho_max_A: float,
                 use_custom_mamba: bool = False):
        super().__init__()
        self.x_dim = state_dim
        self.u_dim = control_dim
        self.hv_dim = hv_dim
        self.hk_dim = hk_dim
        self.l_dim = latent_dim

        self.vae_encoder = VAEEncoder(
            state_dim=self.x_dim, hidden_dim=self.hv_dim, latent_dim=self.l_dim,
            depth=vae_depth,
        )
        self.vae_decoder = VAEDecoder(
            state_dim=self.x_dim, hidden_dim=self.hv_dim, latent_dim=self.l_dim,
            depth=vae_depth,
        )
        self.context_encoder = ContextEncoder(
            state_dim=self.x_dim, control_dim=self.u_dim,
            hidden_dim=self.hv_dim, latent_dim=self.l_dim,
            use_custom_mamba=use_custom_mamba,
        )
        self.koopman = Koopman(
            latent_dim=self.l_dim, control_dim=self.u_dim, hidden_dim=self.hk_dim,
            rho_max_A=rho_max_A
        )


    # ==================== Properties ====================
    @property
    def state_dim(self) -> int:
        return self.x_dim

    @property
    def control_dim(self) -> int:
        return self.u_dim

    @property
    def latent_dim(self) -> int:
        return self.l_dim

    # ==================== Public API ====================
    def get_ctx(self, hist_x: torch.Tensor, hist_u: torch.Tensor) -> torch.Tensor:
        """(b, H+1, x_dim), (b, H, u_dim) -> (b, l_dim)"""
        ctx, _ = self.context_encoder(hist_x, hist_u)
        return ctx

    def get_ctx_with_cache(self, hist_x: torch.Tensor, hist_u: torch.Tensor,
                           inference_params=None) -> tuple[torch.Tensor, object]:
        """Full forward that returns updated inference_params for later incremental use."""
        return self.context_encoder(hist_x, hist_u, inference_params=inference_params)

    def init_ctx_inference_params(self, batch_size: int, max_seqlen: int = 2000):
        return self.context_encoder.init_inference_params(batch_size, max_seqlen)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vae_encoder(x)

    def decode(self, z: torch.Tensor,
               return_raw_qnorm: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.vae_decoder(z, return_raw_qnorm=return_raw_qnorm)

    def get_AB(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.koopman.make_AB(ctx)

    def step(self, z: torch.Tensor, u: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.koopman.step(z, u, A, B)

    # ==================== Forward (Training) ====================
    def forward(self,
                hist_x: torch.Tensor, hist_u: torch.Tensor,
                u_seq: torch.Tensor,
                return_all: bool = False
                ) -> (tuple[torch.Tensor, torch.Tensor] |
                      tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                            tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            hist_x: (b, H+1, x_dim) normalized history states
            hist_u: (b, H, u_dim) normalized history controls
            u_seq:  (b, N, u_dim) normalized future controls
        """
        N = u_seq.shape[1]
        b = hist_x.shape[0]
        device, dtype = hist_x.device, hist_x.dtype

        ctx = self.get_ctx(hist_x, hist_u)
        A, B = self.get_AB(ctx)

        x_init = hist_x[:, -1, :]
        z_mu, z_logvar = self.encode(x_init)
        if self.training:
            z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar.clamp(max=-2.0))
        else:
            z = z_mu

        z_seq = torch.zeros(b, N + 1, self.l_dim, device=device, dtype=dtype)
        x_seq = torch.zeros(b, N + 1, self.x_dim, device=device, dtype=dtype)
        z_seq[:, 0] = z
        x_seq[:, 0] = x_init

        raw_qnorms = []
        for t in range(N):
            z = self.step(z, u_seq[:, t], A, B)
            z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-50.0, 50.0)
            if return_all:
                x, rqn = self.decode(z, return_raw_qnorm=True)
                raw_qnorms.append(rqn)
            else:
                x = self.decode(z)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            z_seq[:, t + 1] = z
            x_seq[:, t + 1] = x

        if return_all:
            raw_qnorms_t = torch.stack(raw_qnorms, dim=1)  # (b, N)
            return x_seq, z_seq, A, B, (z_mu, z_logvar), ctx, raw_qnorms_t
        return x_seq, z_seq
