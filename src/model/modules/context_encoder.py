import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """Encodes past window of states and controls using Mamba"""
    def __init__(self, state_dim: int = 7, control_dim: int = 3, hidden_dim: int = 64, latent_dim: int = 32,
                 use_custom_mamba: bool = False):
        super().__init__()
        self.x_dim = state_dim
        self.u_dim = control_dim
        self.h_dim = hidden_dim
        self.l_dim = latent_dim

        if use_custom_mamba:
            from src.model.modules.mamba_pure import MambaPure as Mamba, InferenceParams
        else:
            from mamba_ssm import Mamba
            from mamba_ssm.utils.generation import InferenceParams

        self._InferenceParams = InferenceParams

        self.token_proj = nn.Linear(self.x_dim + self.u_dim, self.h_dim)
        # layer_idx is crucial for cache management in Mamba
        self.mamba1 = Mamba(d_model=self.h_dim, d_state=16, d_conv=4, expand=2, layer_idx=0)
        self.mamba2 = Mamba(d_model=self.h_dim, d_state=16, d_conv=4, expand=2, layer_idx=1)
        self.ln1 = nn.LayerNorm(self.h_dim)
        self.ln2 = nn.LayerNorm(self.h_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU(),
            nn.Linear(self.h_dim, self.l_dim)
        )

    def forward(self, hist_x: torch.Tensor, hist_u: torch.Tensor,
                inference_params=None) -> tuple[torch.Tensor, object | None]:
        """
        Args:
            hist_x: (B, H+1, x_dim) - history states [t-H, ..., t]
            hist_u: (B, H, u_dim) - history controls [t-H, ..., t-1]
        """
        # Align: hist_x[:, 1:] pairs with hist_u (both length H)
        token_input = torch.cat([hist_x[:, 1:], hist_u], dim=-1)
        token_embeds = self.token_proj(token_input)

        is_incremental = False
        if inference_params is not None:
            is_incremental = inference_params.seqlen_offset > 0

        if is_incremental:
            h2 = None
            for i in range(token_embeds.shape[1]):
                token = token_embeds[:, i:i + 1]
                h1 = self.mamba1(token, inference_params=inference_params)
                h1 = self.ln1(h1 + token)
                h2 = self.mamba2(h1, inference_params=inference_params)
                h2 = self.ln2(h2 + h1)
                inference_params.seqlen_offset += 1

            e = self.output_proj(h2.squeeze(1))
            return e, inference_params

        else:
            h1 = self.mamba1(token_embeds, inference_params=inference_params)
            h1 = self.ln1(h1 + token_embeds)
            h2 = self.mamba2(h1, inference_params=inference_params)
            h2 = self.ln2(h2 + h1)

            e = self.output_proj(h2[:, -1])

            if inference_params is not None:
                inference_params.seqlen_offset += token_embeds.shape[1]
            return e, inference_params

    def init_inference_params(self, batch_size: int, max_seqlen: int = 2000):
        return self._InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
