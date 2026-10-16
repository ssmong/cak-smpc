import torch
import torch.nn as nn
import torch.nn.functional as F

class Koopman(nn.Module):
    # NOTE: Here, to avoid confusion, batch size is denoted as b_sz instead of Bsz
    def __init__(self, latent_dim: int = 32, control_dim: int = 3, hidden_dim: int = 128,
                 rho_max_A: float = 1.03):
        super().__init__()
        self.l_dim = latent_dim
        self.u_dim = control_dim
        self.h_dim = hidden_dim
        self.rho_max_A = rho_max_A
        self._to_A = nn.Sequential(
            nn.Linear(self.l_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU(),
            nn.Linear(self.h_dim, self.l_dim * self.l_dim)
        )
        self._to_B = nn.Sequential(
            nn.Linear(self.l_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU(),
            nn.Linear(self.h_dim, self.l_dim * self.u_dim)
        )
        # Identity matrix를 고정값으로 등록 (학습 파라미터 아님)
        self.register_buffer('I', torch.eye(latent_dim))
        with torch.no_grad():
            self._to_A[-1].weight.data *= 0.01
            self._to_A[-1].bias.data.zero_()  # A_delta = 0 at init -> A = I
            self._to_B[-1].weight.data *= 0.1
            self._to_B[-1].bias.data.fill_(0.0)

    def make_AB(self, ctx: torch.Tensor):
        b_sz = ctx.shape[0]
        A_flat = self._to_A(ctx)
        A = A_flat.view(b_sz, self.l_dim, self.l_dim)
        B_flat = self._to_B(ctx)
        B = B_flat.view(b_sz, self.l_dim, self.u_dim)
        A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
        B = torch.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
        I = self.I.unsqueeze(0).expand(b_sz, -1, -1)
        A = I + A
        return A, B

    def step(self, z: torch.Tensor, u: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        z_next = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        return z_next
