import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method
from choldate import cholupdate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def tensor_remove_row(A: torch.Tensor, p: int) -> torch.Tensor:
    return torch.cat((A[:p], A[p + 1 :]))


def tensor_remove_column(A: torch.Tensor, p: int) -> torch.Tensor:
    return torch.cat((A[:, :p], A[:, p + 1 :]), 1)


def cholesky_add_row(
    L: torch.Tensor, a: torch.Tensor, aa: torch.Tensor,
) -> torch.Tensor:
    """Extend lower-Cholesky L by one new row/column."""
    c = torch.linalg.solve_triangular(L, a, upper=False)
    d = torch.sqrt(aa - c.t() @ c)
    L_new = torch.cat((L, c.t()))
    L_new = torch.cat((L_new, torch.zeros((L_new.shape[0], 1))), dim=1)
    L_new[-1, -1] = d
    return L_new


def cholesky_remove_row(L: torch.Tensor, pos: int) -> torch.Tensor:
    """Remove row/column ``pos`` from lower-Cholesky L."""
    L_new = torch.zeros_like(L[:-1, :-1])
    L_new[:pos, :pos] = L[:pos, :pos]
    L_new[pos:, :pos] = L[pos + 1 :, :pos]
    L33 = L[pos + 1 :, pos + 1 :]
    l32 = L[pos + 1 :, pos]
    L33_new = L33.detach().numpy().copy().astype(float)
    r = l32.detach().numpy().copy().astype(float)
    cholupdate(L33_new.T, r)
    L_new[pos:, pos:] = torch.from_numpy(L33_new)
    return L_new


# ---------------------------------------------------------------------------
#  Single-output Adaptive Sparse GP  (reference code, unmodified logic)
# ---------------------------------------------------------------------------


class AdaptiveSparseGP(GPModel):
    """Adaptive Sparse Gaussian Process Regression model.

    Implements the algorithm from Gómez-Verdejo & Martínez-Ramón (2023).
    Single-output; use :class:`MultiOutputASGP` for multi-output.

    Args:
        X:  (N, D) training inputs.  Oldest first.
        y:  (N,) training targets.  Oldest first.
        kernel:  A Pyro kernel (e.g. ``gp.kernels.RBF``).
        Xu: (M, D) initial inducing points.
        lamb: Forgetting factor (0 < lamb <= 1).
        model_mode: ``'batch'`` or ``'online'``.
        noise: Observation noise variance (scalar tensor).
        jitter: Diagonal jitter for numerical stability.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor],
        kernel: gp.kernels.Kernel,
        Xu: torch.Tensor,
        lamb: float = 1.0,
        model_mode: str = "batch",
        noise: Optional[torch.Tensor] = None,
        mean_function: Optional[torch.nn.Module] = None,
        jitter: float = 1e-6,
    ) -> None:
        assert isinstance(X, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)
        assert isinstance(Xu, torch.Tensor)
        assert 0 < lamb <= 1
        assert model_mode in ("batch", "online")

        super().__init__(X, y, kernel, mean_function, jitter)

        self.Xu = Parameter(Xu)
        if model_mode == "batch":
            self.Xu = Parameter(Xu, requires_grad=False)

        self.Xu_new = None

        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)

        forget_factor = torch.flip(
            torch.cumprod(
                lamb * torch.ones(X.shape[0], dtype=self.X.dtype, device=self.X.device),
                dim=0,
            ),
            [0],
        )
        self.Lambda = torch.cat((forget_factor[1:], self.X.new_tensor([1.0])))
        self.lamb = lamb
        self.model_mode = model_mode

        self.L_updated = False
        self.N = self.X.size(0)
        self.M = self.Xu.size(0)
        self.update_variables_inference()

    # -- mode switching -----------------------------------------------------

    def set_online_learning(self) -> None:
        self.model_mode = "online"
        for n, k in self.named_parameters():
            if n == "Xu":
                k.requires_grad = False

    def set_batch_learning(
        self, fixed_Xu: bool = True, fixed_noise: bool = False,
    ) -> None:
        self.model_mode = "batch"
        for n, k in self.named_parameters():
            if n == "Xu":
                k.requires_grad = not fixed_Xu
            elif n == "noise_unconstrained" and fixed_noise:
                k.requires_grad = False

    # -- generative model ---------------------------------------------------

    def model(self):
        self.set_mode("model")

        if self.model_mode == "batch":
            self.update_variables_inference()
            W = self.W
            D_lamb = self.D
            trace_term = self.trace_term

        elif self.model_mode == "online":
            if self.Xu_new is None:
                Xu_all = self.Xu
                M = self.Xu.size(0)
            else:
                Xu_all = torch.cat((self.Xu, self.Xu_new))
                M = self.Xu.size(0) + 1

            Kuu = self.kernel(Xu_all).contiguous()
            Kuu = 0.5 * (Kuu + Kuu.T)
            Kuu.view(-1)[:: M + 1] += self.jitter
            Luu = torch.linalg.cholesky(Kuu)

            Kuf = self.kernel(Xu_all, self.X)
            W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).T

            D_lamb = self.noise / self.Lambda

            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=-1)
            trace_term = (self.Lambda @ (Kffdiag - Qffdiag) / self.noise).clamp(min=0)

        noise_term = (torch.sum(self.Lambda) - self.Lambda.shape[0]) * torch.log(
            2 * torch.pi * self.noise
        )

        zero_loc = self.X.new_zeros(self.N)
        f_loc = zero_loc + self.mean_function(self.X)

        if self.y is None:
            f_var = D_lamb + W.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            pyro.factor(
                self._pyro_get_fullname("trace_term"),
                -(trace_term + noise_term) / 2.0,
            )
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.LowRankMultivariateNormal(f_loc, W, D_lamb)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            )

    # -- cached variable updates --------------------------------------------

    def update_variables_inference(self) -> None:
        self.update_Luu()
        self.update_W_D()

    def update_Luu(self) -> None:
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu = 0.5 * (Kuu + Kuu.T)
        Kuu.view(-1)[:: self.M + 1] += self.jitter
        self.Luu = torch.linalg.cholesky(Kuu)

    def update_W_D(self) -> None:
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(self.Luu, Kuf, upper=False).t()

        self.Kuf = Kuf
        self.W = W
        self.D = self.noise / self.Lambda

        Kffdiag = self.kernel(self.X, diag=True)
        self.Kffdiag = Kffdiag
        Qffdiag = self.W.pow(2).sum(dim=-1)
        self.dif_KQ = Kffdiag - Qffdiag
        self.trace_term = (self.Lambda @ self.dif_KQ / self.noise).clamp(min=0)

    def update_L(self) -> None:
        W = self.W.t().contiguous()
        D = self.D
        W_Dinv = W / D
        K = W_Dinv.matmul(W.T).contiguous()
        K = 0.5 * (K + K.T)
        K.view(-1)[:: self.M + 1] += 1
        self.L = torch.linalg.cholesky(K)

    def get_L(self) -> torch.Tensor:
        if not self.L_updated:
            self.update_L()
            self.L_updated = True
        return self.L

    # -- online data management ---------------------------------------------

    def add_new_data(self, X_new: torch.Tensor, y_new: torch.Tensor) -> None:
        self.X = torch.cat((self.X, X_new))
        self.y = torch.cat((self.y, y_new))
        self.N = self.X.size(0)

        self.Lambda = torch.cat((self.Lambda * self.lamb, self.X.new_tensor([1.0])))

        Kuf_new = self.kernel(self.Xu, X_new)
        W_new = torch.linalg.solve_triangular(self.Luu, Kuf_new, upper=False).t()
        self.Kuf = torch.cat((self.Kuf, Kuf_new), 1)
        self.W = torch.cat((self.W, W_new))

        self.D = self.noise / self.Lambda

        Kffdiag_new = self.kernel(X_new, diag=True)
        Qffdiag_new = W_new.pow(2).sum(dim=-1)
        dif_KQ_new = Kffdiag_new - Qffdiag_new
        self.dif_KQ = torch.cat((self.dif_KQ, dif_KQ_new))
        self.trace_term = (self.Lambda @ self.dif_KQ / self.noise).clamp(min=0)

        self.Kffdiag = torch.cat((self.Kffdiag, Kffdiag_new))
        self.L_updated = False

    def remove_data(self, L: Optional[int] = None) -> None:
        if L is None:
            Ninit = 1
        else:
            Ninit = self.N - L

        self.X = self.X[Ninit:]
        self.y = self.y[Ninit:]
        self.N = self.X.size(0)

        self.Lambda = self.Lambda[Ninit:]
        self.D = self.noise / self.Lambda

        self.Kuf = self.Kuf[:, Ninit:]
        self.W = self.W[Ninit:]

        self.dif_KQ = self.dif_KQ[Ninit:]
        self.trace_term = (self.Lambda @ self.dif_KQ / self.noise).clamp(min=0)

        self.Kffdiag = self.Kffdiag[Ninit:]

    # -- online inducing point management -----------------------------------

    def add_new_Xu(self) -> None:
        Xu_new = self.Xu_new.clone()
        Xu_aux = torch.cat((self.Xu.detach(), Xu_new))
        self.Xu = Parameter(Xu_aux, requires_grad=False)
        self.M += 1

        a = self.kernel(self.Xu[:-1], Xu_new)
        aa = self.kernel(Xu_new) + self.jitter
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Kuf_new = self.kernel(Xu_new, self.X)
        self.Kuf = torch.cat((self.Kuf, Kuf_new))

        W_new = (Kuf_new - self.Luu[-1:, :-1] @ self.W.t()) / self.Luu[-1, -1]
        self.W = torch.cat((self.W, W_new.t()), 1)

        self.D = self.noise / self.Lambda

        Qffdiag_new = self.W[:, -1:].pow(2).sum(dim=-1)
        self.dif_KQ -= Qffdiag_new
        self.trace_term = (self.Lambda @ self.dif_KQ / self.noise).clamp(min=0)

        self.L_updated = False
        self.Xu_new = None

    def remove_Xu(self, position: Optional[torch.Tensor] = None) -> None:
        Xu_aux = self.Xu.detach()
        ind_all = np.arange(Xu_aux.shape[0])
        ind_keep = np.setdiff1d(ind_all, position)
        self.Xu = Parameter(Xu_aux[ind_keep], requires_grad=False)
        self.M = len(ind_keep)

        # Recompute Luu from Kuu (choldate incremental update is numerically fragile)
        self.update_Luu()
        self.Kuf = self.kernel(self.Xu, self.X)
        self.W = torch.linalg.solve_triangular(self.Luu, self.Kuf, upper=False).t()

        Qffdiag = self.W.pow(2).sum(dim=-1)
        self.dif_KQ = self.Kffdiag - Qffdiag
        self.trace_term = (self.Lambda @ self.dif_KQ / self.noise).clamp(min=0)

        self.L_updated = False

    def update_noise(self) -> None:
        self.D = self.noise / self.Lambda
        self.L_updated = False

    # -- Pyro guide ---------------------------------------------------------

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    # -- prediction ---------------------------------------------------------

    def forward(
        self,
        Xnew: torch.Tensor,
        full_cov: bool = False,
        noiseless: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Predict mean and (co)variance at ``Xnew``."""
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        W = self.W.t().contiguous()
        D = self.D
        W_Dinv = W / D

        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)

        Kus = self.kernel(self.Xu, Xnew)
        Ws = torch.linalg.solve_triangular(self.Luu, Kus, upper=False)

        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        L = self.get_L()
        Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
        Linv_W_Dinv_y = Linv_pack[:, : W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1] :]

        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[:: C + 1] += self.noise
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
            cov = cov.expand(self.y.shape[:-1] + (C, C))
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
            cov = cov.expand(self.y.shape[:-1] + (C,))

        return loc + self.mean_function(Xnew), cov

    # -- high-level update methods ------------------------------------------

    def batch_update(
        self,
        fixed_Xu: bool = True,
        fixed_noise: bool = False,
        num_steps: int = 100,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Batch ELBO training of kernel hyperparameters."""
        self.set_batch_learning(fixed_Xu=fixed_Xu, fixed_noise=fixed_noise)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        loss = None
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.model, self.guide)
            loss.backward()
            optimizer.step()

        self.update_variables_inference()

        if verbose:
            lengthscale = self.kernel.lengthscale.detach().reshape(-1)
            if lengthscale.numel() == 1:
                lengthscale_msg = f"{lengthscale.item():.4f}"
            else:
                preview = ", ".join(f"{v:.4f}" for v in lengthscale[:5].cpu().tolist())
                if lengthscale.numel() > 5:
                    preview += ", ..."
                lengthscale_msg = f"[{preview}] (n={lengthscale.numel()})"
            log.info(f"noise={self.noise.item():.4f}")
            log.info(f"lengthscale={lengthscale_msg}")
            log.info(f"variance={self.kernel.variance.item():.4f}")

        return loss

    def online_update(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        L: int = 100,
        M: int = 10,
        perc_th: float = 0.01,
        num_steps: int = 20,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Online update with kernel-parameter inference (slow)."""
        assert X_new.shape[0] == 1 and y_new.shape[0] == 1

        if self.model_mode != "online":
            self.set_online_learning()

        if self.y.shape[0] > (L - 1):
            self.remove_data(L - 1)
        self.add_new_data(X_new, y_new)

        Qff_relevance = self.Lambda @ self.W.pow(2)
        th = perc_th * torch.max(Qff_relevance)
        pos_remove, _ = torch.where(Qff_relevance < th)[0].sort(descending=True)
        if pos_remove.shape[0] == 0 and self.M >= M:
            pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

        Xu_aux = self.Xu.detach()
        ind_keep = np.setdiff1d(np.arange(Xu_aux.shape[0]), pos_remove)
        self.Xu = Parameter(Xu_aux[ind_keep], requires_grad=False)
        self.M = len(ind_keep)

        self.Xu_new = Parameter(X_new.clone())

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        loss = None
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.model, self.guide)
            loss.backward()
            optimizer.step()

        Xu_aux = torch.cat((self.Xu.detach(), self.Xu_new.clone()))
        self.Xu = Parameter(Xu_aux, requires_grad=False)
        self.M += 1

        self.update_variables_inference()
        return loss

    def fast_online_update(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        L: int = 100,
        M: int = 10,
        perc_th: float = 0.01,
        trace_th: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        """Fast online update without kernel-parameter inference."""
        assert X_new.shape[0] == 1 and y_new.shape[0] == 1

        if self.model_mode != "online":
            self.set_online_learning()

        if self.y.shape[0] > (L - 1):
            self.remove_data(L - 1)
        self.add_new_data(X_new, y_new)

        # Add inducing point if trace is high
        if trace_th is None:
            trace_th = self.Kffdiag.sum() / L
        if self.trace_term.item() > trace_th:
            self.Xu_new = X_new.clone()
            self.add_new_Xu()

        # Prune low-relevance inducing points
        Qff_relevance = self.Lambda @ self.W.pow(2)
        th = perc_th * torch.max(Qff_relevance)
        pos_remove, _ = torch.where(Qff_relevance < th)[0].sort(descending=True)
        if pos_remove.shape[0] == 0 and self.M >= M:
            pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

        if pos_remove.shape[0] > 0:
            self.remove_Xu(position=pos_remove)


# ---------------------------------------------------------------------------
#  Multi-output wrapper
# ---------------------------------------------------------------------------


class MultiOutputASGP:
    """Wrapper managing D independent AdaptiveSparseGPRegression instances
    with shared inducing-point locations.

    Each output dimension has its own RBF kernel (ARD lengthscales +
    outputscale) and noise variance, but all share the same inducing
    point locations ``Xu``.

    Args:
        X: (N, Din) training inputs.
        Y: (N, Dout) training targets (residuals).
        Xu: (M, Din) shared inducing points.
        lengthscale_ard: (Dout, Din) per-output lengthscales.
        outputscale: (Dout,) per-output outputscales.
        noise_var: (Dout,) per-output noise variances.
        lamb: Forgetting factor.
        jitter: Diagonal jitter.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Xu: torch.Tensor,
        lengthscale_ard: np.ndarray,
        outputscale: np.ndarray,
        noise_var: np.ndarray,
        lamb: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        self.output_dim = Y.shape[1]
        self.input_dim = X.shape[1]
        self.models: list[AdaptiveSparseGP] = []

        for j in range(self.output_dim):
            pyro.clear_param_store()

            kernel = gp.kernels.RBF(
                input_dim=self.input_dim,
                lengthscale=torch.tensor(lengthscale_ard[j], dtype=torch.float64),
                variance=torch.tensor(float(outputscale[j]), dtype=torch.float64),
            )

            model_j = AdaptiveSparseGP(
                X=X.clone(),
                y=Y[:, j].clone(),
                kernel=kernel,
                Xu=Xu.clone(),
                lamb=lamb,
                noise=torch.tensor(float(noise_var[j]), dtype=torch.float64),
                jitter=jitter,
            )
            self.models.append(model_j)

    def batch_update(
        self,
        num_steps: int = 200,
        lr: float = 0.05,
        fixed_Xu: bool = True,
        fixed_noise: bool = False,
        verbose: bool = False,
    ) -> list[float]:
        """Run Pyro ELBO batch training on all output dimensions."""
        losses = []
        for j, m in enumerate(self.models):
            pyro.clear_param_store()
            loss = m.batch_update(
                fixed_Xu=fixed_Xu, fixed_noise=fixed_noise,
                num_steps=num_steps, lr=lr, verbose=verbose,
            )
            losses.append(float(loss) if loss is not None else float("nan"))
            if verbose:
                log.info(f"  output {j}: loss={losses[-1]:.4f}")
        return losses

    @torch.no_grad()
    def predict(self, Xnew: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-output prediction (supports batch).

        Args:
            Xnew: (1, Din) or (C, Din) test inputs.

        Returns:
            mu:  (Dout,) or (C, Dout) mean predictions.
            var: (Dout,) or (C, Dout) marginal variances.
        """
        C = Xnew.size(0)
        mu = torch.zeros(C, self.output_dim, dtype=torch.float64)
        var = torch.zeros(C, self.output_dim, dtype=torch.float64)
        for j, m in enumerate(self.models):
            loc_j, cov_j = m(Xnew, noiseless=False)
            mu[:, j] = loc_j.reshape(C)
            var[:, j] = cov_j.reshape(C)
        if C == 1:
            return mu.squeeze(0), var.squeeze(0)
        return mu, var

    def export_params(self) -> dict[str, np.ndarray]:
        """Export learned parameters for NumPy runtime.

        Returns dict with arrays suitable for ``np.savez``.
        """
        Xu = self.models[0].Xu.detach().numpy().copy()
        Dout = self.output_dim
        Din = self.input_dim

        lengthscale_ard = np.zeros((Dout, Din), dtype=np.float64)
        outputscale = np.zeros(Dout, dtype=np.float64)
        noise_var = np.zeros(Dout, dtype=np.float64)

        for j, m in enumerate(self.models):
            ls = m.kernel.lengthscale.detach().numpy()
            lengthscale_ard[j] = ls.reshape(-1)
            outputscale[j] = float(m.kernel.variance.item())
            noise_var[j] = float(m.noise.item())

        return {
            "U_scaled": Xu,
            "lengthscale_ard": lengthscale_ard,
            "outputscale": outputscale,
            "noise_var": noise_var,
        }

    # -------------------------------------------------------------------
    #  Runtime: load pretrained model for online adaptation + inference
    # -------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_file: str,
        window_size: int,
        num_inducing: int,
        perc_th: float,
        lambda_forget: float,
        warmup_no_prune: int = 0,
    ) -> "MultiOutputASGP":
        """Load a pretrained ``.npz`` for runtime inference + online update.

        Creates sub-models with correct kernel hyperparameters and empty
        sliding-window buffers, ready for ``update`` / ``predict_np`` /
        ``predict_with_jacobian``.
        """
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ASGP model not found: {model_file}")

        data = np.load(model_file)

        obj = cls.__new__(cls)
        obj.output_dim = int(data["output_dim"])
        obj.input_dim = int(data["lengthscale_ard"].shape[1])
        obj._z_mean = np.array(data["z_mean"], dtype=np.float64)
        obj._z_std = np.array(data["z_std"], dtype=np.float64)
        obj._y_mean = np.array(data["y_mean"], dtype=np.float64)
        obj._y_std = np.array(data["y_std"], dtype=np.float64)
        obj._window_size = int(window_size)
        obj._max_inducing = int(num_inducing)
        obj._perc_th = float(perc_th)
        obj._mean_clip = 0.2
        obj._var_clip = 1.0
        obj._min_var = 1e-10
        obj._warmup_no_prune = int(warmup_no_prune)
        obj._update_count = 0
        obj._full_agp = False  # set externally by controller
        obj._refresh_period = 1  # recompute Luu every N updates (full-AGP)

        lengthscale_ard = np.array(data["lengthscale_ard"], dtype=np.float64)
        outputscale_arr = np.array(data["outputscale"], dtype=np.float64)
        noise_var_arr = np.array(data["noise_var"], dtype=np.float64)
        # Prefer the caller-supplied lambda_forget over the saved value
        lambda_val = float(lambda_forget)
        jitter_val = float(data["jitter"])

        # Subsample inducing points if saved model has more than num_inducing
        U_saved = np.array(data["U_scaled"], dtype=np.float64)
        if U_saved.shape[0] > num_inducing:
            rng = np.random.RandomState(0)
            idx = rng.choice(U_saved.shape[0], size=num_inducing, replace=False)
            U_scaled = U_saved[idx]
        else:
            U_scaled = U_saved
        Xu_t = torch.tensor(U_scaled, dtype=torch.float64)

        obj.models = []
        for j in range(obj.output_dim):
            pyro.clear_param_store()
            kernel = gp.kernels.RBF(
                input_dim=obj.input_dim,
                lengthscale=torch.tensor(
                    lengthscale_ard[j], dtype=torch.float64,
                ),
                variance=torch.tensor(
                    float(outputscale_arr[j]), dtype=torch.float64,
                ),
            )
            # Bootstrap with a single dummy data point, then remove it
            # so the sliding window starts empty.
            dummy_x = torch.zeros(1, obj.input_dim, dtype=torch.float64)
            dummy_y = torch.zeros(1, dtype=torch.float64)
            model_j = AdaptiveSparseGP(
                X=dummy_x,
                y=dummy_y,
                kernel=kernel,
                Xu=Xu_t.clone(),
                lamb=lambda_val,
                noise=torch.tensor(
                    float(noise_var_arr[j]), dtype=torch.float64,
                ),
                jitter=jitter_val,
            )
            model_j.set_online_learning()
            model_j.remove_data()  # N: 1 → 0
            obj.models.append(model_j)

        return obj

    # -------------------------------------------------------------------
    #  Runtime: shared-Xu online update (Algorithm 1 / Algorithm 2)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def update(self, z_raw: np.ndarray, resid: np.ndarray) -> None:
        """One-step online update with shared inducing points.

        Dispatches to Algorithm 1 (fast-AGP) or Algorithm 2 (full-AGP)
        based on ``self._full_agp``.  See Gómez-Verdejo et al. 2024.
        """
        self._update_count += 1
        z_scaled = (z_raw - self._z_mean) / self._z_std
        resid_scaled = (resid - self._y_mean) / self._y_std
        X_new = torch.tensor(z_scaled, dtype=torch.float64).unsqueeze(0)

        # 1. Sliding-window management + add new data (per-output y)
        for j, m in enumerate(self.models):
            y_new_j = torch.tensor([float(resid_scaled[j])], dtype=torch.float64)
            if m.y.shape[0] > (self._window_size - 1):
                m.remove_data(self._window_size - 1)
            m.add_new_data(X_new, y_new_j)

        if self._full_agp:
            self._update_full_agp(X_new)
        else:
            self._update_fast_agp(X_new)

    def _update_fast_agp(self, X_new: torch.Tensor) -> None:
        """Algorithm 1 (fast-AGP): conditional add, then conditional prune."""
        # 2. Trace criterion (averaged across outputs) → conditional add
        avg_trace = 0.0
        avg_th = 0.0
        M_curr = self.models[0].M
        for m in self.models:
            trace_val = float(
                (m.Lambda @ m.dif_KQ / m.noise).clamp(min=0),
            )
            avg_trace += trace_val
            avg_th += float(m.Lambda @ m.Kffdiag) / self._window_size
        avg_trace /= self.output_dim
        avg_th /= self.output_dim

        if avg_trace > avg_th and M_curr < self._max_inducing:
            for m in self.models:
                m.Xu_new = X_new.clone()
                m.add_new_Xu()

        # 3. Prune (skip during warmup)
        if self._update_count <= self._warmup_no_prune:
            return
        self._prune_inducing_points()

    def _update_full_agp(self, X_new: torch.Tensor) -> None:
        """Algorithm 2 (full-AGP): prune to M-1 first, then always add.

        Per the paper (Sec. III-B2), the new inducing point u_{k+1} is
        initialized at x_{t+1} and its location is jointly optimized with
        kernel hyperparameters in the ELBO step.  We defer the merge into
        Xu until ``optimize_hyperparameters`` so the location can be tuned.
        """
        M_curr = self.models[0].M

        # 2. Prune FIRST: remove Rm < Rth, then force down to M-1
        if self._update_count > self._warmup_no_prune:
            self._prune_inducing_points(max_size=self._max_inducing - 1)
        elif M_curr >= self._max_inducing:
            self._force_remove_lowest()

        # 3. Stage new inducing point (learnable) — merged after ELBO
        for m in self.models:
            m.Xu_new = Parameter(X_new.clone())

    def _compute_relevance(self) -> torch.Tensor:
        """Compute R_m relevance metric (Eq. 21).

        R_m = Σ_n λ_n × W[n,m]²  where W = (Luu⁻¹ Kuf)ᵀ.

        This satisfies the exact decomposition R_tot = Σ λ k_{tt} - Σ_m R_m
        (Nystrom quadratic form decomposed per inducing point).
        """
        M_curr = self.models[0].M
        Rm_total = torch.zeros(M_curr, dtype=torch.float64)
        for m in self.models:
            Rm_total += m.Lambda @ m.W.pow(2)
        Rm_total /= self.output_dim
        return Rm_total

    def _force_remove_lowest(self) -> None:
        """Remove the single lowest-relevance inducing point (no threshold)."""
        M_curr = self.models[0].M
        N_curr = self.models[0].N
        if N_curr == 0 or M_curr <= 2:
            return
        Rm_total = self._compute_relevance()
        pos_remove = torch.tensor([int(torch.argmin(Rm_total))])
        for m in self.models:
            m.remove_Xu(position=pos_remove)

    def _prune_inducing_points(self, max_size: int | None = None) -> None:
        """Shared pruning logic for both Algorithm 1 and 2.

        Args:
            max_size: If given (Algorithm 2), force M down to this value
                      by removing lowest-relevance inducing points even if
                      they exceed the threshold.  If None (Algorithm 1),
                      only remove those below threshold.
        """
        M_curr = self.models[0].M
        N_curr = self.models[0].N
        if N_curr == 0 or M_curr <= 2:
            return

        Rm_total = self._compute_relevance()

        # Remove those below threshold
        th = self._perc_th * torch.max(Rm_total)
        pos_remove = torch.where(Rm_total < th)[0]

        if max_size is not None:
            # Algorithm 2: remove lowest-relevance to reach exactly max_size.
            # Threshold identifies *candidates*; we remove exactly
            # (M_curr - max_size) points — the lowest-relevance ones,
            # preferring those below threshold.
            n_to_remove = M_curr - max_size
            if n_to_remove > 0:
                # Sort by relevance ascending; take the n_to_remove lowest
                _, sorted_idx = torch.sort(Rm_total)
                pos_remove = sorted_idx[:n_to_remove]
        else:
            # Algorithm 1: if nothing to remove but at max, remove worst
            if pos_remove.shape[0] == 0 and M_curr >= self._max_inducing:
                pos_remove = torch.tensor([int(torch.argmin(Rm_total))])

        if pos_remove.shape[0] > 0:
            pos_remove, _ = pos_remove.sort(descending=True)
            for m in self.models:
                m.remove_Xu(position=pos_remove)

    # -------------------------------------------------------------------
    #  Runtime: Full-AGP kernel hyperparameter optimization
    # -------------------------------------------------------------------

    def optimize_hyperparameters(
        self, num_steps: int = 5, lr: float = 0.01,
    ) -> list[float]:
        """Full-AGP (Algorithm 2): optimize θ, σ², and u_{k+1} via ELBO.

        Per the paper and reference implementation, ``Xu_new`` is kept as
        a separate learnable ``Parameter``.  The ``model()`` method (online
        mode) automatically concatenates ``Xu_new`` with ``Xu`` and
        recomputes all kernel matrices from scratch, so ELBO gradients
        flow correctly through kernel HP, noise, and the new point
        location.

        Existing inducing point locations (``Xu``) stay frozen
        (``requires_grad=False``).

        After ELBO optimization, ``Xu_new`` is merged into ``Xu`` and
        cached matrices are rebuilt.

        Args:
            num_steps: Number of Adam gradient steps.
            lr: Learning rate.

        Returns:
            List of final ELBO losses per output dimension.
        """
        losses: list[float] = []
        for j, m in enumerate(self.models):
            if m.N < 2:
                losses.append(float("nan"))
                if m.Xu_new is not None:
                    m.Xu_new = Parameter(m.Xu_new.detach(), requires_grad=False)
                    m.add_new_Xu()
                continue

            pyro.clear_param_store()

            # Xu_new is already a learnable Parameter (set in _update_full_agp).
            # model() in online mode will concat(Xu, Xu_new) and recompute.
            # self.parameters() yields kernel HP + noise + Xu_new (Xu is frozen).
            optimizer = torch.optim.Adam(m.parameters(), lr=lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            loss = None
            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = loss_fn(m.model, m.guide)
                loss.backward()
                optimizer.step()

            # Freeze Xu_new and merge into Xu
            m.Xu_new = Parameter(m.Xu_new.detach(), requires_grad=False)
            m.add_new_Xu()

            # Recompute cached matrices with updated kernel HP
            m.update_variables_inference()
            losses.append(loss.detach().item() if loss is not None else float("nan"))

        return losses

    def optimize_noise_outputscale(
        self, num_steps: int = 5, lr: float = 0.01,
    ) -> list[float]:
        """Optimize noise + outputscale via ELBO, keeping lengthscale frozen.

        Outputscale controls the kss-Qss gap (dominant variance term).
        Lengthscale stays fixed to preserve mean prediction shape.
        """
        losses: list[float] = []
        for j, m in enumerate(self.models):
            if m.N < 2:
                losses.append(float("nan"))
                continue

            pyro.clear_param_store()

            # Unfreeze noise + outputscale only; keep lengthscale & Xu frozen
            opt_params = []
            for n, p in m.named_parameters():
                if "noise" in n or "variance" in n:
                    p.requires_grad_(True)
                    opt_params.append(p)
                else:
                    p.requires_grad_(False)

            optimizer = torch.optim.Adam(opt_params, lr=lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            loss = None
            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = loss_fn(m.model, m.guide)
                loss.backward()
                optimizer.step()

            # Recompute cached matrices with updated params
            m.update_variables_inference()
            losses.append(loss.detach().item() if loss is not None else float("nan"))

            # Restore requires_grad state (all frozen for online mode)
            for n, p in m.named_parameters():
                p.requires_grad_(False)

        return losses

    def merge_pending_inducing(self) -> None:
        """Merge any staged Xu_new without ELBO optimization.

        Called when optimize_hyperparameters is skipped (e.g. not yet
        at hp_update_period).
        """
        for m in self.models:
            if m.Xu_new is not None:
                m.Xu_new = Parameter(m.Xu_new.detach(), requires_grad=False)
                m.add_new_Xu()
        # Recompute cached matrices
        for m in self.models:
            m.update_variables_inference()

    # -------------------------------------------------------------------
    #  Runtime: prediction (NumPy interface)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict_np(
        self, z_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at a single raw input.

        Handles z-score normalization, denormalization, and variance
        calibration internally.

        Returns:
            mu:  (Dout,) clipped mean (raw scale).
            var: (Dout,) clipped, calibrated variance (raw scale).
        """
        z_scaled = (z_raw - self._z_mean) / self._z_std
        Xnew = torch.tensor(z_scaled, dtype=torch.float64).unsqueeze(0)

        mu_s = np.zeros(self.output_dim, dtype=np.float64)
        var_s = np.zeros(self.output_dim, dtype=np.float64)

        for j, m in enumerate(self.models):
            if m.N == 0:
                mu_s[j] = 0.0
                var_s[j] = float(m.kernel.variance.detach())
                continue
            mu_j, cov_j = self._predict_output_j(m, Xnew)
            mu_s[j] = mu_j.detach().item()
            var_s[j] = max(cov_j.detach().item(), self._min_var)

        # Denormalize: mu_raw = mu_s * y_std + y_mean, var_raw = var_s * y_std^2
        mu = mu_s * self._y_std + self._y_mean
        var = var_s * self._y_std ** 2

        mu = np.clip(mu, -self._mean_clip, self._mean_clip)
        var = np.clip(var, self._min_var, self._var_clip)
        return mu, var

    def predict_with_jacobian(
        self, z_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict mean, variance, and Jacobian of mean w.r.t. raw input.

        Uses ``torch.autograd.grad`` to differentiate through the GP
        predictive mean (no manual kernel-gradient code needed).

        Returns:
            mu:     (Dout,) clipped mean (raw scale).
            var:    (Dout,) clipped, calibrated variance (raw scale).
            dmu_dz: (Dout, Din_raw) Jacobian of GP mean w.r.t. raw input.
        """
        z_scaled = (z_raw - self._z_mean) / self._z_std
        z_t = torch.tensor(z_scaled, dtype=torch.float64, requires_grad=True)
        Xnew = z_t.unsqueeze(0)  # (1, Din)

        Din = z_raw.shape[0]
        mu_s = np.zeros(self.output_dim, dtype=np.float64)
        var_s = np.zeros(self.output_dim, dtype=np.float64)
        dmu_dz_scaled = np.zeros((self.output_dim, Din), dtype=np.float64)

        for j, m in enumerate(self.models):
            if m.N == 0:
                mu_s[j] = 0.0
                var_s[j] = float(m.kernel.variance.detach())
                continue

            mu_j, cov_j = self._predict_output_j(m, Xnew)
            mu_s[j] = mu_j.item()
            var_s[j] = max(cov_j.detach().item(), self._min_var)

            grad_j = torch.autograd.grad(mu_j, z_t, retain_graph=False)[0]
            dmu_dz_scaled[j] = grad_j.detach().numpy()

        # Denormalize: mu_raw = mu_s * y_std + y_mean, var_raw = var_s * y_std^2
        mu = mu_s * self._y_std + self._y_mean
        var = var_s * self._y_std ** 2

        mu = np.clip(mu, -self._mean_clip, self._mean_clip)
        var = np.clip(var, self._min_var, self._var_clip)

        # Chain rule: dmu_raw/dz_raw = (y_std * dmu_s/dz_scaled) / z_std
        dmu_dz = (dmu_dz_scaled * self._y_std.reshape(-1, 1)) / self._z_std.reshape(1, -1)
        return mu, var, dmu_dz

    # -------------------------------------------------------------------
    #  Internal: single-output prediction (supports autograd through Xnew)
    # -------------------------------------------------------------------

    @staticmethod
    def _predict_output_j(
        m: AdaptiveSparseGP,
        Xnew: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict for one output dimension.

        All model-internal tensors (W, Lambda, Luu, …) are ``.detach()``-ed
        so that only ``Xnew`` carries gradient (if present).  This lets
        ``torch.autograd.grad`` compute dmu/dXnew efficiently.
        """
        # Detach model state (constants w.r.t. the test input)
        Wt = m.W.detach().t().contiguous()          # (M, N)
        D = m.noise.detach() / m.Lambda.detach()    # (N,)
        W_Dinv = Wt / D                             # (M, N)
        W_Dinv_y = W_Dinv @ m.y.detach()            # (M,)

        # RBF-ARD kernel k(Xu, z) — only Xnew carries grad
        ls = m.kernel.lengthscale.detach()           # (Din,)
        vs = m.kernel.variance.detach()              # scalar
        Xu = m.Xu.detach()                           # (M, Din)
        Luu = m.Luu.detach()                         # (M, M)

        diff = Xu - Xnew                            # (M, Din)  via broadcast
        dist2 = (diff / ls).pow(2).sum(dim=1)       # (M,)
        k_u = vs * torch.exp(-0.5 * dist2)          # (M,)

        Ws = torch.linalg.solve_triangular(
            Luu, k_u.unsqueeze(1), upper=False,
        ).squeeze(1)                                 # (M,)

        # L = chol(I + W D^{-1} W^T)  — constant w.r.t. Xnew
        K = W_Dinv @ Wt.T
        K = 0.5 * (K + K.T)
        K.view(-1)[:: m.M + 1] += 1.0
        try:
            L = torch.linalg.cholesky(K)
        except torch._C._LinAlgError:
            K.view(-1)[:: m.M + 1] += 1e-3
            L = torch.linalg.cholesky(K)

        pack = torch.stack([W_Dinv_y, Ws], dim=1)   # (M, 2)
        Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
        Linv_W_Dinv_y = Linv_pack[:, 0]             # constant
        Linv_Ws = Linv_pack[:, 1]                    # has grad

        mu_j = Linv_W_Dinv_y @ Linv_Ws

        kss = vs
        Qss = Ws @ Ws
        cov_j = kss - Qss + Linv_Ws @ Linv_Ws + m.noise.detach()

        return mu_j, cov_j
