import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Union
import numpy as np
import time

from src.model.party import Party
from src.dataset import DataProcessor

log = logging.getLogger(__name__)

class PartyInterface(nn.Module):
    """
    Wrapper for Party model that handles input/output normalization.
    Context embedding (from ContextEncoder) is used only by Koopman (A, B generation).
    """
    def __init__(self, model: Party, processor: DataProcessor | None = None):
        super().__init__()
        self.model = model

        if processor is not None:
            self.processor = processor
        else:
            log.info("Creating new DataProcessor... load_stats should be called.")
            self.processor = DataProcessor(model.state_dim, model.control_dim)

    def set_eval(self):
        self.model.eval()
        log.info("PartyInterface set to eval mode")

    def load_stats(self, stats_path: Union[str, Path]):
        self.processor.load_stats(stats_path)

    def save_stats(self, stats_path: Union[str, Path]):
        self.processor.save_stats(stats_path)

    @property
    def state_dim(self) -> int:
        return self.model.state_dim

    @property
    def latent_dim(self) -> int:
        return self.model.latent_dim

    @property
    def control_dim(self) -> int:
        return self.model.control_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ==================== Public API ====================
    def get_ctx(self, hist_x_raw: torch.Tensor, hist_u_raw: torch.Tensor) -> torch.Tensor:
        """Raw history -> context embedding. Auto-unsqueezes unbatched inputs."""
        if hist_x_raw.dim() == 2:
            hist_x_raw = hist_x_raw.unsqueeze(0)
        if hist_u_raw.dim() == 2:
            hist_u_raw = hist_u_raw.unsqueeze(0)
        hist_x_n = self.processor.normalize_state(hist_x_raw)
        hist_u_n = self.processor.normalize_control(hist_u_raw)
        return self.model.get_ctx(hist_x_n, hist_u_n)

    def encode(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw state -> latent (normalizes, then encodes)."""
        x_n = self.processor.normalize_state(x_raw)
        return self.model.encode(x_n)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> raw state (denormalizes output)."""
        x_n = self.model.decode(z)
        return self.processor.denormalize_state(x_n)

    def get_AB(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Context embedding -> (A, B)."""
        return self.model.get_AB(ctx)

    def step(self, z: torch.Tensor, u_raw: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Single Koopman step (normalizes control internally)."""
        u_n = self.processor.normalize_control(u_raw)
        return self.model.step(z, u_n, A, B)

    def init_ctx_inference_params(self, batch_size: int = 1, max_seqlen: int = 2000):
        """Create fresh InferenceParams for incremental context encoding."""
        return self.model.init_ctx_inference_params(batch_size, max_seqlen)

    def get_mpc_initials_with_cache(
        self, hist_x_raw: torch.Tensor, hist_u_raw: torch.Tensor,
        inference_params=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, object, dict]:
        """Like get_mpc_initials but returns updated inference_params for incremental use."""
        hist_x_n = self.processor.normalize_state(hist_x_raw.unsqueeze(0))
        hist_u_n = self.processor.normalize_control(hist_u_raw.unsqueeze(0))
        x_curr_n = self.processor.normalize_state(hist_x_raw[-1:].unsqueeze(0))

        with torch.no_grad():
            _t0 = time.perf_counter()
            ctx, inference_params = self.model.get_ctx_with_cache(
                hist_x_n, hist_u_n, inference_params=inference_params)
            _t1 = time.perf_counter()
            A_t, B_t = self.model.get_AB(ctx)
            _t2 = time.perf_counter()
            z_t, z_lv_t = self.model.encode(x_curr_n)
            _t3 = time.perf_counter()

        z = z_t.reshape(-1).cpu().numpy().astype(np.float64)
        z_lv = z_lv_t.reshape(-1).cpu().numpy().astype(np.float64)
        A = A_t.squeeze(0).cpu().numpy().astype(np.float64)
        B = B_t.squeeze(0).cpu().numpy().astype(np.float64)
        _t4 = time.perf_counter()
        profile = {'get_ctx': _t1 - _t0, 'get_AB': _t2 - _t1,
                   'encode': _t3 - _t2, 'to_numpy': _t4 - _t3}
        return z, z_lv, A, B, ctx, inference_params, profile

    def get_mpc_initials_incremental(
        self, hist_x_new_raw: torch.Tensor, hist_u_new_raw: torch.Tensor,
        inference_params,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, object, dict]:
        """Incremental context update with K new tokens.
        hist_x_new_raw: (K+1, x_dim) — [x_{t-K}, ..., x_t]  (K+1 states to form K tokens)
        hist_u_new_raw: (K, u_dim)   — [u_{t-K}, ..., u_{t-1}]
        """
        hist_x_n = self.processor.normalize_state(hist_x_new_raw.unsqueeze(0))
        hist_u_n = self.processor.normalize_control(hist_u_new_raw.unsqueeze(0))
        x_curr_n = self.processor.normalize_state(hist_x_new_raw[-1:].unsqueeze(0))

        with torch.no_grad():
            _t0 = time.perf_counter()
            ctx, inference_params = self.model.get_ctx_with_cache(
                hist_x_n, hist_u_n, inference_params=inference_params)
            _t1 = time.perf_counter()
            A_t, B_t = self.model.get_AB(ctx)
            _t2 = time.perf_counter()
            z_t, z_lv_t = self.model.encode(x_curr_n)
            _t3 = time.perf_counter()

        z = z_t.reshape(-1).cpu().numpy().astype(np.float64)
        z_lv = z_lv_t.reshape(-1).cpu().numpy().astype(np.float64)
        A = A_t.squeeze(0).cpu().numpy().astype(np.float64)
        B = B_t.squeeze(0).cpu().numpy().astype(np.float64)
        _t4 = time.perf_counter()
        profile = {'get_ctx': _t1 - _t0, 'get_AB': _t2 - _t1,
                   'encode': _t3 - _t2, 'to_numpy': _t4 - _t3}
        return z, z_lv, A, B, ctx, inference_params, profile

    def get_mpc_initials(
        self, hist_x_raw: torch.Tensor, hist_u_raw: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, dict]:
        """
        Args:
            hist_x_raw: (H+1, x_dim) raw history states
            hist_u_raw: (H, u_dim) raw history controls
        Returns:
            z_curr, z_logvar, A, B (numpy float64), ctx (torch tensor for reuse),
            profile (dict with keys 'get_ctx', 'get_AB', 'encode', 'to_numpy', seconds)
        """
        hist_x_n = self.processor.normalize_state(hist_x_raw.unsqueeze(0))
        hist_u_n = self.processor.normalize_control(hist_u_raw.unsqueeze(0))
        # Decoder normalizes quaternion to unit norm (no sign canonicalization)
        x_curr_n = self.processor.normalize_state(hist_x_raw[-1:].unsqueeze(0))

        with torch.no_grad():
            _t0 = time.perf_counter()
            ctx = self.model.get_ctx(hist_x_n, hist_u_n)
            _t1 = time.perf_counter()
            A_t, B_t = self.model.get_AB(ctx)
            _t2 = time.perf_counter()
            z_t, z_lv_t = self.model.encode(x_curr_n)
            _t3 = time.perf_counter()

        z = z_t.reshape(-1).cpu().numpy().astype(np.float64)
        z_lv = z_lv_t.reshape(-1).cpu().numpy().astype(np.float64)
        A = A_t.squeeze(0).cpu().numpy().astype(np.float64)
        B = B_t.squeeze(0).cpu().numpy().astype(np.float64)
        _t4 = time.perf_counter()
        profile = {'get_ctx': _t1 - _t0, 'get_AB': _t2 - _t1,
                   'encode': _t3 - _t2, 'to_numpy': _t4 - _t3}
        return z, z_lv, A, B, ctx, profile
