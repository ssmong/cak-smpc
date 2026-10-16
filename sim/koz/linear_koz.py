import logging
import numpy as np
import yaml
from typing import List
from omegaconf import DictConfig
from scipy.stats import norm

from sim.koz.koz import KeepoutZone

logger = logging.getLogger(__name__)


class LinearKOZ:
    def __init__(
        self,
        koz_list: List[DictConfig],
        cc_cfg: DictConfig,
        horizon: int,
        margin_file: str | None = None,
        margin_key: str = "margin",
        w_beta: float = 1.0,
    ):
        self.cfg = cc_cfg
        self.horizon = horizon
        self.w_beta = float(w_beta)

        self.koz_objects = KeepoutZone.from_cfg(koz_list)
        self.num_koz = len(self.koz_objects)

        self.koz_dirs = np.stack([k.direction for k in self.koz_objects])       # (M, 3)
        self.koz_angles_deg = np.array([k.angle_deg for k in self.koz_objects]) # (M,)

        # -------------------------------------------------------
        # Risk Allocation & Pre-calculation
        # -------------------------------------------------------
        self.k_alpha = self._compute_k_alpha(horizon)
        if self.w_beta != 1.0:
            logger.debug(f"CC tightening calibration: w_beta={self.w_beta:.4f} "
                        f"(effective k_alpha*beta = {self.k_alpha * self.w_beta:.4f})")

        # Buffer & CC Limits (Degree -> Cosine)
        # Note: Cosine is decreasing function in [0, 180]
        self.buf_koz_cos = np.cos(np.deg2rad(self.koz_angles_deg + self.cfg.koz_buffer_deg))
        self.buf_cc_cos = np.cos(np.deg2rad(self.koz_angles_deg + self.cfg.cc_buffer_deg))

        self.c_nominal = np.cos(np.deg2rad(self.koz_angles_deg))
        self.c_cap = np.cos(np.deg2rad(self.koz_angles_deg + self.cfg.max_tighten_deg))

        self.max_delta = self.c_nominal - self.c_cap

        # -------------------------------------------------------
        # Empirical Linearization Margin
        # -------------------------------------------------------
        self.lin_margin = np.zeros(horizon)  # (H,) default: no margin
        if margin_file is not None:
            self._load_margin(margin_file, margin_key)

    def _load_margin(self, margin_file: str, margin_key: str = "margin_max") -> None:
        """Load empirical linearization margin from YAML file.

        Args:
            margin_file: Path to YAML containing margin arrays.
            margin_key: Key to read from the YAML (e.g. ``"margin_max"``,
                ``"margin_p95"``, ``"margin_p99"``).
        """
        with open(margin_file, "r") as f:
            data = yaml.safe_load(f)
        if margin_key not in data:
            available = [k for k in data if k.startswith("margin")]
            raise KeyError(
                f"Margin key '{margin_key}' not found in {margin_file}. "
                f"Available margin keys: {available}"
            )
        margin = np.array(data[margin_key], dtype=np.float64)
        if len(margin) != self.horizon:
            raise ValueError(
                f"Margin length ({len(margin)}) != horizon ({self.horizon}). "
                f"Re-run MC margin computation with matching horizon."
            )
        self.lin_margin = np.maximum(margin, 0.0)  # only tighten, never loosen
        logger.debug(
            "Loaded linearization margin [%s] from %s: range=[%.6f, %.6f]",
            margin_key, margin_file, margin.min(), margin.max(),
        )

    def _compute_k_alpha(self, horizon):
        """Compute quantile for Boole's inequality risk allocation."""
        risk_total = 1.0 - float(self.cfg.alpha)
        risk_step = risk_total / float(horizon)
        return float(norm.ppf(1.0 - risk_step))

    def build_constraints(self, z_nom, bs_N_batch, J_bs_batch, P_seq):
        """Build linearized KOZ constraints with chance-constraint tightening.

        Args:
            z_nom: (N, D) nominal latent trajectory.
            bs_N_batch: (N, L, 3) boresight vectors in inertial frame.
            J_bs_batch: (N, L, 3, D) Jacobians of boresight w.r.t. latent state.
            P_seq: (N, D, D) covariance matrix sequence.

        Returns:
            a_k: (N_act, D) linearized constraint coefficient vectors.
            rhs: (N_act,) right-hand side of linearized constraints.
            active_idx: Tuple of (time, boresight, KOZ) index arrays.
            cos_theta: (N, L, M) nonlinear cosine values at z_nom.
        """
        cos_theta = np.einsum('hli, mi -> hlm', bs_N_batch, self.koz_dirs)

        # Gating
        is_active = cos_theta >= self.buf_koz_cos.reshape(1, 1, -1)
        active_idx = np.where(is_active)
        n_active = len(active_idx[0])

        if n_active == 0:
            return (
                np.empty((0, z_nom.shape[1])),
                np.empty((0,)),
                active_idx,
                cos_theta,
            )

        # Extraction
        J_act = J_bs_batch[active_idx[0], active_idx[1]]
        koz_act = self.koz_dirs[active_idx[2]]
        z_act = z_nom[active_idx[0]]
        cos_theta_act = cos_theta[active_idx]

        # Linearization
        a_k = np.einsum('ni, nid -> nd', koz_act, J_act)

        # Chance Constraint tightening
        buf_cc_act = self.buf_cc_cos[active_idx[2]]
        is_cc = cos_theta_act >= buf_cc_act

        delta_c = np.zeros(n_active)
        if np.any(is_cc):
            a_k_cc = a_k[is_cc]
            time_idx_cc = active_idx[0][is_cc]
            P_cc = P_seq[time_idx_cc]

            sigma2 = np.einsum('ni, nij, nj -> n', a_k_cc, P_cc, a_k_cc)
            sigma_b = np.sqrt(np.maximum(sigma2, 0.0))

            delta_c[is_cc] = self.k_alpha * self.w_beta * sigma_b

        # Empirical linearization margin (per horizon step)
        margin_act = self.lin_margin[active_idx[0]]  # (N_act,)

        # Combined tightening with cap
        total_tighten = delta_c + margin_act
        max_delta_act = self.max_delta[active_idx[2]]
        total_tighten = np.minimum(total_tighten, max_delta_act)

        c_final = self.c_nominal[active_idx[2]] - total_tighten
        rhs = c_final - cos_theta_act + np.sum(a_k * z_act, axis=1)

        return a_k, rhs, active_idx, cos_theta