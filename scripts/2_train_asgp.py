import os
import logging
import hydra
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.dataset import PartyDataset
from src.utils import set_seed
from src.train.train_asgp import (
    collect_mismatch_data_omega,
    fit_hyperparams_per_sequence,
    prepare_gp_inputs,
    kmeans_inducing_points,
    save_asgp_model_direct,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="learning_asgp", version_base=None)
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       0. Seed & WandB & Output Directory
    # ------------------------------------------------------------------
    set_seed(cfg.data.seed)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    run_name = f"asgp_{cfg.exp_name}"

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=wandb_cfg,
        name=run_name,
    )

    # ------------------------------------------------------------------
    #       1. Dataset
    # ------------------------------------------------------------------
    log.info("Loading dataset...")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_cfg, dict):
        raise TypeError("cfg.data must resolve to a dict-like config")
    data_cfg.pop("_target_", None)
    dataset = PartyDataset(**data_cfg, split="train", use_norm=False)

    pred_stride = int(round(float(cfg.data.pred_dt) / float(cfg.data.log_dt)))
    if pred_stride < 1:
        raise ValueError(f"Invalid pred_stride: {pred_stride}")

    inertia = np.array(cfg.asgp.inertia, dtype=np.float64).reshape(3, 3)
    inertia_inv = np.linalg.inv(inertia)
    rk4_steps = int(cfg.asgp.rk4_steps)
    jitter = float(cfg.asgp.jitter)

    log.info(f"RK4 sub-steps: {rk4_steps} (sub_dt={float(cfg.data.pred_dt)/rk4_steps:.4f}s)")

    # ------------------------------------------------------------------
    #       2. Residual Collection & GP Preparation
    # ------------------------------------------------------------------
    log.info("Collecting per-sequence dω residuals (J_nom) ...")
    z_per_seq, y_per_seq = collect_mismatch_data_omega(
        dataset, pred_stride, float(cfg.data.pred_dt), inertia, inertia_inv,
        rk4_steps=rk4_steps,
    )

    # Pool → global normalization
    z_pool = np.concatenate(z_per_seq, axis=0)
    y_pool = np.concatenate(y_per_seq, axis=0)
    log.info(f"Pooled samples: {len(z_pool)}, input_dim={z_pool.shape[1]}, output_dim={y_pool.shape[1]}")

    z_s, z_mean, z_std, y_s, y_mean, y_std = prepare_gp_inputs(
        z_pool, y_pool, int(cfg.asgp.max_samples), int(cfg.data.seed))
    log.info(f"Prepared {len(z_s)} samples after subsampling")
    log.info(f"y_mean={y_mean}, y_std={y_std}")

    # Per-sequence exact GP fitting → median HP
    log.info("Per-sequence exact GP fitting ...")
    median_ls, median_os, median_nv = fit_hyperparams_per_sequence(
        z_per_seq, y_per_seq,
        z_mean, z_std, y_mean, y_std,
        num_steps=int(cfg.asgp.perseq_num_steps),
        lr=float(cfg.asgp.perseq_lr),
    )
    lengthscale_ard = median_ls
    outputscale = median_os
    if bool(cfg.asgp.fixed_noise):
        # Use config noise_var instead of per-sequence median
        # (sparse GP ELBO tends to absorb unexplained variance into noise)
        noise_var = np.full(outputscale.shape[0], float(cfg.asgp.noise_var), dtype=np.float64)
        log.info(f"fixed_noise=True → overriding noise_var to {cfg.asgp.noise_var}")
    else:
        noise_var = median_nv

    log.info(f"Final HP — lengthscale: {lengthscale_ard}")
    log.info(f"Final HP — outputscale: {outputscale}")
    log.info(f"Final HP — noise_var:   {noise_var}")

    # ------------------------------------------------------------------
    #       3. Inducing Points (k-means on normalized inputs)
    # ------------------------------------------------------------------
    m = min(int(cfg.asgp.num_inducing), len(z_s))
    u_s = kmeans_inducing_points(z_s, m, seed=int(cfg.data.seed) + 17)
    log.info(f"Inducing points: {m}")

    # ------------------------------------------------------------------
    #       4. Save & Log
    # ------------------------------------------------------------------
    sparse_dir = os.path.join(output_dir, "sparse_gp")
    save_asgp_model_direct(
        sparse_dir=sparse_dir,
        exp_name=cfg.exp_name,
        u_s=u_s,
        lengthscale_ard=lengthscale_ard,
        outputscale=outputscale,
        noise_var=noise_var,
        z_mean=z_mean, z_std=z_std,
        y_mean=y_mean, y_std=y_std,
        lambda_forget=float(cfg.asgp.lambda_forget),
        jitter=jitter,
        state_dim=int(cfg.data.state_dim),
        control_dim=int(cfg.data.control_dim),
        pred_dt=float(cfg.data.pred_dt),
        num_samples=len(z_s),
        num_inducing=m,
    )

    wandb.log({
        "num_samples": len(z_s),
        "num_inducing": m,
        "noise_var_mean": float(np.mean(noise_var)),
        "outputscale_mean": float(np.mean(outputscale)),
    })
    wandb.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
