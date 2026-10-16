"""
RAMP-Net training entry point.

Usage:
    python -m scripts.2_train_ramp
    python -m scripts.2_train_ramp data.num_data_points=5000 data.num_physics_points=5000
"""

import os
import sys
import subprocess
import logging
import time

import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.dataset.ramp_dataset import RampDataset
from src.train.train_ramp import train_ramp
from src.utils import set_seed, print_model_summary

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="learning_ramp", version_base=None)
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       0. Seed & WandB & Output Directory
    # ------------------------------------------------------------------
    seed = cfg.data.seed
    set_seed(seed)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    orig_cwd = hydra.utils.get_original_cwd()
    rel_path = os.path.relpath(output_dir, orig_cwd)
    run_name = "_".join(rel_path.split(os.path.sep)[2:])

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
    #       1. Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")

    # ------------------------------------------------------------------
    #       2. Dataset
    # ------------------------------------------------------------------
    log.info("Initializing RAMP-Net dataset...")

    dataset = RampDataset(
        name=cfg.data.name,
        data_root=hydra.utils.to_absolute_path(cfg.data.data_root),
        num_seqs=cfg.data.num_seqs,
        sim_time=cfg.data.sim_time,
        log_dt=cfg.data.log_dt,
        mpc_dt=cfg.data.mpc_dt,
        num_data_points=cfg.data.num_data_points,
        num_physics_points=cfg.data.num_physics_points,
        omega_max=cfg.data.omega_max,
        u_min=cfg.data.u_min,
        u_max=cfg.data.u_max,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=seed,
        state_key=cfg.data.state_key,
        control_key=cfg.data.control_key,
        device=cfg.device,
    )

    train_batch = dataset.get_train_batch()
    val_batch = dataset.get_val_batch()
    test_batch = dataset.get_test_batch()

    # ------------------------------------------------------------------
    #       3. Model
    # ------------------------------------------------------------------
    log.info("Initializing RAMP-Net model...")

    model = hydra.utils.instantiate(cfg.model).to(device)
    print_model_summary(model)

    # ------------------------------------------------------------------
    #       4. Training
    # ------------------------------------------------------------------
    log.info("Starting RAMP-Net training (L-BFGS, full-batch)...")

    J_diag = torch.tensor(cfg.physics.nominal_inertia, dtype=torch.float32)

    # Measure total training time on GPU (with synchronization for accuracy)
    torch.cuda.synchronize()
    train_start = time.perf_counter()

    best_epoch, best_val_loss = train_ramp(
        model=model,
        train_batch=train_batch,
        val_batch=val_batch,
        J_diag=J_diag,
        noise_std=float(cfg.physics.parametric_noise_std),
        dt=float(cfg.data.mpc_dt),
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        max_lbfgs_iter=cfg.train.max_lbfgs_iter,
        history_size=cfg.train.history_size,
        early_stopping_patience=cfg.train.early_stopping_patience,
        output_dir=output_dir,
        device=device,
    )

    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start

    log.info(f"Best epoch: {best_epoch+1}, best val loss: {best_val_loss:.6f}")
    log.info(f"Total training time: {train_elapsed:.2f}s ({train_elapsed/60:.1f}min)")
    wandb.log({"train/total_time_s": train_elapsed})

    # ------------------------------------------------------------------
    #       5. Test Evaluation (using best model)
    # ------------------------------------------------------------------
    log.info("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        batch_size = test_batch["x_data"].shape[0]
        dt = float(cfg.data.mpc_dt)
        t_test = torch.full(
            (batch_size, 1), dt, device=device, dtype=test_batch["x_data"].dtype,
        )
        x_pred_test = model(test_batch["x_data"], test_batch["u_data"], t_test)
        test_data_loss = torch.mean((x_pred_test - test_batch["y_data"]) ** 2).item()

        # Per-component test loss for diagnostics
        q_pred = x_pred_test[:, :4]
        q_gt = test_batch["y_data"][:, :4]
        w_pred = x_pred_test[:, 4:]
        w_gt = test_batch["y_data"][:, 4:]
        test_q_loss = torch.mean((q_pred - q_gt) ** 2).item()
        test_w_loss = torch.mean((w_pred - w_gt) ** 2).item()

    log.info(
        f"Test data loss: {test_data_loss:.6f} "
        f"(quat: {test_q_loss:.6f}, omega: {test_w_loss:.6f})"
    )
    wandb.log({
        "test/data": test_data_loss,
        "test/quat": test_q_loss,
        "test/omega": test_w_loss,
    })

    # ------------------------------------------------------------------
    #       5.1 Inference Time Measurement (CPU only, no GPU parallelism)
    # ------------------------------------------------------------------
    log.info("Measuring single-step inference time on CPU...")
    model_cpu = model.cpu()
    model_cpu.eval()

    # Warm-up
    x_dummy = torch.randn(1, model.state_dim)
    u_dummy = torch.randn(1, model.control_dim)
    for _ in range(10):
        model_cpu.predict_next_state(x_dummy, u_dummy, dt)

    # Measure
    n_trials = 100
    cpu_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        model_cpu.predict_next_state(x_dummy, u_dummy, dt)
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)

    import numpy as np
    cpu_times = np.array(cpu_times)
    log.info(
        f"CPU inference time ({n_trials} trials): "
        f"mean={cpu_times.mean()*1e3:.3f}ms, "
        f"std={cpu_times.std()*1e3:.3f}ms, "
        f"max={cpu_times.max()*1e3:.3f}ms"
    )
    wandb.log({
        "inference/cpu_mean_ms": cpu_times.mean() * 1e3,
        "inference/cpu_std_ms": cpu_times.std() * 1e3,
    })

    # Move model back to GPU for any remaining operations
    model.to(device)

    # ------------------------------------------------------------------
    #       6. Save config alongside model
    # ------------------------------------------------------------------
    config_save_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_save_path)
    log.info(f"Config saved to {config_save_path}")

    wandb.finish()
    log.info("RAMP-Net training complete.")

    # ------------------------------------------------------------------
    #       7. Run Simulation Experiments (using trained model)
    # ------------------------------------------------------------------
    # Compute model_dir relative to "outputs/learning_ramp/" for ramp_net
    learning_base = os.path.join(orig_cwd, "outputs", "learning_ramp")
    model_dir = os.path.relpath(output_dir, learning_base)

    experiments = [
        ("exp1a", "exp1a", "Constant MOI (In-Distribution)"),
        ("exp1b", "exp1b", "Constant MOI (Non-Diagonal)"),
        ("exp3",  "exp3",  "Constant MOI (OOD)"),
        ("exp2a", "exp2a", "Continuous MOI Drift"),
        ("exp2b", "exp2b", "Abrupt MOI Change"),
    ]

    # Strip WANDB_* env vars so child processes don't create spurious sweep runs
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}

    for sc_name, constraint_name, desc in experiments:
        log.info(f"Running experiment: {desc} (sc={sc_name}, constraint={constraint_name})")
        cmd = [
            sys.executable, "-m", "scripts.3_run_sim",
            "--config-name=sim_ramp",
            f"sc={sc_name}",
            f"constraint={constraint_name}",
            "controller=ramp_net",
            f"controller.model_dir={model_dir}",
            f"hydra.run.dir={output_dir}/experiments/{sc_name}",
        ]
        result = subprocess.run(cmd, cwd=orig_cwd, env=child_env)
        if result.returncode != 0:
            log.error(f"Experiment {sc_name} failed with return code {result.returncode}")
        else:
            log.info(f"Experiment {sc_name} completed successfully.")


if __name__ == "__main__":
    main()
