import os
import sys
import subprocess
import logging
import hydra
import wandb
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader

from src.train import train, evaluate, save_W_offline
from src.utils import set_seed, print_model_summary, save_trajectory_plots

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="learning_party", version_base=None)
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       0. Seed & WandB & Output Directory
    # ------------------------------------------------------------------
    set_seed(cfg.data.seed)
    
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
    #       1. Dataset & DataLoader
    # ------------------------------------------------------------------
    log.info("Initializing dataset...")

    stats_dir = os.path.join(output_dir, "stats")
        
    train_dataset = hydra.utils.instantiate(
        cfg.data,
        split="train",
        stats_dir=stats_dir,
        use_loaded_stats=False,
        load_inertia=False,
    )
    shared_data = train_dataset.get_shared_data()
    shared_split_indices = train_dataset.split_indices

    val_dataset = hydra.utils.instantiate(
        cfg.data,
        split="val",
        stats_dir=stats_dir,
        use_norm=False,
        load_inertia=False,
        shared_data=shared_data,
        shared_split_indices=shared_split_indices,
    )
    test_dataset = hydra.utils.instantiate(
        cfg.data,
        split="test",
        stats_dir=stats_dir,
        use_norm=False,
        load_inertia=False,
        shared_data=shared_data,
        shared_split_indices=shared_split_indices,
    )

    data_on_cuda = getattr(train_dataset, "storage_device", torch.device("cpu")).type == "cuda"

    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": not data_on_cuda,
        "persistent_workers": False,
    }

    log.info(
        "DataLoader config - data_on_cuda=%s, pin_memory=%s",
        data_on_cuda,
        loader_kwargs["pin_memory"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(
            min_steps=cfg.train.train_min_steps,
            max_steps=cfg.train.train_max_steps,
        ),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=val_dataset.get_collate_fn(
            rollout_steps=cfg.train.valid_steps,
        ),
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=test_dataset.get_collate_fn(
            rollout_steps=cfg.train.test_steps,
        ),
        drop_last=False,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    #       2. Model & Optimizer & Scheduler
    # ------------------------------------------------------------------
    log.info("Initializing model...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")
    
    model = hydra.utils.instantiate(cfg.model).to(device)

    # Fine-tuning: load pretrained weights if specified
    if cfg.train.pretrained_model_dir:
        weights_path = os.path.join(
            "outputs/learning_party", cfg.train.pretrained_model_dir, "best_model.pt"
        )
        log.info("Loading pretrained model from %s", weights_path)
        ckpt = torch.load(weights_path, weights_only=False)
        state_dict = {
            k.replace('log_var_head', 'logvar_head'): v
            for k, v in ckpt['model_state_dict'].items()
        }
        model.load_state_dict(state_dict)

        if cfg.train.reset_logvar_head:
            head = model.vae_encoder.logvar_head
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -10.0)
            log.info("Reset logvar_head (weight=0, bias=-10.0, P0_init≈%.2e, std≈%.4f)",
                     torch.exp(torch.tensor(-10.0)).item(),
                     torch.exp(torch.tensor(-5.0)).item())

    print_model_summary(model)

    if cfg.train.pretrained_model_dir and cfg.train.get("finetune_lr_scale"):
        scales = cfg.train.finetune_lr_scale
        base_lr = cfg.train.lr
        param_groups = []
        grouped_params = set()

        # logvar_head
        logvar_params = list(model.vae_encoder.logvar_head.parameters())
        if scales.logvar_head > 0:
            param_groups.append({"params": logvar_params, "lr": base_lr * scales.logvar_head})
        grouped_params.update(id(p) for p in logvar_params)

        # encoder (excluding logvar_head)
        encoder_params = [p for p in model.vae_encoder.parameters() if id(p) not in grouped_params]
        if scales.encoder > 0:
            param_groups.append({"params": encoder_params, "lr": base_lr * scales.encoder})
        grouped_params.update(id(p) for p in encoder_params)

        # decoder
        if scales.decoder > 0:
            param_groups.append({"params": list(model.vae_decoder.parameters()), "lr": base_lr * scales.decoder})
        grouped_params.update(id(p) for p in model.vae_decoder.parameters())

        # context_encoder
        if scales.context_encoder > 0:
            param_groups.append({"params": list(model.context_encoder.parameters()), "lr": base_lr * scales.context_encoder})
        grouped_params.update(id(p) for p in model.context_encoder.parameters())

        # koopman
        if scales.koopman > 0:
            param_groups.append({"params": list(model.koopman.parameters()), "lr": base_lr * scales.koopman})
        grouped_params.update(id(p) for p in model.koopman.parameters())

        # any remaining params
        remaining = [p for p in model.parameters() if id(p) not in grouped_params]
        if remaining:
            param_groups.append({"params": remaining, "lr": base_lr})

        log.info("Finetune LR scales: logvar=%.1f, enc=%.2f, dec=%.2f, ctx=%.2f, koop=%.3f",
                 scales.logvar_head, scales.encoder, scales.decoder,
                 scales.context_encoder, scales.koopman)
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=cfg.train.scheduler_patience)
    
    # ------------------------------------------------------------------
    #       3. Training
    # ------------------------------------------------------------------
    log.info("Starting training...")

    norm_mean, norm_std = train_dataset.get_stats()
    train_stats = train_dataset.processor.get_stats()
    fmt_stats = {
        k: [round(float(v), 4) for v in t.detach().cpu().tolist()]
        for k, t in train_stats.items()
    }
    log.info(
        "Train normalization stats (single print): "
        "mean=%s std=%s min=%s max=%s u_mean=%s u_std=%s u_min=%s u_max=%s",
        fmt_stats["mean"],
        fmt_stats["std"],
        fmt_stats["min"],
        fmt_stats["max"],
        fmt_stats["u_mean"],
        fmt_stats["u_std"],
        fmt_stats["u_min"],
        fmt_stats["u_max"],
    )

    best_epoch, best_loss, global_step = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        norm_mean=norm_mean,
        norm_std=norm_std,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device, output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    #       4. Test (using best model)
    # ------------------------------------------------------------------
    log.info(f"Loading best model from epoch {best_epoch+1}...")
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    avg_test_loss = evaluate(model, test_loader, device, cfg.train.weights)

    test_log = {f"test/{k}": v for k, v in avg_test_loss.items()}
    wandb.log(test_log, step=global_step)

    log.info(f"Test Loss: {avg_test_loss['total']:.4f}")
    log.info(f"Training complete. Best epoch: {best_epoch+1}, Best val loss: {best_loss:.4f}")

    # Final trajectory plots (using best model)
    save_trajectory_plots(
        model, test_loader, device, cfg.data.pred_dt, cfg.data.quat_indices,
        norm_mean, norm_std, output_dir, tag="final_best"
    )

    # 4.2. Save 1-step latent modeling error covariance (for party_smpc W_offline)
    save_W_offline(model, test_loader, device, stats_dir)

    # 4.2.1. Diagnose W_offline & Koopman A spectral radius
    import numpy as np
    W_path = os.path.join(stats_dir, "W_offline.npy")
    W = np.load(W_path)
    eigvals_W = np.linalg.eigvalsh(W)
    log.info(
        "W_offline diagnostics: shape=%s, trace=%.6e, max_eigval=%.6e, "
        "min_eigval=%.6e, cond=%.1f",
        W.shape, np.trace(W), eigvals_W.max(), eigvals_W.min(),
        eigvals_W.max() / max(eigvals_W.min(), 1e-15),
    )

    # Sample A spectral radii from test set
    model.eval()
    spectral_radii = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v if v.device.type == device.type else v.to(device))
                for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            ctx = model.get_ctx(batch["hist_x"], batch["hist_u"])
            A_batch, _ = model.get_AB(ctx)
            for i in range(A_batch.shape[0]):
                A_np = A_batch[i].cpu().numpy()
                sr = np.max(np.abs(np.linalg.eigvals(A_np)))
                spectral_radii.append(sr)
    spectral_radii = np.array(spectral_radii)
    log.info(
        "Koopman A spectral radius: mean=%.4f, max=%.4f, min=%.4f, "
        "std=%.4f, pct>1.0=%.2f%%",
        spectral_radii.mean(), spectral_radii.max(), spectral_radii.min(),
        spectral_radii.std(), 100.0 * np.mean(spectral_radii > 1.0),
    )

    # 4.3. Long-term stability test
    long_term_collate_fn = test_dataset.get_collate_fn(
        rollout_steps=cfg.train.long_term_steps,
    )

    long_term_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=long_term_collate_fn,
        drop_last=False,
        **loader_kwargs,
    )

    save_trajectory_plots(
        model, long_term_loader, device, cfg.data.pred_dt, cfg.data.quat_indices,
        norm_mean, norm_std, output_dir, tag="long_term"
    )

    wandb.finish()

    # ------------------------------------------------------------------
    #       5. Run Simulation Experiments (using trained model)
    # ------------------------------------------------------------------
    if not cfg.train.run_sim:
        log.info("Simulation skipped (train.run_sim=false).")
        return

    # Compute model_dir relative to "outputs/learning_party/" for party_smpc
    learning_base = os.path.join(orig_cwd, "outputs", "learning_party")
    model_dir = os.path.relpath(output_dir, learning_base)

    experiments = [
        ("exp1a",    "exp1a", "Constant MOI (In-Distribution)"),
        ("exp1b", "exp1b", "Constant MOI (Non-Diagonal)"),
        ("exp3",     "exp3", "Constant MOI (OOD)"),
        ("exp2a",        "exp2a", "Continuous MOI Drift"),
        ("exp2b",        "exp2b", "Abrupt MOI Change"),
    ]

    # Strip WANDB_* env vars so child processes don't create spurious sweep runs
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}

    for sc_name, constraint_name, desc in experiments:
        log.info(f"Running experiment: {desc} (sc={sc_name}, constraint={constraint_name})")
        cmd = [
            sys.executable, "-m", "scripts.3_run_sim",
            "--config-name=sim_party",
            f"sc={sc_name}",
            f"constraint={constraint_name}",
            "controller=party_smpc",
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
