"""
RAMP-Net training loop using L-BFGS optimizer (full-batch).

Reference:
    Sanyal & Roy, "RAMP-Net: A Robust Adaptive MPC for Quadrotors via
    Physics-informed Neural Network", ICRA 2023, Section V-A.

    "We used the entire dataset as a single batch using the memory-efficient
     quasi-newton L-BFGS optimizer following [12], [13]."
"""

import os
import time
import logging
from typing import Optional

import torch
import wandb

from src.model.ramp_net import RampNet
from src.losses.ramp_loss import ramp_loss

log = logging.getLogger(__name__)


def train_ramp(
    model: RampNet,
    train_batch: dict[str, torch.Tensor],
    val_batch: dict[str, torch.Tensor],
    J_diag: torch.Tensor,
    noise_std: float,
    dt: float,
    epochs: int,
    lr: float,
    max_lbfgs_iter: int,
    history_size: int,
    early_stopping_patience: int,
    output_dir: str,
    device: torch.device,
) -> tuple[int, float]:
    """
    Train RAMP-Net using L-BFGS full-batch optimization.

    Args:
        model: RampNet model
        train_batch: Dict with x_phys, u_phys, x_data, u_data, y_data (train)
        val_batch: Dict with x_phys, u_phys, x_data, u_data, y_data (val)
        J_diag: Nominal inertia diagonal [3] tensor
        noise_std: Parametric uncertainty noise std
        dt: MPC time horizon T
        epochs: Number of training epochs
        lr: L-BFGS learning rate
        max_lbfgs_iter: L-BFGS max iterations per step
        history_size: L-BFGS history size
        early_stopping_patience: Early stopping patience
        output_dir: Directory to save best model
        device: Torch device

    Returns:
        (best_epoch, best_val_loss)
    """
    model.to(device)
    J_diag = J_diag.to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_lbfgs_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training step (L-BFGS closure) ----
        model.train()
        train_loss_dict: Optional[dict] = None

        # Per-epoch timing (GPU-synchronized for accuracy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # Fix random seed per epoch so that all L-BFGS closure calls within
        # a single optimizer.step() produce identical noise (t, delta_J).
        # This satisfies L-BFGS's deterministic function assumption while
        # preserving noise diversity across epochs (different seed each epoch).
        epoch_rng_seed = torch.randint(0, 2**31, (1,)).item()

        def closure():
            nonlocal train_loss_dict
            optimizer.zero_grad()
            loss_dict = ramp_loss(
                model=model,
                x_phys=train_batch["x_phys"],
                u_phys=train_batch["u_phys"],
                x_data=train_batch["x_data"],
                u_data=train_batch["u_data"],
                y_data=train_batch["y_data"],
                J_diag=J_diag,
                noise_std=noise_std,
                dt=dt,
                rng_seed=epoch_rng_seed,
            )
            loss = loss_dict["total"]
            loss.backward()
            train_loss_dict = {k: v.item() for k, v in loss_dict.items()}
            return loss

        optimizer.step(closure)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            # For validation, data loss only (no physics autograd needed)
            batch_size = val_batch["x_data"].shape[0]
            t_val = torch.full(
                (batch_size, 1), dt, device=device, dtype=val_batch["x_data"].dtype,
            )
            x_pred_val = model(val_batch["x_data"], val_batch["u_data"], t_val)
            val_data_loss = torch.mean((x_pred_val - val_batch["y_data"]) ** 2).item()

        # ---- Per-epoch timing ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        # ---- Logging ----
        wandb.log({
            "train/total": train_loss_dict["total"],
            "train/physics": train_loss_dict["physics"],
            "train/data": train_loss_dict["data"],
            "val/data": val_data_loss,
            "train/epoch_time_s": epoch_time,
            "epoch": epoch,
        })

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.info(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"train: total={train_loss_dict['total']:.6f} "
                f"phys={train_loss_dict['physics']:.6f} "
                f"data={train_loss_dict['data']:.6f} | "
                f"val_data={val_data_loss:.6f}"
            )

        # ---- Early stopping ----
        if val_data_loss < best_val_loss:
            best_val_loss = val_data_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_data_loss,
                },
                os.path.join(output_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            log.info(
                f"Early stopping at epoch {epoch+1}. "
                f"Best epoch: {best_epoch+1}, best val loss: {best_val_loss:.6f}"
            )
            break

    log.info(f"Training complete. Best epoch: {best_epoch+1}, best val loss: {best_val_loss:.6f}")
    return best_epoch, best_val_loss
