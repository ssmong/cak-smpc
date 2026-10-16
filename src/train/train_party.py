import os
import logging
import torch
import wandb
import numpy as np
from tqdm import tqdm

from src.losses import party_loss
from src.utils.plot_utils import save_trajectory_plots

log = logging.getLogger(__name__)


def train(
    cfg, model,
    train_loader, val_loader, test_loader,
    norm_mean, norm_std,
    optimizer, scheduler,
    device,
    output_dir,
):
    epochs = cfg.train.epochs
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    # wandb: epoch metrics use 'epoch' as x-axis (not global_step)
    wandb.define_metric("epoch")
    wandb.define_metric("train_avg/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    total_skipped = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = {}
        valid_train_batches = 0
        epoch_skipped = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            batch = {
                k: (v if v.device.type == device.type else v.to(device, non_blocking=True))
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()

            x_seq_hat, z_seq_hat, A, B, z_mu_logvar, ctx, raw_qnorms = model(
                batch['hist_x'],
                batch['hist_u'],
                batch['u_seq'],
                return_all=True
            )

            loss, loss_dict = party_loss(
                model=model,
                x_seq_hat=x_seq_hat,
                z_seq_hat=z_seq_hat,
                x_seq_gt=batch['x_seq'],
                A=A,
                z_mu_logvar=z_mu_logvar,
                weight_cfg=cfg.train.weights,
                raw_qnorms=raw_qnorms,
                hist_x=batch['hist_x'],
                hist_u=batch['hist_u'],
            )

            loss_val = loss.item()
            if not torch.isfinite(loss):
                log.warning(f"Batch {batch_idx}: Loss is {loss_val} (NaN/Inf). Skipping batch.")
                optimizer.zero_grad()
                epoch_skipped += 1
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            # Logging
            loss_dict['total'] = loss
            step_log = {f"train/{k}": v.item() for k, v in loss_dict.items()}
            step_log['epoch'] = epoch
            step_log['train/grad_norm'] = grad_norm.item()
            step_log['train/lr'] = optimizer.param_groups[0]['lr']
            wandb.log(step_log, step=global_step)
            global_step += 1

            # Accumulate for epoch average
            for k, v in loss_dict.items():
                train_loss_sum[k] = train_loss_sum.get(k, 0.0) + v.item()
            valid_train_batches += 1
            
            # Update tqdm progress bar with key metrics
            avg_tot = train_loss_sum['total'] / valid_train_batches
            avg_px = train_loss_sum['pred_x'] / valid_train_batches
            avg_pz = train_loss_sum['pred_z'] / valid_train_batches
            avg_rz = train_loss_sum['recon_z'] / valid_train_batches
            
            pbar.set_postfix({
                'tot': f"{avg_tot:.6f}",
                'px': f"{avg_px:.6f}",
                'pz': f"{avg_pz:.6f}",
                'rz': f"{avg_rz:.6f}"
            })

        # Epoch average
        total_skipped += epoch_skipped
        if valid_train_batches == 0:
            raise RuntimeError(f"No valid training batches at epoch {epoch+1}")
        avg_train_loss = {k: v / valid_train_batches for k, v in train_loss_sum.items()}
        if epoch_skipped > 0:
            log.info(f"Epoch {epoch+1}: skipped {epoch_skipped} NaN/Inf batches (total: {total_skipped})")

        # Validation
        avg_val_loss = evaluate(model, val_loader, device, cfg.train.weights)

        # Scheduler step
        scheduler.step(avg_val_loss['total'])

        # Epoch logging (x-axis: epoch, via wandb.define_metric)
        epoch_log = {f"train_avg/{k}": v for k, v in avg_train_loss.items()}
        epoch_log.update({f"val/{k}": v for k, v in avg_val_loss.items()})
        epoch_log['epoch'] = epoch
        epoch_log['train_avg/skipped_batches'] = epoch_skipped
        wandb.log(epoch_log)

        # Plot trajectory (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            save_trajectory_plots(
                model, test_loader, device, cfg.data.pred_dt, cfg.data.quat_indices,
                norm_mean, norm_std, output_dir, tag=f"epoch_{epoch+1}"
            )

        # Save best model & early stopping
        if avg_val_loss['total'] < best_loss:
            best_loss = avg_val_loss['total']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_loss,
            }, os.path.join(output_dir, "best_model.pt"))
            is_best = True
        else:
            patience_counter += 1
            is_best = False
            if patience_counter >= cfg.train.early_stopping_patience:
                log.info(f"Early stopping triggered at epoch {epoch+1} (patience: {cfg.train.early_stopping_patience})")
                break

        # Readable epoch summary table (train/val loss components).
        _key_order = ['total', 'pred_x', 'pred_z', 'nll', 'recon_z', 'svd', 'qnorm', 'metric', 'geodesic', 'P0', 'ctx_smooth']
        all_keys = [k for k in _key_order if k in avg_train_loss or k in avg_val_loss]

        metric_width = 12
        val_width = 11
        log_msg = []
        log_msg.append(f"Epoch {epoch+1}/{epochs}")
        
        # Table Header
        header = f"{'Type':<{metric_width}} | " + " | ".join([f"{k:<{val_width}}" for k in all_keys])
        log_msg.append(header)
        log_msg.append("-" * len(header))

        # Train Row (avg_train_metrics 대신 avg_train_loss 직접 사용, 소수점 4자리 통일)
        train_row = f"{'Train':<{metric_width}} | "
        train_vals = []
        for k in all_keys:
            val = avg_train_loss.get(k, float('nan'))
            train_vals.append(f"{val:<{val_width}.6f}")
        train_row += " | ".join(train_vals)
        log_msg.append(train_row)

        val_label = "Val   [BEST]" if is_best else "Val"
        val_row = f"{val_label:<{metric_width}} | "
        val_vals = []
        for k in all_keys:
            val = avg_val_loss.get(k, float('nan'))
            val_vals.append(f"{val:<{val_width}.6f}")
        val_row += " | ".join(val_vals)
        log_msg.append(val_row)

        log.info("\n".join(log_msg))

    return best_epoch, best_loss, global_step

def evaluate(model, loader, device, weight_cfg):
    model.eval()
    loss_sum = {}

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v if v.device.type == device.type else v.to(device, non_blocking=True))
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            x_seq_hat, z_seq_hat, A, B, z_mu_logvar, ctx, raw_qnorms = model(
                batch['hist_x'],
                batch['hist_u'],
                batch['u_seq'],
                return_all=True
            )

            loss, loss_dict = party_loss(
                model=model,
                x_seq_hat=x_seq_hat,
                z_seq_hat=z_seq_hat,
                x_seq_gt=batch['x_seq'],
                A=A,
                z_mu_logvar=z_mu_logvar,
                weight_cfg=weight_cfg,
                raw_qnorms=raw_qnorms,
                hist_x=batch['hist_x'],
                hist_u=batch['hist_u'],
            )

            loss_dict['total'] = loss
            for k, v in loss_dict.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + v.item()

    return {k: v / len(loader) for k, v in loss_sum.items()}


def save_W_offline(model, test_loader, device, stats_dir):
    """
    Compute covariance of 1-step latent modeling error: e = z_next_true - (A @ z + B @ u).
    Saves W_offline.npy for use in party_smpc.
    """
    model.eval()
    dataset = test_loader.dataset
    hist_len = dataset.hist_len
    hist_stride = int(round(dataset.hist_dt / dataset.log_dt))
    pred_stride = int(round(dataset.pred_dt / dataset.log_dt))

    # History offsets (same as collator)
    hist_x_offsets = list(range(-hist_len * hist_stride, 1, hist_stride))      # len = hist_len + 1
    hist_u_offsets = list(range(-hist_len * hist_stride, 0, hist_stride))       # len = hist_len

    all_errors = []

    with torch.no_grad():
        for traj_idx in dataset.sequence_indices:
            x_traj = dataset.x_traj_list[traj_idx]   # (T, x_dim)
            u_traj = dataset.u_traj_list[traj_idx]   # (T, u_dim)
            T = x_traj.shape[0]

            min_k = hist_len * hist_stride
            max_k = T - 1 - pred_stride   # need x[k + pred_stride]

            if max_k < min_k:
                continue

            k_indices = list(range(min_k, max_k + 1, pred_stride))

            # Process in mini-batches to avoid OOM
            batch_size = 256
            for b_start in range(0, len(k_indices), batch_size):
                ks = k_indices[b_start:b_start + batch_size]

                hist_x = torch.stack([x_traj[[k + o for o in hist_x_offsets]] for k in ks])  # (B, H+1, x_dim)
                hist_u = torch.stack([u_traj[[k + o for o in hist_u_offsets]] for k in ks])  # (B, H, u_dim)
                x_curr = torch.stack([x_traj[k] for k in ks])                                # (B, x_dim)
                x_next = torch.stack([x_traj[k + pred_stride] for k in ks])                   # (B, x_dim)
                u_curr = torch.stack([u_traj[k] for k in ks])                                 # (B, u_dim)

                hist_x = hist_x.to(device, non_blocking=True)
                hist_u = hist_u.to(device, non_blocking=True)
                x_curr = x_curr.to(device, non_blocking=True)
                x_next = x_next.to(device, non_blocking=True)
                u_curr = u_curr.to(device, non_blocking=True)

                ctx = model.get_ctx(hist_x, hist_u)
                z_mu, _ = model.encode(x_curr)
                z_next_true, _ = model.encode(x_next)
                A, B = model.get_AB(ctx)
                z_next_pred = model.step(z_mu, u_curr, A, B)
                e = (z_next_true - z_next_pred).cpu().numpy()
                all_errors.append(e)

    all_errors = np.vstack(all_errors)
    W = np.cov(all_errors, rowvar=False)
    W = 0.5 * (W + W.T)

    os.makedirs(stats_dir, exist_ok=True)
    out_path = os.path.join(stats_dir, "W_offline.npy")
    np.save(out_path, W.astype(np.float64))

    log.info(f"Saved W_offline to {out_path} (shape {W.shape}, trace={np.trace(W):.6e}, "
             f"{all_errors.shape[0]} samples)")
