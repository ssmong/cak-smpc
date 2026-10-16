import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.math_torch import quat_angle_error

log = logging.getLogger(__name__)


def _unwrap_quaternion(q_seq):
    """
    q_seq: (T, 4) 
    If the quaternion at time t is opposite to the quaternion at time t-1, 
    invert the sign to maintain continuity.
    """
    q_out = q_seq.clone()
    for t in range(1, q_out.shape[0]):
        if torch.sum(q_out[t] * q_out[t-1]) < 0:
            q_out[t] = -q_out[t]
    return q_out


def plot_trajectory(pred_dt, x_seq_hat, x_seq_gt, quat_indices, save_path=None, mean=None, std=None):
    """
    Plot predicted vs ground truth trajectories.
    
    Args:
        pred_dt: Time step for prediction
        x_seq_hat: (T, state_dim) Predicted trajectory
        x_seq_gt: (T, state_dim) Ground truth trajectory
        quat_indices: List of raw indices where quaternions start (will consume 4 cols each)
        save_path: Path to save the figure
        mean: Optional normalization mean (for denormalizing non-quat states)
        std: Optional normalization std
    """
    x_seq_hat = x_seq_hat.detach()
    x_seq_gt = x_seq_gt.detach()

    if mean is not None and std is not None:
        device = x_seq_hat.device
        mean, std = mean.to(device), std.to(device)

    plots = [] 
    quat_set = set(quat_indices) if quat_indices else set()
    current_col, raw_idx = 0, 0
    
    while current_col < x_seq_hat.shape[-1]:
        if raw_idx in quat_set:
            q_pred = _unwrap_quaternion(x_seq_hat[:, current_col:current_col+4])
            q_gt = _unwrap_quaternion(x_seq_gt[:, current_col:current_col+4])

            if torch.sum(q_pred[0] * q_gt[0]) < 0:
                q_pred = -q_pred

            err = torch.rad2deg(quat_angle_error(q_pred, q_gt))
            plots.append((f"State {raw_idx} (Quat Error)", err, None, "Degree"))

            for i, label in enumerate(['0', '1', '2', '3']):
                plots.append((f"State {raw_idx + i} (q_{label})", q_pred[:, i], q_gt[:, i], "Value"))
            
            current_col += 4
            raw_idx += 4
        else:
            val_p, val_g = x_seq_hat[:, current_col], x_seq_gt[:, current_col]
            if mean is not None:
                val_p = val_p * std[current_col] + mean[current_col]
                val_g = val_g * std[current_col] + mean[current_col]
            plots.append((f"State {raw_idx}", val_p, val_g, "Value"))
            current_col += 1
            raw_idx += 1

    n_plots = len(plots)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    time_steps = np.arange(x_seq_hat.shape[0]) * pred_dt

    for ax, (title, p_seq, g_seq, ylabel) in zip(axes, plots):
        if g_seq is not None:
            ax.plot(time_steps, g_seq.cpu().numpy(), 'k-', label='GT', alpha=0.6)
            ax.plot(time_steps, p_seq.cpu().numpy(), 'r--', label='Pred')
        else:
            ax.plot(time_steps, p_seq.cpu().numpy(), 'b-', label='Error')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def save_trajectory_plots(model, loader, device, pred_dt, quat_indices, mean, std, output_dir, tag, num_plots=10):
    figure_dir = os.path.join(output_dir, "figures", tag)
    os.makedirs(figure_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, batch in zip(range(num_plots), loader):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            x_seq_hat, _ = model(batch['hist_x'], batch['hist_u'], batch['u_seq'], return_all=False)
            
            plot_trajectory(
                pred_dt, x_seq_hat[0], batch['x_seq'][0], quat_indices,
                save_path=os.path.join(figure_dir, f"test_plot_{i}.png"),
                mean=mean, std=std
            )
    
    log.info(f"Saved trajectory plots to {figure_dir}")
