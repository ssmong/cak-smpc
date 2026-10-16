import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sim.koz import KeepoutZone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sim.utils.math_utils import quat2boresight_batch
from sim.montecarlo.mc_utils import unwrap_quaternions

log = logging.getLogger(__name__)


def plot_sim_results(x_traj: np.ndarray,
                     u_traj: np.ndarray,
                     x_goal: np.ndarray,
                     koz_object: List[KeepoutZone],
                     boresight_list: List[Dict] | None,
                     u_min: float,
                     u_max: float,
                     log_dt: float,
                     figure_dir: str,
                     save_prefix: str,
                     print_summary: bool = True):
    """
    Args:
        x_traj: (T+1, 7) [q_w, q_x, q_y, q_z, w_x, w_y, w_z]
        u_traj: (T+1, 3)   [u1, u2, u3]
        x_goal: (7,)
        koz_object: KeepoutZone list
        boresight_list: list of {"name": str, "vec": [3]} body-frame boresight vectors
        u_min: control lower bound (scalar)
        u_max: control upper bound (scalar)
        log_dt: logging time step, e.g. 0.1
        figure_dir: directory to save figures, e.g. "outputs/.../figures"
        save_prefix: save file name prefix, e.g. "SMPC"
    """
    T, _ = x_traj.shape

    time_vec = np.arange(T) * log_dt

    # Unwrap quaternion trajectory to remove double-cover sign flips for plotting
    q_traj_plot = unwrap_quaternions(x_traj[:, :4])

    q_goal = x_goal[:4] / np.linalg.norm(x_goal[:4])
    # Flip q_goal sign to match the hemisphere of the final unwrapped quaternion
    q_final = q_traj_plot[-1] / np.linalg.norm(q_traj_plot[-1])
    if np.dot(q_final, q_goal) < 0:
        q_goal = -q_goal

    # === 1. Quaternion components vs Time plot ===
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Quaternion Components vs Time', fontsize=16)
    q_labels = ['q_w (scalar)', 'q_x', 'q_y', 'q_z']
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        ax = axes1[i//2, i%2]
        ax.plot(time_vec, q_traj_plot[:, i], color=colors[i], linewidth=2, label=f'Actual {q_labels[i]}')
        ax.axhline(q_goal[i], color=colors[i], linestyle='--', alpha=0.7, label=f'Target {q_labels[i]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(q_labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Quaternion Component: {q_labels[i]}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{save_prefix}_quat.png'), dpi=200, bbox_inches='tight')
    log.info(f"Quaternion components plot saved: {save_prefix}_quat.png")
    plt.close(fig1)

    u_min_val = float(u_min)
    u_max_val = float(u_max)
    if u_min_val > u_max_val:
        raise ValueError(f"Control bounds invalid: u_min={u_min_val} > u_max={u_max_val}.")

    # === 2. 3D Boresight trajectory with KOZ ===

    boresight_trajs = {}   # name -> (N, 3)
    boresight_goals = {}   # name -> (3,)
    for bs_cfg in boresight_list:
        name = bs_cfg["name"] if isinstance(bs_cfg, dict) else bs_cfg.name
        vec = bs_cfg["vec"] if isinstance(bs_cfg, dict) else list(bs_cfg.vec)
        boresight_trajs[name] = quat2boresight_batch(x_traj[:, :4], np.array(vec))
        boresight_goals[name] = quat2boresight_batch(q_goal.reshape(1, -1), np.array(vec))[0]

    fig3 = plt.figure(figsize=(15, 12))
    ax3 = fig3.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.28, color='gray', linewidth=1.0)
    ax3.set_box_aspect([1, 1, 1])

    traj_colors = ['blue', 'orange', 'green', 'purple', 'red', 'gray']
    traj_markers = ['D', 's', 'p', 'h', '8', '*']
    for idx, (bs_name, bs_traj) in enumerate(boresight_trajs.items()):
        show_common_markers = (idx == 0)
        color = traj_colors[idx % len(traj_colors)]
        marker = traj_markers[idx % len(traj_markers)]
        ax3.plot(bs_traj[:, 0], bs_traj[:, 1], bs_traj[:, 2],
                 color=color, linewidth=3.0,
                 label=f'{bs_name} Trajectory ({time_vec[0]:.1f}s->{time_vec[-1]:.1f}s)')
        ax3.scatter(*bs_traj[0], color=color, s=70, marker='o',
                    label=f'Start ({time_vec[0]:.1f}s)' if show_common_markers else '_nolegend_',
                    zorder=10)
        ax3.scatter(*boresight_goals[bs_name], color='gold', s=180, marker='*',
                    edgecolors='black', linewidths=0.8,
                    label='Target' if show_common_markers else '_nolegend_',
                    zorder=10)
        ax3.scatter(*bs_traj[-1], color=color, s=100, marker=marker,
                    label=f'Final ({time_vec[-1]:.1f}s)' if show_common_markers else '_nolegend_',
                    zorder=10)

    koz_colors = ['yellow', 'lightcoral', 'orange', 'lightgreen']
    for i, koz in enumerate(koz_object):
        color = koz_colors[i % len(koz_colors)]
        direction = np.asarray(koz.direction, dtype=float)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            raise ValueError(f"KOZ direction for '{koz.name}' is near-zero and cannot be normalized.")
        direction = direction / norm_dir

        ax3.scatter(*direction, color=color, s=80, marker='^', alpha=0.9, zorder=8)
        angle_rad = np.deg2rad(koz.angle_deg)
        cone_height = np.cos(angle_rad)
        cone_radius = np.sin(angle_rad)
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        theta_cone = np.linspace(0, 2*np.pi, 64)
        phi_cap = np.linspace(0, angle_rad, 32)
        theta_grid, phi_grid = np.meshgrid(theta_cone, phi_cap)
        cap_x = (np.cos(phi_grid) * direction[0] +
                 np.sin(phi_grid) * (np.cos(theta_grid) * perp1[0] + np.sin(theta_grid) * perp2[0]))
        cap_y = (np.cos(phi_grid) * direction[1] +
                 np.sin(phi_grid) * (np.cos(theta_grid) * perp1[1] + np.sin(theta_grid) * perp2[1]))
        cap_z = (np.cos(phi_grid) * direction[2] +
                 np.sin(phi_grid) * (np.cos(theta_grid) * perp1[2] + np.sin(theta_grid) * perp2[2]))
        ax3.plot_surface(cap_x, cap_y, cap_z, color=color, alpha=0.22, linewidth=0, shade=False, zorder=3)

        cone_points = []
        for t in theta_cone:
            boundary_dir = (cone_height * direction +
                            cone_radius * (np.cos(t) * perp1 + np.sin(t) * perp2))
            boundary_dir = boundary_dir / np.linalg.norm(boundary_dir)
            cone_points.append(boundary_dir)
        cone_points = np.array(cone_points)
        ax3.plot(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2],
                 color=color, alpha=0.95, linewidth=2.5, label=f'{koz.name} ({koz.angle_deg})')
        ax3.plot([0.0, direction[0]], [0.0, direction[1]], [0.0, direction[2]],
                 color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        label_pos = direction * 1.3
        ax3.text(label_pos[0], label_pos[1], label_pos[2],
                 f'{koz.name}\n{koz.angle_deg}', fontsize=10, ha='center')

    ax3.set_xlim([-1.3, 1.3])
    ax3.set_ylim([-1.3, 1.3])
    ax3.set_zlim([-1.3, 1.3])
    ax3.set_xlabel('X (Inertial)')
    ax3.set_ylabel('Y (Inertial)')
    ax3.set_zlabel('Z (Inertial)')
    ax3.set_title('3D Boresight Trajectory with Keep-Out Zones', fontsize=14)
    handles, labels = ax3.get_legend_handles_labels()
    unique_legend = {}
    for h, l in zip(handles, labels):
        if l and l != '_nolegend_' and l not in unique_legend:
            unique_legend[l] = h
    ax3.legend(
        list(unique_legend.values()),
        list(unique_legend.keys()),
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
    )
    ax3.grid(False)
    ax3.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax3.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{save_prefix}_3d_boresight.png'), dpi=200, bbox_inches='tight')
    log.info(f"3D boresight plot saved: {save_prefix}_3d_boresight.png")
    plt.close(fig3)

    # === 3. Interactive 3D boresight (Plotly HTML) ===
    try:
        import plotly.graph_objects as go

        fig_html = go.Figure()

        # Unit sphere wireframe
        _u = np.linspace(0, 2 * np.pi, 36)
        _v = np.linspace(0, np.pi, 18)
        sx = np.outer(np.cos(_u), np.sin(_v))
        sy = np.outer(np.sin(_u), np.sin(_v))
        sz = np.outer(np.ones_like(_u), np.cos(_v))
        fig_html.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            opacity=0.08, colorscale=[[0, 'gray'], [1, 'gray']],
            showscale=False, hoverinfo='skip',
        ))

        # Boresight trajectories
        _colors = ['blue', 'orange', 'green', 'purple', 'red', 'gray']
        for idx, (bs_name, bs_traj) in enumerate(boresight_trajs.items()):
            color = _colors[idx % len(_colors)]
            fig_html.add_trace(go.Scatter3d(
                x=bs_traj[:, 0], y=bs_traj[:, 1], z=bs_traj[:, 2],
                mode='lines', line=dict(color=color, width=4),
                name=f'{bs_name} trajectory',
                customdata=time_vec,
                hovertemplate='t=%{customdata:.1f}s<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})',
            ))
            fig_html.add_trace(go.Scatter3d(
                x=[bs_traj[0, 0]], y=[bs_traj[0, 1]], z=[bs_traj[0, 2]],
                mode='markers', marker=dict(color=color, size=5, symbol='circle'),
                name=f'{bs_name} start', showlegend=False,
            ))
            fig_html.add_trace(go.Scatter3d(
                x=[bs_traj[-1, 0]], y=[bs_traj[-1, 1]], z=[bs_traj[-1, 2]],
                mode='markers', marker=dict(color=color, size=6, symbol='diamond'),
                name=f'{bs_name} final', showlegend=False,
            ))
            goal_pt = boresight_goals[bs_name]
            fig_html.add_trace(go.Scatter3d(
                x=[goal_pt[0]], y=[goal_pt[1]], z=[goal_pt[2]],
                mode='markers', marker=dict(color='gold', size=8, symbol='x',
                                            line=dict(color='black', width=1)),
                name=f'{bs_name} target',
            ))

        # KOZ cones
        _koz_colors = ['rgba(255,255,0,0.25)', 'rgba(255,100,100,0.25)',
                        'rgba(255,165,0,0.25)', 'rgba(144,238,144,0.25)']
        for i, koz in enumerate(koz_object):
            kcolor = _koz_colors[i % len(_koz_colors)]
            direction = np.asarray(koz.direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            angle_rad = np.deg2rad(koz.angle_deg)

            if abs(direction[2]) < 0.9:
                perp1 = np.cross(direction, [0, 0, 1])
            else:
                perp1 = np.cross(direction, [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)

            theta_c = np.linspace(0, 2 * np.pi, 48)
            phi_c = np.linspace(0, angle_rad, 16)
            th_g, ph_g = np.meshgrid(theta_c, phi_c)
            cx = np.cos(ph_g) * direction[0] + np.sin(ph_g) * (np.cos(th_g) * perp1[0] + np.sin(th_g) * perp2[0])
            cy = np.cos(ph_g) * direction[1] + np.sin(ph_g) * (np.cos(th_g) * perp1[1] + np.sin(th_g) * perp2[1])
            cz = np.cos(ph_g) * direction[2] + np.sin(ph_g) * (np.cos(th_g) * perp1[2] + np.sin(th_g) * perp2[2])
            fig_html.add_trace(go.Surface(
                x=cx, y=cy, z=cz,
                opacity=0.25, colorscale=[[0, kcolor.replace('0.25', '1')], [1, kcolor.replace('0.25', '1')]],
                showscale=False, hoverinfo='skip', name=f'{koz.name} ({koz.angle_deg}\u00b0)',
            ))

        fig_html.update_layout(
            title=f'{save_prefix} — Interactive Boresight Trajectory',
            scene=dict(
                xaxis_title='X (Inertial)', yaxis_title='Y (Inertial)', zaxis_title='Z (Inertial)',
                aspectmode='cube',
                xaxis=dict(range=[-1.3, 1.3]),
                yaxis=dict(range=[-1.3, 1.3]),
                zaxis=dict(range=[-1.3, 1.3]),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        html_path = os.path.join(figure_dir, f'{save_prefix}_3d_boresight.html')
        fig_html.write_html(html_path, include_plotlyjs='cdn')
        log.info(f"Interactive boresight plot saved: {html_path}")
    except ImportError:
        log.debug("plotly not installed — skipping interactive boresight plot")

    # === 4. KOZ avoidance analysis (per boresight) ===
    num_boresights = len(boresight_trajs)
    fig4, axes4 = plt.subplots(num_boresights, 1, figsize=(15, 8 * num_boresights), squeeze=False)
    fig4.suptitle('Keep-Out Zone Avoidance Analysis', fontsize=16)
    for bs_idx, (bs_name, bs_traj) in enumerate(boresight_trajs.items()):
        ax4 = axes4[bs_idx, 0]
        koz_separations = {koz.name: [] for koz in koz_object}
        for t in range(len(bs_traj)):
            b_t = bs_traj[t]
            for koz in koz_object:
                dot_product = np.clip(np.dot(b_t, koz.direction), -1, 1)
                angle_deg = np.degrees(np.arccos(dot_product))
                koz_separations[koz.name].append(angle_deg)
        for i, koz in enumerate(koz_object):
            color = koz_colors[i % len(koz_colors)]
            separations = koz_separations[koz.name]
            ax4.plot(time_vec, separations, color=color, linewidth=2,
                     label=f'{koz.name} separation')
            ax4.axhline(koz.angle_deg, color=color, linestyle='--', alpha=0.7,
                        label=f'{koz.name} limit ({koz.angle_deg})')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Separation (degrees)')
        ax4.set_title(f'KOZ Avoidance -- {bs_name}', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([0, 180])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{save_prefix}_koz_analysis.png'), dpi=200, bbox_inches='tight')
    log.info(f"KOZ analysis plot saved: {save_prefix}_koz_analysis.png")
    plt.close(fig4)

    # === 5. Tracking error analysis ===
    fig5, axes5 = plt.subplots(2, 1, figsize=(15, 10))
    fig5.suptitle('Tracking Error Analysis', fontsize=16)
    quat_errors = []
    for t in range(len(x_traj)):
        q_current = x_traj[t, :4] / np.linalg.norm(x_traj[t, :4])
        dot_product = np.abs(np.dot(q_current, q_goal))
        dot_product = np.clip(dot_product, 0, 1)
        geodesic_dist = 2 * np.arccos(dot_product)
        quat_errors.append(np.degrees(geodesic_dist))
    axes5[0].plot(time_vec, quat_errors, 'r-', linewidth=2, label='Quaternion Geodesic Error')
    axes5[0].set_xlabel('Time (s)')
    axes5[0].set_ylabel('Angular Error (degrees)')
    axes5[0].set_title('Quaternion Tracking Error (Geodesic Distance)')
    axes5[0].grid(True, alpha=0.3)
    axes5[0].legend()
    omega_errors = np.linalg.norm(x_traj[:, 4:7] - x_goal[4:7], axis=1)
    axes5[1].plot(time_vec, omega_errors, 'b-', linewidth=2, label='Angular Velocity Error')
    axes5[1].set_xlabel('Time (s)')
    axes5[1].set_ylabel('Angular Velocity Error (rad/s)')
    axes5[1].set_title('Angular Velocity Tracking Error')
    axes5[1].grid(True, alpha=0.3)
    axes5[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{save_prefix}_tracking_errors.png'), dpi=200, bbox_inches='tight')
    log.info(f"Tracking errors plot saved: {save_prefix}_tracking_errors.png")
    plt.close(fig5)

    # === 6. Control performance analysis ===
    fig6, axes6 = plt.subplots(2, 2, figsize=(15, 10))
    fig6.suptitle('Control Performance Analysis', fontsize=16)
    control_norms = np.linalg.norm(u_traj, axis=1)
    axes6[0, 0].plot(time_vec, control_norms, 'g-', linewidth=2)
    axes6[0, 0].set_xlabel('Time (s)')
    axes6[0, 0].set_ylabel('||u|| (Nm)')
    axes6[0, 0].set_title('Control Input Magnitude')
    axes6[0, 0].grid(True, alpha=0.3)
    if len(u_traj) > 1:
        control_rates = np.diff(u_traj, axis=0) / log_dt
        control_rate_norms = np.linalg.norm(control_rates, axis=1)
        axes6[0, 1].plot(time_vec[1:], control_rate_norms, 'm-', linewidth=2)
        axes6[0, 1].set_xlabel('Time (s)')
        axes6[0, 1].set_ylabel('||du/dt|| (Nm/s)')
        axes6[0, 1].set_title('Control Rate Magnitude')
        axes6[0, 1].grid(True, alpha=0.3)
    cumulative_energy = np.cumsum(control_norms**2 * log_dt)
    axes6[1, 0].plot(time_vec, cumulative_energy, 'c-', linewidth=2)
    axes6[1, 0].set_xlabel('Time (s)')
    axes6[1, 0].set_ylabel('Cumulative Energy (Nm2*s)')
    axes6[1, 0].set_title('Cumulative Control Energy')
    axes6[1, 0].grid(True, alpha=0.3)
    axes6[1, 1].hist(u_traj.flatten(), bins=30, alpha=0.7, color='orange', density=True)
    axes6[1, 1].axvline(u_max_val, color='red', linestyle='--', alpha=0.7, label='Upper bound')
    axes6[1, 1].axvline(u_min_val, color='red', linestyle='--', alpha=0.7, label='Lower bound')
    axes6[1, 1].set_xlabel('Control Input (Nm)')
    axes6[1, 1].set_ylabel('Probability Density')
    axes6[1, 1].set_title('Control Input Distribution')
    axes6[1, 1].legend()
    axes6[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{save_prefix}_control_analysis.png'), dpi=200, bbox_inches='tight')
    log.info(f"Control analysis plot saved: {save_prefix}_control_analysis.png")
    plt.close(fig6)

    if print_summary:
        print(f"\n{'='*60}")
        print(f"{save_prefix} PERFORMANCE SUMMARY")
        print('='*60)
        print(f"Final quaternion error: {quat_errors[-1]:.3f} deg")
        print(f"Final angular velocity error: {np.degrees(omega_errors[-1]):.3f} deg/s")
        print(f"Total control energy: {cumulative_energy[-1]:.3f} Nm2*s")
        print(f"Mean control effort: {np.mean(control_norms):.3f} Nm")


def plot_W_history(W_history: np.ndarray,
                   mpc_dt: float,
                   figure_dir: str,
                   save_prefix: str,
                   s_history: Optional[np.ndarray] = None):
    """
    Visualize the evolution of the latent covariance matrix W over time.

    Produces two figures:
      1. Scalar summary metrics (trace, log-det, spectral radius, condition number)
      2. Detailed per-dimension analysis (eigenvalue evolution, diagonal heatmap, snapshots)

    Args:
        W_history: (Time, Dim, Dim) numpy array of W matrices over MPC steps
        mpc_dt: MPC prediction time step [s]
        figure_dir: directory to save figures
        save_prefix: file name prefix
        s_history: (Time,) optional scalar W multiplier history (s_final at each MPC step)
    """
    T, D, _ = W_history.shape
    time_axis = np.arange(T) * mpc_dt

    # ---- Precompute metrics ----
    trace_hist = np.trace(W_history, axis1=1, axis2=2)
    frob_hist = np.linalg.norm(W_history, axis=(1, 2), ord='fro')
    eigenvals = np.linalg.eigvalsh(W_history)  # (T, D), ascending
    max_eig = eigenvals[:, -1]
    min_eig = eigenvals[:, 0]
    cond_num = max_eig / (min_eig + 1e-12)

    # Log-determinant (uncertainty ellipsoid volume proxy)
    # sign should always be positive for valid covariance, but handle gracefully
    sign, logdet = np.linalg.slogdet(W_history)
    logdet_valid = np.where(sign > 0, logdet, np.nan)

    # ================================================================
    # Figure 1: Scalar Summary Metrics
    # ================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig1.suptitle('Covariance W: Scalar Summary Over Time', fontsize=15)

    # (a) Trace & Frobenius Norm (+ s_history on twin axis)
    axes[0, 0].plot(time_axis, trace_hist, label='Trace (sum of variances)', color='blue', linewidth=2)
    axes[0, 0].plot(time_axis, frob_hist, label='Frobenius Norm', color='orange', linestyle='--')
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].set_title('Overall Uncertainty Magnitude')
    if s_history is not None and len(s_history) >= T:
        ax_s = axes[0, 0].twinx()
        ax_s.plot(time_axis, s_history[:T], color='red', linewidth=1.5, alpha=0.7, label='s (W multiplier)')
        ax_s.set_ylabel('s', color='red')
        ax_s.tick_params(axis='y', labelcolor='red')
        lines1, labels1 = axes[0, 0].get_legend_handles_labels()
        lines2, labels2 = ax_s.get_legend_handles_labels()
        axes[0, 0].legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Log-Determinant (ellipsoid volume)
    axes[0, 1].plot(time_axis, logdet_valid, color='green', linewidth=2)
    axes[0, 1].set_ylabel('log det(W)')
    axes[0, 1].set_title('Uncertainty Volume (log-determinant)')
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Spectral Radius (max eigenvalue)
    axes[1, 0].plot(time_axis, max_eig, label='Max eigenvalue', color='red', linewidth=2)
    axes[1, 0].plot(time_axis, min_eig, label='Min eigenvalue', color='blue', linewidth=1, alpha=0.7)
    axes[1, 0].fill_between(time_axis, min_eig, max_eig, color='red', alpha=0.08)
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_title('Spectral Bounds (worst/best-case direction)')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Condition Number
    axes[1, 1].semilogy(time_axis, cond_num, color='purple', linewidth=2)
    axes[1, 1].set_ylabel('Condition Number')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_title('Anisotropy (condition number)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path1 = os.path.join(figure_dir, f"{save_prefix}_W_scalar_summary.png")
        plt.savefig(path1, dpi=300, bbox_inches='tight')
        log.info(f"W scalar summary saved: {os.path.basename(path1)}")
    plt.close(fig1)

    # ================================================================
    # Figure 2: Per-Dimension Detail
    # ================================================================
    fig2 = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig2, height_ratios=[1, 1, 1.2])
    fig2.suptitle('Covariance W: Per-Dimension Detail', fontsize=15)

    # (a) Top-k eigenvalue evolution
    ax_eig = fig2.add_subplot(gs[0, :])
    top_k = min(8, D)
    for k_idx in range(top_k):
        dim_idx = D - 1 - k_idx
        ax_eig.plot(time_axis, eigenvals[:, dim_idx],
                    linewidth=1.5, label=f'eig #{k_idx+1}')
    ax_eig.set_ylabel('Eigenvalue')
    ax_eig.set_title(f'Top-{top_k} Eigenvalue Evolution')
    ax_eig.legend(fontsize=8, ncol=4, loc='upper right')
    ax_eig.grid(True, alpha=0.3)

    # (b) Diagonal variance heatmap
    ax_hm = fig2.add_subplot(gs[1, :])
    diag_elements = np.diagonal(W_history, axis1=1, axis2=2).T  # (D, T)
    im = ax_hm.imshow(diag_elements, aspect='auto', cmap='viridis',
                       extent=[0, time_axis[-1], D - 0.5, -0.5], interpolation='nearest')
    ax_hm.set_ylabel('Latent Dimension')
    ax_hm.set_xlabel('Time [s]')
    ax_hm.set_title('Diagonal Variance per Latent Dimension')
    plt.colorbar(im, ax=ax_hm, label='Variance', shrink=0.8)

    # (c) Matrix snapshots at key time points
    n_snapshots = min(4, T)
    snapshot_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    for col_idx, snap_idx in enumerate(snapshot_indices):
        if col_idx >= 2:
            break
        ax_snap = fig2.add_subplot(gs[2, col_idx])
        W_snap = W_history[snap_idx]
        # Normalize for display: show correlation matrix
        diag_std = np.sqrt(np.diag(W_snap))
        diag_std_safe = np.where(diag_std > 1e-12, diag_std, 1.0)
        corr = W_snap / np.outer(diag_std_safe, diag_std_safe)
        im_snap = ax_snap.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                                  interpolation='nearest')
        ax_snap.set_title(f't = {time_axis[snap_idx]:.1f}s (correlation)', fontsize=11)
        ax_snap.set_xlabel('Dimension')
        ax_snap.set_ylabel('Dimension')
        plt.colorbar(im_snap, ax=ax_snap, shrink=0.8)

    plt.tight_layout()
    if figure_dir:
        path2 = os.path.join(figure_dir, f"{save_prefix}_W_detail.png")
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        log.info(f"W detail plot saved: {os.path.basename(path2)}")
    plt.close(fig2)

    return trace_hist, max_eig


def plot_uncertainty_state_space(
    W_history: np.ndarray,
    model,
    z_history: np.ndarray,
    ctx_history: list,
    mpc_dt: float,
    figure_dir: str,
    save_prefix: str,
    inertia_change_time: Optional[float] = None,
):
    """
    Transform latent covariance W to state-space covariance Sigma_x via decoder Jacobian,
    then visualize the per-state uncertainty over time.

    For each time step k:
        J_k = d(decoder)/dz |_{z_k, ctx_k}   (nx x nz)
        Sigma_x_k = J_k @ W_k @ J_k^T         (nx x nx)

    Uses the exact context embedding from each MPC step for accurate Jacobian computation.

    Args:
        W_history:   (T, nz, nz) latent covariance history
        model:       Party model (or PartyInterface) with .decode() method
        z_history:   (T, nz) latent state trajectory (from MPC solve steps)
        ctx_history: list of T context tensors (each from the corresponding MPC step)
        mpc_dt:      MPC time step [s]
        figure_dir:  directory to save figures
        save_prefix: file name prefix
        inertia_change_time: optional time [s] of abrupt inertia change (for vertical line)
    """
    import torch

    T_w, nz, _ = W_history.shape
    T_z = z_history.shape[0]
    T_ctx = len(ctx_history)
    T = min(T_w, T_z, T_ctx)
    time_axis = np.arange(T) * mpc_dt

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'parameters'):
        device = next(model.parameters()).device
    else:
        device = torch.device('cpu')

    state_labels = ['q_w', 'q_x', 'q_y', 'q_z', 'w_x', 'w_y', 'w_z']
    nx = len(state_labels)

    # Compute state-space covariance at each step via Jacobian
    sigma_x_diag = np.zeros((T, nx))  # diagonal of Sigma_x (per-state variance)
    sigma_x_trace = np.zeros(T)

    model.eval()
    for k in range(T):
        ctx_k = ctx_history[k].to(device)
        z_k = torch.from_numpy(z_history[k]).float().to(device)

        # Closure with per-step ctx for Jacobian: z (nz,) -> x_normalized (nx,)
        # Apply quaternion normalization (same as get_latent_Q) to avoid inflated uncertainty
        def _decode_fn(z_vec: torch.Tensor) -> torch.Tensor:
            x_raw = model.decode(z_vec.unsqueeze(0)).squeeze(0)
            q = torch.nn.functional.normalize(x_raw[:4], p=2, dim=-1, eps=1e-8)
            return torch.cat([q, x_raw[4:]], dim=-1)

        # Compute Jacobian J = dx/dz  (nx x nz) via torch.func
        J = torch.func.jacrev(_decode_fn)(z_k)  # (nx, nz)

        J_np = J.detach().cpu().numpy()
        W_k = W_history[k]
        Sigma_x_k = J_np @ W_k @ J_np.T  # (nx, nx)

        sigma_x_diag[k] = np.diag(Sigma_x_k)
        sigma_x_trace[k] = np.trace(Sigma_x_k)

    # ================================================================
    # Figure: State-Space Uncertainty
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('State-Space Uncertainty (from Latent W)', fontsize=15)

    # (a) Per-state standard deviation
    ax_q = axes[0]
    q_colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        std_i = np.sqrt(np.maximum(sigma_x_diag[:, i], 0))
        ax_q.plot(time_axis, std_i, color=q_colors[i], linewidth=1.8, label=state_labels[i])
    ax_q.set_ylabel('Std. Dev.')
    ax_q.set_title('Quaternion Uncertainty (1-sigma)')
    ax_q.legend(fontsize=9, ncol=4)
    ax_q.grid(True, alpha=0.3)

    ax_w = axes[1]
    w_colors = ['red', 'blue', 'green']
    for i in range(3):
        std_i = np.sqrt(np.maximum(sigma_x_diag[:, 4 + i], 0))
        ax_w.plot(time_axis, std_i, color=w_colors[i], linewidth=1.8, label=state_labels[4 + i])
    ax_w.set_ylabel('Std. Dev. (rad/s)')
    ax_w.set_title('Angular Velocity Uncertainty (1-sigma)')
    ax_w.legend(fontsize=9, ncol=3)
    ax_w.grid(True, alpha=0.3)

    # (c) Total state-space uncertainty (trace)
    ax_tr = axes[2]
    ax_tr.plot(time_axis, sigma_x_trace, color='black', linewidth=2)
    ax_tr.set_ylabel('Trace(Sigma_x)')
    ax_tr.set_xlabel('Time [s]')
    ax_tr.set_title('Total State-Space Uncertainty')
    ax_tr.grid(True, alpha=0.3)

    # Mark inertia change time if provided
    if inertia_change_time is not None:
        for ax in axes:
            ax.axvline(inertia_change_time, color='gray', linestyle=':', linewidth=2,
                       alpha=0.7, label='Inertia change')
        axes[0].legend(fontsize=9, ncol=5)

    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, f"{save_prefix}_state_uncertainty.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f"State-space uncertainty plot saved: {os.path.basename(path)}")
    plt.close(fig)

    return sigma_x_diag


def plot_control_decomposition(
    Kz_history: np.ndarray,
    v_history: np.ndarray,
    dt: float,
    u_min: float,
    u_max: float,
    figure_dir: str,
    save_prefix: str,
):
    """Plot control decomposition: u = v_ref + K@(z - z_ref) at each ctrl step.

    Args:
        Kz_history: (N_steps, 3) K@(z-z_ref) feedback component per ctrl step
        v_history:  (N_steps, 3) v_ref nominal MPC component per ctrl step
        dt:         Time step between entries [s] (ctrl_dt)
        u_min:      control lower bound (scalar)
        u_max:      control upper bound (scalar)
        figure_dir: directory to save figures
        save_prefix: file name prefix
    """
    N = len(v_history)
    if N == 0:
        return
    time_axis = np.arange(N) * dt
    u_total = Kz_history + v_history
    axis_labels = ['x', 'y', 'z']
    colors_v = ['#2980b9', '#e74c3c', '#27ae60']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Control Output: u (ZOH)', fontsize=15)

    for i in range(3):
        ax = axes[i]
        ax.plot(time_axis, v_history[:, i],
                color=colors_v[i], linewidth=1.5, linestyle='--',
                drawstyle='steps-post', label=f'v_{axis_labels[i]}')
        ax.plot(time_axis, u_total[:, i],
                color='black', linewidth=2,
                drawstyle='steps-post', label=f'u_{axis_labels[i]}')
        ax.axhline(u_max, color='gray', linestyle=':', alpha=0.6)
        ax.axhline(u_min, color='gray', linestyle=':', alpha=0.6)
        ax.set_ylabel(f'Torque {axis_labels[i]} (Nm)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, f"{save_prefix}_control_decomp.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f"Control decomposition plot saved: {os.path.basename(path)}")
    plt.close(fig)



def plot_AB_diff_diagnostics(
    A_diff_history: np.ndarray,
    B_diff_history: np.ndarray,
    u_diff_history: np.ndarray,
    mpc_dt: float,
    figure_dir: str,
    save_prefix: str,
    B_sv_history: Optional[np.ndarray] = None,
    ctx_diff_history: Optional[np.ndarray] = None,
    A_rel_diff_history: Optional[np.ndarray] = None,
    A_norm_history: Optional[np.ndarray] = None,
):
    """Plot A/B change and control jump between consecutive MPC solves.

    Args:
        A_diff_history: (N_solve,) ||A_k - A_{k-1}||_F
        B_diff_history: (N_solve,) ||B_k - B_{k-1}||_F
        u_diff_history: (N_solve,) ||v_k[0] - v_{k-1}[0]||
        mpc_dt: MPC solve interval [s]
        figure_dir: directory to save figures
        save_prefix: file name prefix
        B_sv_history: (N_solve, 3) of (σ_min, σ_max, cond) — optional
        ctx_diff_history: (N_solve,) ||ctx_k - ctx_{k-1}|| — optional
        A_rel_diff_history: (N_solve,) ||ΔA||_F / ||A||_F — optional
        A_norm_history: (N_solve,) ||A||_F — optional
    """
    N = len(A_diff_history)
    if N == 0:
        return
    t = np.arange(N) * mpc_dt

    has_sv = B_sv_history is not None and len(B_sv_history) > 0
    has_rel = A_rel_diff_history is not None and len(A_rel_diff_history) == N
    n_rows = 3 + int(has_sv) + int(has_rel)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)
    fig.suptitle('Inter-Solve Variation Diagnostics', fontsize=15)

    row = 0

    # Row 0: Absolute A diff + ctx diff
    axes[row].plot(t, A_diff_history, 'r-', linewidth=1.0, label='||ΔA||_F')
    axes[row].set_ylabel('||ΔA||_F', color='r')
    if ctx_diff_history is not None and len(ctx_diff_history) == N:
        ax0r = axes[row].twinx()
        ax0r.plot(t, ctx_diff_history, 'b-', linewidth=1.0, alpha=0.7, label='||Δctx||')
        ax0r.set_ylabel('||Δctx||', color='b')
        ax0r.tick_params(axis='y', labelcolor='b')
    axes[row].set_title('A Matrix & Context Change Between Consecutive Solves')
    axes[row].grid(True, alpha=0.3)
    row += 1

    # Row 1 (optional): Relative A diff + A norm
    if has_rel:
        rel_pct = np.array(A_rel_diff_history) * 100
        axes[row].plot(t, rel_pct, 'r-', linewidth=1.0, label='||ΔA||_F / ||A||_F')
        axes[row].set_ylabel('Relative ΔA [%]', color='r')
        mean_rel = float(np.mean(rel_pct[1:])) if N > 1 else 0.0
        max_rel = float(np.max(rel_pct[1:])) if N > 1 else 0.0
        if A_norm_history is not None and len(A_norm_history) == N:
            ax_r = axes[row].twinx()
            ax_r.plot(t, A_norm_history, 'g--', linewidth=0.8, alpha=0.6, label='||A||_F')
            ax_r.set_ylabel('||A||_F', color='g')
            ax_r.tick_params(axis='y', labelcolor='g')
        axes[row].set_title(f'Relative A Change  (mean={mean_rel:.2f}%, max={max_rel:.2f}%)')
        axes[row].grid(True, alpha=0.3)
        row += 1

    # B diff
    axes[row].plot(t, B_diff_history, 'b-', linewidth=1.0)
    axes[row].set_ylabel('||ΔB||_F')
    axes[row].set_title('B Matrix Change Between Consecutive Solves')
    axes[row].grid(True, alpha=0.3)
    row += 1

    # Control jump
    axes[row].plot(t, u_diff_history, 'k-', linewidth=1.0)
    axes[row].set_ylabel('||Δv||')
    axes[row].set_title('First-Step Control Jump Between Consecutive Solves')
    axes[row].grid(True, alpha=0.3)
    row += 1

    # B singular values
    if has_sv:
        t_sv = np.arange(len(B_sv_history)) * mpc_dt
        sv = np.asarray(B_sv_history)
        axes[row].semilogy(t_sv, sv[:, 0], 'g-', linewidth=1.0, label='σ_min(B)')
        axes[row].semilogy(t_sv, sv[:, 1], 'm-', linewidth=1.0, label='σ_max(B)')
        axes[row].set_ylabel('Singular Value')
        axes[row].set_title(f'B Singular Values  (cond mean={sv[:, 2].mean():.1f}, max={sv[:, 2].max():.1f})')
        axes[row].legend(loc='upper right')
        axes[row].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, f"{save_prefix}_AB_diff.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f"A/B diff plot saved: {os.path.basename(path)}")
    plt.close(fig)


def plot_v_plan_diagnostics(
    v_plan_history: list,
    mpc_dt: float,
    figure_dir: str,
    save_prefix: str,
):
    """Analyze planned v trajectories: intra-plan oscillation & inter-solve shift.

    Subplot 1: Per-axis v trajectory from a few representative solves (shows if
               the QP itself plans an oscillating sequence).
    Subplot 2: Inter-solve plan shift ||v_plan_curr[0:N-1] - v_plan_prev[1:N]||
               per step (shows how much the plan changes between consecutive solves).
    """
    N_solve = len(v_plan_history)
    if N_solve < 2:
        return
    nu = v_plan_history[0].shape[0]
    N = v_plan_history[0].shape[1]

    # --- Subplot 1: intra-plan oscillation for selected solves ---
    # Pick enough solves so consecutive plans overlap by ~half the horizon
    n_picks = max(6, 2 * N_solve // max(1, N) + 1)
    pick_idx = np.linspace(0, N_solve - 1, min(n_picks, N_solve), dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Planned v Trajectory Diagnostics', fontsize=15)

    ax0 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(pick_idx)))
    for ci, si in enumerate(pick_idx):
        v_plan = v_plan_history[si]  # (nu, N)
        t_plan = (si + np.arange(N)) * mpc_dt
        for j in range(nu):
            ls = ['-', '--', ':'][j % 3]
            ax0.plot(t_plan, v_plan[j, :], ls, color=colors[ci], linewidth=0.8,
                     alpha=0.8, label=f'solve {si}, ax{j}' if ci == 0 else None)
    ax0.set_ylabel('v')
    ax0.set_title('Planned v Sequence (selected solves, per-axis)')
    ax0.grid(True, alpha=0.3)
    # Compact legend: only first solve labels
    handles, labels = ax0.get_legend_handles_labels()
    if handles:
        ax0.legend(handles[:nu], labels[:nu], fontsize=8, loc='upper right')

    # --- Subplot 2: inter-solve shift ---
    shift_norms = []
    for i in range(1, N_solve):
        v_prev = v_plan_history[i - 1]  # (nu, N)
        v_curr = v_plan_history[i]      # (nu, N)
        # Compare shifted: prev[:,1:] vs curr[:,:-1]
        common = min(v_prev.shape[1] - 1, v_curr.shape[1] - 1)
        if common > 0:
            diff = v_curr[:, :common] - v_prev[:, 1:common + 1]
            shift_norms.append(float(np.linalg.norm(diff, 'fro') / np.sqrt(common)))
        else:
            shift_norms.append(0.0)

    t_shift = np.arange(1, N_solve) * mpc_dt
    ax1 = axes[1]
    ax1.plot(t_shift, shift_norms, 'k-', linewidth=1.0)
    ax1.set_ylabel('RMS ||Δv_plan||')
    ax1.set_title('Inter-Solve Plan Shift: ||v_curr[0:N-1] - v_prev[1:N]||_rms')
    ax1.set_xlabel('Time [s]')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, f"{save_prefix}_v_plan_diag.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f"v plan diagnostics plot saved: {os.path.basename(path)}")
    plt.close(fig)


def plot_latent_diagnostics(
    z_history: np.ndarray,
    z_goal_history: np.ndarray,
    model,
    ctx_history: list,
    mpc_dt: float,
    figure_dir: str,
    save_prefix: str,
    z_cost_Q_history: Optional[np.ndarray] = None,
):
    """Visualize latent-space and decoded state-space diagnostics.

    Produces two figures:
      1. Latent error: top-k dimensions of z_curr vs z_goal, plus ||z_err|| scalar.
      2. Decoded state error: per-component x_curr vs x_goal from decoding z.

    Args:
        z_history:      (N, nz) latent states at each MPC solve
        z_goal_history: (N, nz) latent goals at each MPC solve
        model:          Party model with .decode()
        ctx_history:    list of N context tensors
        mpc_dt:         MPC time step [s]
        figure_dir:     directory to save figures
        save_prefix:    file name prefix
    """
    import torch

    N = min(len(z_history), len(z_goal_history), len(ctx_history))
    if N == 0:
        return
    nz = z_history.shape[1]
    time_axis = np.arange(N) * mpc_dt

    z_err = z_history[:N] - z_goal_history[:N]  # (N, nz)
    z_err_norm = np.linalg.norm(z_err, axis=1)  # (N,)

    # Top-k dimensions by max absolute error
    max_abs_err = np.max(np.abs(z_err), axis=0)  # (nz,)
    top_k = min(8, nz)
    top_dims = np.argsort(max_abs_err)[-top_k:][::-1]

    # ---- Figure 1: Latent Space ----
    fig1, axes1 = plt.subplots(top_k + 1, 1, figsize=(14, 2.5 * (top_k + 1)), sharex=True)
    fig1.suptitle('Latent Space: z_curr vs z_goal (top error dims)', fontsize=14)

    for i, dim in enumerate(top_dims):
        ax = axes1[i]
        ax.plot(time_axis, z_history[:N, dim], 'b-', linewidth=1.5, label='z_curr')
        ax.plot(time_axis, z_goal_history[:N, dim], 'r--', linewidth=1.5, label='z_goal')
        ax.fill_between(time_axis, z_history[:N, dim], z_goal_history[:N, dim],
                         alpha=0.15, color='gray')
        ax.set_ylabel(f'dim {dim}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, ncol=2)

    ax_norm = axes1[-1]
    if z_cost_Q_history is not None and len(z_cost_Q_history) >= N:
        ax_norm.plot(time_axis, z_cost_Q_history[:N], 'k-', linewidth=2)
        ax_norm.set_ylabel(r'$\| z_{err} \|_{Q}^2$')
        ax_norm.set_yscale('log')
    else:
        ax_norm.plot(time_axis, z_err_norm, 'k-', linewidth=2)
        ax_norm.set_ylabel('||z_err||')
    ax_norm.set_xlabel('Time [s]')
    ax_norm.grid(True, alpha=0.3)

    plt.tight_layout()
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path1 = os.path.join(figure_dir, f"{save_prefix}_latent_diag.png")
        plt.savefig(path1, dpi=300, bbox_inches='tight')
        log.info(f"Latent diagnostics plot saved: {os.path.basename(path1)}")
    plt.close(fig1)

    # ---- Figure 2: Decoded State Space ----
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    state_labels = ['q_w', 'q_x', 'q_y', 'q_z', 'w_x', 'w_y', 'w_z']
    nx = len(state_labels)
    x_decoded = np.zeros((N, nx))
    x_goal_decoded = np.zeros((N, nx))

    model.eval()
    with torch.no_grad():
        for k in range(N):
            ctx_k = ctx_history[k].to(device)
            z_k = torch.from_numpy(z_history[k]).float().to(device)
            z_g = torch.from_numpy(z_goal_history[k]).float().to(device)
            x_decoded[k] = model.decode(z_k.unsqueeze(0)).squeeze(0).cpu().numpy()[:nx]
            x_goal_decoded[k] = model.decode(z_g.unsqueeze(0)).squeeze(0).cpu().numpy()[:nx]

    # Unwrap decoded quaternion trajectories and sign-align goal to final current
    x_decoded[:, :4] = unwrap_quaternions(x_decoded[:, :4])
    x_goal_decoded[:, :4] = unwrap_quaternions(x_goal_decoded[:, :4])
    q_curr_final = x_decoded[-1, :4]
    q_goal_final = x_goal_decoded[-1, :4]
    if np.dot(q_curr_final, q_goal_final) < 0:
        x_goal_decoded[:, :4] = -x_goal_decoded[:, :4]

    fig2, axes2 = plt.subplots(nx, 1, figsize=(14, 2.5 * nx), sharex=True)
    fig2.suptitle('Decoded State: decode(z_curr) vs decode(z_goal)', fontsize=14)

    colors = ['red', 'blue', 'green', 'orange', 'red', 'blue', 'green']
    for i in range(nx):
        ax = axes2[i]
        ax.plot(time_axis, x_decoded[:, i], '-', color=colors[i], linewidth=1.5, label='decoded curr')
        ax.plot(time_axis, x_goal_decoded[:, i], '--', color=colors[i], linewidth=1.5, alpha=0.7, label='decoded goal')
        ax.set_ylabel(state_labels[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, ncol=2)

    axes2[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    if figure_dir:
        path2 = os.path.join(figure_dir, f"{save_prefix}_decoded_state_diag.png")
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        log.info(f"Decoded state diagnostics plot saved: {os.path.basename(path2)}")
    plt.close(fig2)


def plot_mc_koz_separation(
    all_koz_traces: Dict[str, list],
    koz_object: List[KeepoutZone],
    log_dt: float,
    figure_dir: str,
    save_prefix: str,
    title_suffix: str = "",
):
    """Plot MC overlay of per-KOZ min-across-boresights separation angles.

    Each episode contributes one line per KOZ (the minimum separation angle
    across all boresight sensors at each timestep).

    Args:
        all_koz_traces: Dict mapping KOZ name to list of (T_i,) arrays.
        koz_object: List of KeepoutZone instances (for limit angles).
        log_dt: Logging time step [s].
        figure_dir: Directory to save the figure.
        save_prefix: File name prefix.
        title_suffix: Optional suffix for the figure title.
    """
    num_koz = len(koz_object)
    fig, axes = plt.subplots(num_koz, 1, figsize=(14, 5 * num_koz), squeeze=False)
    fig.suptitle(f'MC KOZ Separation (min across boresights){title_suffix}', fontsize=16)

    koz_colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']

    for k_idx, koz in enumerate(koz_object):
        ax = axes[k_idx, 0]
        traces = all_koz_traces.get(koz.name, [])
        color = koz_colors[k_idx % len(koz_colors)]

        # Individual episode traces
        for trace in traces:
            time_vec = np.arange(len(trace)) * log_dt
            ax.plot(time_vec, trace, color=color, alpha=0.15, linewidth=0.6)

        # Median + percentile bands (if enough episodes)
        if len(traces) >= 3:
            max_len = max(len(tr) for tr in traces)
            # Pad shorter traces with NaN for stacking
            padded = np.full((len(traces), max_len), np.nan)
            for i, tr in enumerate(traces):
                padded[i, :len(tr)] = tr
            time_vec_full = np.arange(max_len) * log_dt
            median = np.nanmedian(padded, axis=0)
            p5 = np.nanpercentile(padded, 5, axis=0)
            p95 = np.nanpercentile(padded, 95, axis=0)
            ax.fill_between(time_vec_full, p5, p95, color=color, alpha=0.15,
                            label='5-95 percentile')
            ax.plot(time_vec_full, median, color=color, linewidth=2.0,
                    alpha=0.9, label='Median')

        # KOZ exclusion limit line
        ax.axhline(koz.angle_deg, color='black', linestyle='--', linewidth=2.0,
                    label=f'{koz.name} limit ({koz.angle_deg}\u00b0)')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Min Separation Angle (deg)')
        ax.set_title(f'{koz.name} -- {len(traces)} episodes', fontsize=13)
        ax.set_ylim([0, 180])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'{save_prefix}_mc_koz_separation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    log.info(f"MC KOZ separation plot saved: {os.path.basename(path)}")
    plt.close(fig)


def plot_mc_boresight_3d(
    boresight_trajs: Dict[str, list],
    koz_objects: List[KeepoutZone],
    q_goal: np.ndarray,
    figure_dir: str,
    save_prefix: str,
    title_suffix: str = "",
    max_traces: int = 50,
    elev: float = 20,
    azim: float = 45,
):
    """Plot MC 3D boresight trajectories on the unit sphere with KOZ cones.

    Visual style matches ``plot_sim_results`` (single-run 3D boresight plot).

    Args:
        boresight_trajs: Dict mapping boresight name to list of (T_i, 3) arrays
                         (inertial-frame unit vectors per episode).
        koz_objects: List of KeepoutZone instances.
        q_goal: (4,) goal quaternion [w, x, y, z].
        figure_dir: Directory to save figures.
        save_prefix: File name prefix.
        title_suffix: Optional suffix for figure title.
        max_traces: Maximum number of episode traces to draw (for readability).
    """
    bs_names = list(boresight_trajs.keys())
    if not bs_names:
        return

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- Wireframe unit sphere (same mesh as plot_sim_results) ---
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z,
                      alpha=0.28, color='gray', linewidth=1.0)
    ax.set_box_aspect([1, 1, 1])

    # --- KOZ cones (same construction as plot_sim_results) ---
    koz_colors = ['yellow', 'lightcoral', 'orange', 'lightgreen']
    for i, koz in enumerate(koz_objects):
        color = koz_colors[i % len(koz_colors)]
        direction = np.asarray(koz.direction, dtype=float)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            continue
        direction = direction / norm_dir

        angle_rad = np.deg2rad(koz.angle_deg)
        cone_height = np.cos(angle_rad)
        cone_radius = np.sin(angle_rad)

        # Perpendicular basis
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Spherical cap surface
        theta_cone = np.linspace(0, 2 * np.pi, 64)
        phi_cap = np.linspace(0, angle_rad, 32)
        theta_grid, phi_grid = np.meshgrid(theta_cone, phi_cap)
        cap_x = (np.cos(phi_grid) * direction[0]
                 + np.sin(phi_grid) * (np.cos(theta_grid) * perp1[0]
                                       + np.sin(theta_grid) * perp2[0]))
        cap_y = (np.cos(phi_grid) * direction[1]
                 + np.sin(phi_grid) * (np.cos(theta_grid) * perp1[1]
                                       + np.sin(theta_grid) * perp2[1]))
        cap_z = (np.cos(phi_grid) * direction[2]
                 + np.sin(phi_grid) * (np.cos(theta_grid) * perp1[2]
                                       + np.sin(theta_grid) * perp2[2]))
        ax.plot_surface(cap_x, cap_y, cap_z, color=color, alpha=0.22,
                        linewidth=0, shade=False, zorder=3)

        # Boundary circle
        cone_points = []
        for t in theta_cone:
            boundary_dir = (cone_height * direction
                            + cone_radius * (np.cos(t) * perp1
                                             + np.sin(t) * perp2))
            boundary_dir = boundary_dir / np.linalg.norm(boundary_dir)
            cone_points.append(boundary_dir)
        cone_points = np.array(cone_points)
        ax.plot(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2],
                color=color, alpha=0.95, linewidth=2.5,
                label=f'{koz.name} ({koz.angle_deg}\u00b0)')

        # Direction marker + dashed line from origin
        ax.scatter(*direction, color=color, s=80, marker='^', alpha=0.9,
                   zorder=8)
        ax.plot([0.0, direction[0]], [0.0, direction[1]], [0.0, direction[2]],
                color=color, linestyle='--', linewidth=1.2, alpha=0.8)

        # Text label
        label_pos = direction * 1.3
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f'{koz.name}\n{koz.angle_deg}\u00b0', fontsize=10, ha='center')

    # --- Boresight goal targets ---
    q_goal_norm = np.asarray(q_goal, dtype=float)
    q_goal_norm = q_goal_norm / (np.linalg.norm(q_goal_norm) + 1e-12)

    # --- Boresight trajectories ---
    traj_colors = ['blue', 'orange', 'green', 'purple', 'red', 'gray']
    traj_markers = ['D', 's', 'p', 'h', '8', '*']
    for b_idx, bs_name in enumerate(bs_names):
        traces = boresight_trajs[bs_name]
        color = traj_colors[b_idx % len(traj_colors)]
        marker = traj_markers[b_idx % len(traj_markers)]
        show_common = (b_idx == 0)

        # Compute goal boresight for this sensor
        bs_goal = quat2boresight_batch(q_goal_norm.reshape(1, -1),
                                       traces[0][0]  # dummy; overridden below
                                       )[0] if False else None
        # We cannot recover the body-frame vector from inertial traces alone,
        # so we skip the goal marker if q_goal is not meaningful for MC overlay.

        draw_traces = traces[:max_traces]
        for i, traj in enumerate(draw_traces):
            traj = np.asarray(traj, dtype=float)
            norms = np.linalg.norm(traj, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            traj = traj / norms

            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=color, linewidth=1.0, alpha=0.25)

            if i == 0:
                # Start marker
                ax.scatter(*traj[0], color=color, s=70, marker='o',
                           label='Start' if show_common else '_nolegend_',
                           zorder=10)
                # Final marker
                ax.scatter(*traj[-1], color=color, s=100, marker=marker,
                           label='Final' if show_common else '_nolegend_',
                           zorder=10)

        # Legend entry for this boresight (one dummy line)
        ax.plot([], [], [], color=color, linewidth=3.0,
                label=f'{bs_name} ({len(draw_traces)} ep)')

    # --- Axis styling (same as plot_sim_results) ---
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_xlabel('X (Inertial)')
    ax.set_ylabel('Y (Inertial)')
    ax.set_zlabel('Z (Inertial)')
    ax.set_title(f'MC 3D Boresight Trajectories with Keep-Out Zones{title_suffix}',
                 fontsize=14)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = {}
    for h, l in zip(handles, labels):
        if l and l != '_nolegend_' and l not in unique_legend:
            unique_legend[l] = h
    ax.legend(
        list(unique_legend.values()),
        list(unique_legend.keys()),
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
    )

    # Transparent panes, no grid
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'{save_prefix}_mc_boresight_3d.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    log.info(f"MC boresight 3D plot saved: {os.path.basename(path)}")
    plt.close(fig)
