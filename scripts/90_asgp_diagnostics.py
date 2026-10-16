"""ASGP diagnostics — offline validation, tighten check, residual tracking.

Subcommands
-----------
offline   Offline GP validation (RMSE, calibration, NLPD, scatter) + online
          adaptation test.  Loads the learning_asgp training config.
tighten   Monkey-patches _build_local_models to check whether max_tighten
          clips real uncertainty signal during a live episode.
residual  Runs episodes and plots actual residual vs GP mean ± 2σ.

Usage::

    python scripts/90_asgp_diagnostics.py offline --exp_name exp1a
    python scripts/90_asgp_diagnostics.py offline --exp_name exp1b --sc exp1b
    python scripts/90_asgp_diagnostics.py tighten [--exp_name exp1b] [--gp_mode full]
    python scripts/90_asgp_diagnostics.py residual [--exps exp1b exp2] [--modes fast full]
"""

import argparse
import logging
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM_NAMES = ["dω₁", "dω₂", "dω₃"]


# ---------------------------------------------------------------------------
#  Shared: suppress verbose sub-loggers used by online/tighten/residual modes
# ---------------------------------------------------------------------------

def _silence_sim_loggers():
    for _name in [
        "sim.koz.koz_utils", "sim.utils.bsk_setup",
        "sim.montecarlo.single_episode",
        "src.utils.party_utils", "src.model.party_interface",
        "src.dataset.processor",
        "PartySMPC", "NMPC", "ASGP_SMPC", "L_SMPC", "UKF_SMPC",
    ]:
        logging.getLogger(_name).setLevel(logging.CRITICAL)

    from Basilisk.architecture import bskLogging
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


# ---------------------------------------------------------------------------
#  Shared: build episode config (used by tighten and residual)
# ---------------------------------------------------------------------------

def build_episode_cfg(exp_name: str, gp_mode: str = "full"):
    """Assemble a full OmegaConf episode config for the given exp/mode."""
    sc_default = OmegaConf.load(f"{REPO}/configs/sc/sc_default.yaml")
    sc_default.pop("defaults", None)
    sc_exp = OmegaConf.load(f"{REPO}/configs/sc/{exp_name}.yaml")
    sc_exp.pop("defaults", None)
    sc = OmegaConf.merge(sc_default, sc_exp)

    cst_default = OmegaConf.load(f"{REPO}/configs/constraint/default.yaml")
    cst_default.pop("defaults", None)
    cst_name = "exp1b" if exp_name == "exp2b" else exp_name
    cst_exp = OmegaConf.load(f"{REPO}/configs/constraint/{cst_name}.yaml")
    cst_exp.pop("defaults", None)
    constraint = OmegaConf.merge(cst_default, cst_exp)

    ctrl_cfg = OmegaConf.load(f"{REPO}/configs/controller/asgp_smpc.yaml")
    ctrl_cfg.pop("defaults", None)
    overrides = OmegaConf.create({
        "exp_name": exp_name,
        "gp": {
            "mode": gp_mode,
            "hp_update_period": 1,
            "hp_num_steps": 1,
            "hp_lr": 0.05,
        },
        "cc": {"alpha": 0.95},
    })
    ctrl_cfg = OmegaConf.merge(ctrl_cfg, overrides)

    ctrl_dt = 0.5
    OmegaConf.update(ctrl_cfg, "ctrl_dt", ctrl_dt)
    OmegaConf.update(ctrl_cfg, "mpc_dt", ctrl_dt)
    OmegaConf.update(ctrl_cfg, "inertia", OmegaConf.to_container(sc.inertia))
    OmegaConf.update(ctrl_cfg, "model_root",
                     f"{REPO}/outputs/learning_asgp/sparse_gp")

    return OmegaConf.create({
        "sim": {
            "sim_time": 50.0, "sim_dt": 0.01,
            "log_dt": 0.01, "ctrl_dt": ctrl_dt,
        },
        "env": {
            "gravity_order": 4, "enable_drag": True,
            "enable_srp": True, "enable_gg": True,
        },
        "sc": OmegaConf.to_container(sc),
        "constraint": OmegaConf.to_container(constraint),
        "controller": OmegaConf.to_container(ctrl_cfg),
    })


# ---------------------------------------------------------------------------
#  Shared: load learning_asgp training config (used by offline mode)
# ---------------------------------------------------------------------------

def load_training_cfg(exp_name: str, sc_name: str | None = None):
    """Load the learning_asgp config and resolve interpolations manually."""
    cfg = OmegaConf.load(f"{REPO}/configs/learning_asgp.yaml")
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    cfg.pop("wandb", None)

    sc_key = sc_name or exp_name
    sc_default = OmegaConf.load(f"{REPO}/configs/sc/sc_default.yaml")
    sc_default.pop("defaults", None)
    sc_exp = OmegaConf.load(f"{REPO}/configs/sc/{sc_key}.yaml")
    sc_exp.pop("defaults", None)
    sc = OmegaConf.merge(sc_default, sc_exp)

    # Resolve interpolations that Hydra would handle
    OmegaConf.update(cfg, "exp_name", exp_name)
    OmegaConf.update(cfg, "data.name", f"calib_{exp_name}")
    OmegaConf.update(cfg, "asgp.inertia", OmegaConf.to_container(sc.inertia))

    return cfg


# ===========================================================================
#  MODE: offline
# ===========================================================================

def offline_validation(
    gp_model,
    z_raw: np.ndarray,
    y_raw: np.ndarray,
    max_val: int,
    seed: int,
    figure_dir: str,
):
    """Predict on held-out residuals and report RMSE/calibration/NLPD + plots."""
    n = min(max_val, len(z_raw))
    rng = np.random.RandomState(seed + 999)
    idx = rng.choice(len(z_raw), size=n, replace=False)
    z_val = z_raw[idx]
    y_val = y_raw[idx]

    out_dim = gp_model.output_dim
    mu_all = np.zeros((n, out_dim), dtype=np.float64)
    var_all = np.zeros((n, out_dim), dtype=np.float64)

    log.info(f"Running offline prediction on {n} samples...")
    for i in range(n):
        mu, var = gp_model.predict_np(z_val[i])
        mu_all[i] = mu
        var_all[i] = var

    std_all = np.sqrt(var_all)
    err = y_val - mu_all
    rmse = np.sqrt(np.mean(err ** 2, axis=0))

    z_score = np.abs(err) / np.maximum(std_all, 1e-15)
    calib = {k: np.mean(z_score < k, axis=0) for k in [1, 2, 3]}

    nlpd = 0.5 * np.mean(
        np.log(2 * np.pi * var_all) + err ** 2 / np.maximum(var_all, 1e-15),
        axis=0,
    )

    log.info("")
    log.info("=" * 80)
    log.info("  Offline Validation Results")
    log.info("=" * 80)
    log.info(f"  {'Dim':>6s}  {'RMSE':>10s}  {'mean(|μ|)':>10s}  {'mean(σ)':>10s}  "
             f"{'±1σ(%)':>8s}  {'±2σ(%)':>8s}  {'±3σ(%)':>8s}  {'NLPD':>8s}")
    log.info("-" * 80)
    for j in range(out_dim):
        log.info(
            f"  {DIM_NAMES[j]:>6s}  {rmse[j]:10.2e}  "
            f"{np.mean(np.abs(mu_all[:, j])):10.2e}  "
            f"{np.mean(std_all[:, j]):10.2e}  "
            f"{calib[1][j] * 100:7.1f}%  {calib[2][j] * 100:7.1f}%  "
            f"{calib[3][j] * 100:7.1f}%  {nlpd[j]:8.3f}"
        )
    log.info("-" * 80)
    log.info("  Expected (Gaussian):          68.3%    95.4%    99.7%")
    log.info("=" * 80)

    import matplotlib.pyplot as plt
    os.makedirs(figure_dir, exist_ok=True)

    # Scatter plot
    ncols = min(out_dim, 3)
    nrows = (out_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = [axes] if out_dim == 1 else axes.flatten()
    for j in range(out_dim):
        ax = axes[j]
        ax.errorbar(
            y_val[:, j], mu_all[:, j],
            yerr=2 * std_all[:, j],
            fmt=".", markersize=1.5, elinewidth=0.3, alpha=0.3, color="C0",
        )
        lims = [
            min(y_val[:, j].min(), mu_all[:, j].min()),
            max(y_val[:, j].max(), mu_all[:, j].max()),
        ]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Actual residual")
        ax.set_ylabel("GP predicted mean")
        ax.set_title(f"{DIM_NAMES[j]}  RMSE={rmse[j]:.2e}")
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    fig.suptitle("Offline Validation: GP Mean vs Actual Residual (±2σ)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(figure_dir, "asgp_offline_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Scatter plot saved to {path}")

    # Calibration bar chart
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 2 * out_dim), 5))
    x_pos = np.arange(out_dim)
    width = 0.25
    for ki, k in enumerate([1, 2, 3]):
        expected = {1: 68.3, 2: 95.4, 3: 99.7}[k]
        ax.bar(x_pos + ki * width, calib[k] * 100, width,
               label=f"±{k}σ (exp {expected:.1f}%)")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(DIM_NAMES[:out_dim])
    ax.set_ylabel("% samples within band")
    ax.set_title("Variance Calibration")
    ax.legend()
    ax.axhline(68.3, color="C0", ls=":", lw=0.7)
    ax.axhline(95.4, color="C1", ls=":", lw=0.7)
    ax.axhline(99.7, color="C2", ls=":", lw=0.7)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = os.path.join(figure_dir, "asgp_calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Calibration plot saved to {path}")

    return rmse, calib, nlpd


def online_adaptation_test(
    model_file: str,
    dataset,
    pred_stride: int,
    pred_dt: float,
    nominal_inertia: np.ndarray,
    nominal_inertia_inv: np.ndarray,
    gp_cfg,
    figure_dir: str,
    num_seqs: int = 3,
    seed: int = 42,
):
    """Feed residuals from mismatched-inertia sequences and watch GP adaptation."""
    from src.model.adaptive_sgp import MultiOutputASGP
    from src.utils.math_numpy import attitude_dyn_dt_batch

    log.info("")
    log.info("=" * 80)
    log.info("  Online Adaptation Test")
    log.info("=" * 80)

    has_inertia = dataset.i_traj_list is not None and len(dataset.i_traj_list) > 0
    if not has_inertia:
        log.warning("No per-sequence inertia data — skipping online adaptation test.")
        return

    import matplotlib.pyplot as plt

    test_indices = list(dataset.split_indices.get("test", dataset.split_indices["train"]))
    rng = np.random.RandomState(seed + 777)
    rng.shuffle(test_indices)
    test_indices = test_indices[:num_seqs]

    os.makedirs(figure_dir, exist_ok=True)

    for seq_idx in test_indices:
        gp_model = MultiOutputASGP.from_pretrained(
            model_file=model_file,
            window_size=int(gp_cfg.window_size),
            num_inducing=int(gp_cfg.num_inducing),
            perc_th=float(gp_cfg.perc_th),
            lambda_forget=float(gp_cfg.lambda_forget),
        )
        out_dim = gp_model.output_dim

        x_traj = dataset.x_traj_list[seq_idx].cpu().numpy().astype(np.float64)
        u_traj = dataset.u_traj_list[seq_idx].cpu().numpy().astype(np.float64)
        max_t = min(len(u_traj), len(x_traj) - pred_stride)
        if max_t <= 0:
            continue

        from src.train.train_asgp import _jvec_to_mat
        j_vec = dataset.i_traj_list[seq_idx][0].cpu().numpy().astype(np.float64)
        j_actual = _jvec_to_mat(j_vec)
        j_diff = np.linalg.norm(j_actual - nominal_inertia, "fro")
        log.info(f"  Seq {seq_idx}: ||J_actual - J_nom||_F = {j_diff:.4f}")

        t_steps = min(max_t, 200)
        actual_resid = np.zeros((t_steps, out_dim))
        gp_mu = np.zeros((t_steps, out_dim))
        gp_std = np.zeros((t_steps, out_dim))

        for t in range(t_steps):
            x_t = x_traj[t]
            u_t = u_traj[t]
            x_next_actual = x_traj[t + pred_stride]

            x_nom = attitude_dyn_dt_batch(
                x_t.reshape(1, -1), u_t.reshape(1, -1),
                pred_dt, nominal_inertia, nominal_inertia_inv,
            )[0]

            resid = x_next_actual[4:] - x_nom[4:]
            z_raw = np.concatenate([x_t[4:], u_t])
            actual_resid[t] = resid

            mu, var = gp_model.predict_np(z_raw)
            gp_mu[t] = mu
            gp_std[t] = np.sqrt(var)
            gp_model.update(z_raw, resid)

        time_axis = np.arange(t_steps) * pred_dt
        ncols = min(out_dim, 3)
        nrows = (out_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes = [axes] if out_dim == 1 else axes.flatten()
        for j in range(out_dim):
            ax = axes[j]
            ax.plot(time_axis, actual_resid[:, j], "k-", lw=0.8, label="Actual", alpha=0.8)
            ax.plot(time_axis, gp_mu[:, j], "C0-", lw=0.8, label="GP mean")
            ax.fill_between(
                time_axis,
                gp_mu[:, j] - 2 * gp_std[:, j],
                gp_mu[:, j] + 2 * gp_std[:, j],
                alpha=0.2, color="C0", label="±2σ",
            )
            ax.set_xlabel("Time [s]")
            ax.set_title(DIM_NAMES[j])
            ax.ticklabel_format(style="sci", scilimits=(-2, 2))
            if j == 0:
                ax.legend(fontsize=8)
        fig.suptitle(
            f"Online Adaptation — Seq {seq_idx}  (||ΔJ||={j_diff:.3f})",
            fontsize=13,
        )
        fig.tight_layout()
        path = os.path.join(figure_dir, f"asgp_online_seq{seq_idx}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"  Online adaptation plot saved to {path}")

    log.info("=" * 80)


def run_offline(args):
    from src.dataset import PartyDataset
    from src.model.adaptive_sgp import MultiOutputASGP
    from src.train.train_asgp import collect_mismatch_data_omega
    from src.utils import set_seed

    cfg = load_training_cfg(args.exp_name, args.sc)
    set_seed(cfg.data.seed)

    model_file = os.path.join(
        REPO, "outputs", "learning_asgp", "sparse_gp",
        f"model_{args.exp_name}.npz",
    )
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")
    log.info(f"Model: {model_file}")

    gp_cfg = OmegaConf.create({
        "window_size": 100,
        "num_inducing": 30,
        "perc_th": 0.01,
        "lambda_forget": float(cfg.asgp.lambda_forget),
    })

    gp_model = MultiOutputASGP.from_pretrained(
        model_file=model_file,
        window_size=int(gp_cfg.window_size),
        num_inducing=int(gp_cfg.num_inducing),
        perc_th=float(gp_cfg.perc_th),
        lambda_forget=float(gp_cfg.lambda_forget),
    )

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_cfg, dict):
        raise TypeError("cfg.data must resolve to a dict")
    data_cfg.pop("_target_", None)
    dataset = PartyDataset(**data_cfg, split="train", use_norm=False)

    pred_stride = int(round(float(cfg.data.pred_dt) / float(cfg.data.log_dt)))
    inertia = np.array(cfg.asgp.inertia, dtype=np.float64).reshape(3, 3)
    inertia_inv = np.linalg.inv(inertia)
    rk4_steps = int(cfg.asgp.rk4_steps)

    log.info("Collecting residuals for offline validation...")
    z_per_seq, y_per_seq = collect_mismatch_data_omega(
        dataset, pred_stride, float(cfg.data.pred_dt), inertia, inertia_inv,
        rk4_steps=rk4_steps,
    )
    z_raw = np.concatenate(z_per_seq, axis=0)
    y_raw = np.concatenate(y_per_seq, axis=0)
    log.info(f"Total samples: {len(z_raw)}")

    figure_dir = os.path.join(
        REPO, "outputs", "learning_asgp", "diagnostics", args.exp_name,
    )

    offline_validation(
        gp_model, z_raw, y_raw,
        max_val=2000, seed=int(cfg.data.seed),
        figure_dir=figure_dir,
    )

    online_adaptation_test(
        model_file=model_file,
        dataset=dataset,
        pred_stride=pred_stride,
        pred_dt=float(cfg.data.pred_dt),
        nominal_inertia=inertia,
        nominal_inertia_inv=inertia_inv,
        gp_cfg=gp_cfg,
        figure_dir=figure_dir,
        num_seqs=3,
        seed=int(cfg.data.seed),
    )


# ===========================================================================
#  MODE: tighten
# ===========================================================================

# Diagnostic accumulator (module-level so the monkey-patch can write to it)
_DIAG = {
    "var_res_all": [],
    "tighten_raw_all": [],
    "tighten_all": [],
    "binding_all": [],
    "call_count": 0,
}


def _make_patched_build_local_models():
    """Return a replacement for ASGPSMPC._build_local_models that logs diagnostics."""

    def _patched(self, x_curr):
        from src.utils.math_numpy import attitude_dyn_dt_jac

        if self._prev_u_sol is not None:
            u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)
            u_nom[:-1] = self._prev_u_sol[:, 1:].T
            u_nom[-1] = self._prev_u_sol[:, -1]
        else:
            u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)

        x_nom = np.zeros((self.horizon + 1, self.nx), dtype=np.float64)
        x_nom[0] = x_curr.copy()
        A_seq = np.zeros((self.horizon, self.nx, self.nx), dtype=np.float64)
        q_proc_seq = np.zeros((self.horizon, self.nx, self.nx), dtype=np.float64)
        var_res_this_call = np.zeros((self.horizon, 3), dtype=np.float64)

        for k in range(self.horizon):
            xk, uk = x_nom[k], u_nom[k]
            x_nom_k, A_nom, _ = attitude_dyn_dt_jac(
                xk, uk, self.mpc_dt, self.J, self.J_inv,
                steps=self.integrator_steps,
            )
            gp_active = self._gp_update_count >= self._gp_warmup_steps
            z_raw = np.concatenate([xk[4:], uk])
            if gp_active:
                mu_res, var_res, dmu_dz = self.gp_model.predict_with_jacobian(z_raw)
            else:
                mu_res = np.zeros(3, dtype=np.float64)
                var_res = np.zeros(3, dtype=np.float64)
                dmu_dz = np.zeros((3, 6), dtype=np.float64)

            var_res_this_call[k] = var_res
            x_nom[k + 1, :4] = x_nom_k[:4]
            x_nom[k + 1, 4:] = x_nom_k[4:] + mu_res

            Bd = np.zeros((self.nx, 3), dtype=np.float64)
            Bd[4:, :] = np.eye(3)
            dmu_dw = dmu_dz[:, :3]
            dmu_dx_full = np.zeros((3, self.nx), dtype=np.float64)
            dmu_dx_full[:, 4:] = dmu_dw
            A_seq[k] = A_nom + Bd @ dmu_dx_full

            q_proc = np.zeros((self.nx, self.nx), dtype=np.float64)
            q_proc[4:, 4:] = np.diag(np.maximum(var_res, 1e-10))
            q_proc += 1e-12 * np.eye(self.nx)
            q_proc_seq[k] = q_proc

        num_lm = self.num_bs * self.num_koz
        tighten = np.zeros((num_lm, self.horizon), dtype=np.float64)
        tighten_raw_arr = np.zeros((num_lm, self.horizon), dtype=np.float64)
        if self._use_koz:
            P = np.zeros((self.nx, self.nx), dtype=np.float64)
            for k in range(self.horizon):
                P = A_seq[k] @ P @ A_seq[k].T + q_proc_seq[k]
                qk = x_nom[k + 1, :4]
                Pq = P[:4, :4]
                for l in range(self.num_bs):
                    for m in range(self.num_koz):
                        lm_idx = l * self.num_koz + m
                        grad_q = 2.0 * (self.koz_matrices[l, m] @ qk)
                        sigma = np.sqrt(max(float(grad_q.T @ Pq @ grad_q), 0.0))
                        tr_raw = self.beta_cc * sigma
                        tighten_raw_arr[lm_idx, k] = tr_raw
                        tighten[lm_idx, k] = np.clip(tr_raw, 0.0, self.max_tighten)

        binding = tighten_raw_arr > self.max_tighten
        _DIAG["var_res_all"].append(var_res_this_call.copy())
        _DIAG["tighten_raw_all"].append(tighten_raw_arr.copy())
        _DIAG["tighten_all"].append(tighten.copy())
        _DIAG["binding_all"].append(binding.copy())
        _DIAG["call_count"] += 1

        return x_nom, tighten

    return _patched


def run_tighten(args):
    _silence_sim_loggers()

    from sim.controller.asgp_smpc import ASGPSMPC
    ASGPSMPC._build_local_models = _make_patched_build_local_models()

    from sim.montecarlo.single_episode import run_episode

    cfg = build_episode_cfg(args.exp_name, args.gp_mode)
    sc_cfg = cfg.sc
    q_init = np.array(sc_cfg.attitude.quat_init, dtype=float)
    q_goal = np.array(sc_cfg.attitude.quat_goal, dtype=float)
    omega_init = np.array(
        sc_cfg.attitude.get("omega_init", [0.0, 0.0, 0.0]), dtype=float
    )

    print("=" * 70)
    print(f"  ASGP tighten diagnostic  ({args.exp_name} / {args.gp_mode})")
    print("=" * 70)
    print(f"q_init     : {q_init}")
    print(f"q_goal     : {q_goal}")
    print(f"omega_init : {omega_init}")
    print(f"GP mode    : {args.gp_mode}")
    print(f"max_tighten: {cfg.controller.cc.max_tighten}")
    print(f"cc.alpha   : {cfg.controller.cc.alpha}")
    print()
    print("Running episode...")

    run_episode(cfg, q_init=q_init, q_goal=q_goal, omega_init=omega_init)

    n_calls = _DIAG["call_count"]
    print(f"\nEpisode finished. _build_local_models called {n_calls} times.")

    if n_calls == 0:
        print("WARNING: no diagnostic data collected (patch may not have been applied).")
        return

    all_var_res = np.concatenate(_DIAG["var_res_all"], axis=0)
    all_tighten_raw = np.concatenate(
        [x.flatten() for x in _DIAG["tighten_raw_all"]]
    )
    all_tighten = np.concatenate(
        [x.flatten() for x in _DIAG["tighten_all"]]
    )
    all_binding = np.concatenate(
        [x.flatten() for x in _DIAG["binding_all"]]
    )

    n_total = len(all_tighten_raw)
    n_binding = int(np.sum(all_binding))
    frac_binding = n_binding / max(n_total, 1)
    max_tighten = float(cfg.controller.cc.max_tighten)

    print("\n" + "=" * 70)
    print("  GP Variance (var_res) per output dimension")
    print("=" * 70)
    for j, name in enumerate(DIM_NAMES):
        v = all_var_res[:, j]
        print(f"  {name}  mean={v.mean():.4e}  std={v.std():.4e}  "
              f"min={v.min():.4e}  max={v.max():.4e}")

    print("\n" + "=" * 70)
    print("  Tightening values (across all KOZ × horizon × calls)")
    print("=" * 70)
    print(f"  Total entries      : {n_total}")
    print(f"  tighten_raw — mean : {all_tighten_raw.mean():.4e}")
    print(f"  tighten_raw — std  : {all_tighten_raw.std():.4e}")
    print(f"  tighten_raw — min  : {all_tighten_raw.min():.4e}")
    print(f"  tighten_raw — max  : {all_tighten_raw.max():.4e}")
    print(f"  tighten     — mean : {all_tighten.mean():.4e}")
    print(f"  tighten     — max  : {all_tighten.max():.4e}")
    print()
    print(f"  max_tighten ceiling: {max_tighten:.4f}")
    print(f"  Entries hitting ceiling: {n_binding} / {n_total}  "
          f"({frac_binding * 100:.1f}%)")

    pcts = [50, 75, 90, 95, 99, 100]
    print("\n  tighten_raw percentiles:")
    for p in pcts:
        print(f"    P{p:3d} = {np.percentile(all_tighten_raw, p):.4e}")

    binding_per_call = [b.mean() for b in _DIAG["binding_all"]]
    print(f"\n  Binding rate over time (per MPC call):")
    print(f"    first 10 calls : {np.mean(binding_per_call[:10]):.1%}")
    print(f"    last  10 calls : {np.mean(binding_per_call[-10:]):.1%}")
    print(f"    overall        : {np.mean(binding_per_call):.1%}")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if frac_binding > 0.5:
        print(f"  max_tighten IS binding on {frac_binding*100:.0f}% of tighten values.")
        print(f"  The ceiling of {max_tighten} is likely clipping real uncertainty signal.")
    elif frac_binding > 0.1:
        print(f"  max_tighten is moderately binding ({frac_binding*100:.0f}% of values).")
    else:
        print(f"  max_tighten is NOT the primary issue ({frac_binding*100:.1f}% binding).")
        mean_var = all_var_res.mean()
        if mean_var < 1e-6:
            print(f"  GP variance is very small (mean={mean_var:.2e}) — "
                  f"var_calib may be squashing uncertainty.")
        else:
            print(f"  GP variance looks non-trivial (mean={mean_var:.2e}).")
    print("=" * 70)


# ===========================================================================
#  MODE: residual
# ===========================================================================

def run_and_extract_residual(exp_name: str, gp_mode: str):
    from sim.montecarlo.single_episode import run_episode

    cfg = build_episode_cfg(exp_name, gp_mode)
    sc_cfg = cfg.sc
    q_init = np.array(sc_cfg.attitude.quat_init, dtype=float)
    q_goal = np.array(sc_cfg.attitude.quat_goal, dtype=float)
    omega_init = np.array(
        sc_cfg.attitude.get("omega_init", [0.0, 0.0, 0.0]), dtype=float
    )

    print(f"Running {exp_name} ({gp_mode})...")
    _, _, controller = run_episode(
        cfg, q_init=q_init, q_goal=q_goal, omega_init=omega_init,
    )

    var_scale = float(cfg.controller.cc.var_scale) if gp_mode == "fast" else 1.0
    data = {
        "pred": np.array(controller.gp_diag_pred),
        "actual_resid": np.array(controller.gp_diag_actual),
        "var": np.array(controller.gp_diag_var) * var_scale,
    }

    noise_vars = [float(m.noise.detach()) for m in controller.gp_model.models]
    y_std = controller.gp_model._y_std
    noise_raw = [nv * ys**2 for nv, ys in zip(noise_vars, y_std)]
    print(f"  noise_var σ (mrad/s): {[f'{np.sqrt(nr)*1e3:.3f}' for nr in noise_raw]}")

    return data


def plot_residuals(results: dict, output_path: str):
    """Each (exp, mode) gets its own column. Rows = ω₁, ω₂, ω₃."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = sorted(results.keys(), key=lambda k: (k[0], k[1]))
    n_cols = len(cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    dim_labels = [r"$\delta\omega_1$", r"$\delta\omega_2$", r"$\delta\omega_3$"]
    mode_colors = {"fast": "C0", "full": "C3"}
    ctrl_dt = 0.5
    warmup = 5

    for col, (exp, mode) in enumerate(cols):
        data = results[(exp, mode)]
        T = len(data["pred"])
        t = np.arange(T) * ctrl_dt + warmup * ctrl_dt
        resid_actual = data["actual_resid"]
        resid_pred = data["pred"]
        sigma = np.sqrt(data["var"])
        scale = 1e3
        color = mode_colors[mode]

        for row in range(3):
            ax = axes[row, col]
            act = resid_actual[:, row] * scale
            pred = resid_pred[:, row] * scale
            sig = sigma[:, row] * scale

            ax.plot(t, act, "k-", lw=0.7, label="actual residual", zorder=3)
            ax.plot(t, pred, color=color, ls="-", lw=0.8, label="GP mean")
            ax.fill_between(t, pred - 2 * sig, pred + 2 * sig,
                            alpha=0.2, color=color, label=r"$\pm 2\sigma$")
            ax.set_ylabel(f"{dim_labels[row]} [mrad/s]")
            if row == 0:
                ax.set_title(f"{exp} / {mode}")
            if row == 2:
                ax.set_xlabel("Time [s]")
            ax.set_ylim(-10, 10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")

            err = act - pred
            rmse = np.sqrt(np.mean(err ** 2))
            mean_sig = np.mean(sig)
            inside_2s = np.mean(np.abs(err) <= 2 * sig) * 100
            ax.text(0.02, 0.95,
                    f"RMSE={rmse:.2f}  σ̄={mean_sig:.2f}  CI={inside_2s:.0f}%",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


def run_residual(args):
    _silence_sim_loggers()

    results = {}
    for exp in args.exps:
        for mode in args.modes:
            results[(exp, mode)] = run_and_extract_residual(exp, mode)

    output_path = os.path.join(REPO, "outputs", "asgp_residual_tracking.png")
    plot_residuals(results, output_path)


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ASGP diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- offline --
    p_off = sub.add_parser("offline", help="Offline GP validation + online adaptation")
    p_off.add_argument("--exp_name", type=str, required=True,
                       help="Experiment name (e.g. exp1a, exp1b)")
    p_off.add_argument("--sc", type=str, default=None,
                       help="SC config key if different from exp_name")

    # -- tighten --
    p_tight = sub.add_parser("tighten", help="Check whether max_tighten clips uncertainty")
    p_tight.add_argument("--exp_name", type=str, default="exp1b")
    p_tight.add_argument("--gp_mode", type=str, default="full",
                         choices=["fast", "full"])

    # -- residual --
    p_res = sub.add_parser("residual", help="Plot actual residual vs GP mean ± 2σ")
    p_res.add_argument("--exps", nargs="+", default=["exp1b", "exp2a", "exp2b"])
    p_res.add_argument("--modes", nargs="+", default=["fast", "full"])

    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args)
    elif args.mode == "tighten":
        run_tighten(args)
    elif args.mode == "residual":
        run_residual(args)

    log.info("Done.")


if __name__ == "__main__":
    main()
