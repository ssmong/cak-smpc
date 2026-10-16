"""
BSK-based validation of the 6-state Augmented UKF estimator.

Runs the full BSK simulation (gravity gradient, SRP, drag, sensor noise)
with the UKF-SMPC controller and records estimator diagnostics:
  - J_hat convergence (estimated vs true inertia)
  - P_J covariance evolution
  - State estimation error (attitude + omega)

Supports batch experiments (configured in sim_ukf.yaml `experiments` list)
and inertia change scenarios via sc.inertia_change config.

Usage:
    python -m scripts.10_calibrate_ukf
    python -m scripts.10_calibrate_ukf sim.sim_time=200
"""

import os
import copy
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging

from Basilisk.fswAlgorithms import inertial3D

from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragDynamicEffector
from Basilisk.simulation import facetSRPDynamicEffector
from Basilisk.simulation import GravityGradientEffector
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav

from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import simSetPlanetEnvironment
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import unitTestSupport

from sim.utils.bsk_setup import set_sc, set_facets, normal2dcm_F0B
from sim.utils.inertia_transition import InertiaTransition
from sim.utils.terminal_utils import format_time_stats_ms
from src.utils.math_numpy import j_mat_to_vec

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

bskPath = __path__[0]


# ======================================================================
#  Plotting
# ======================================================================

J_LABELS = ["Jxx", "Jxy", "Jxz", "Jyy", "Jyz", "Jzz"]
# Subplot order: Row1 Jxx|Jxy, Row2 Jyy|Jyz, Row3 Jzz|Jzx  (matches MATLAB)
SUBPLOT_ORDER = [0, 1, 3, 4, 5, 2]


def plot_inertia_estimation(
    t_est: np.ndarray,
    j_hat: np.ndarray,
    P_J_diag: np.ndarray,
    j_true: np.ndarray,
    t_switch: float | None,
    figure_dir: str,
    prefix: str = "",
):
    """Plot estimated vs true inertia with +/-2sigma bands."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{prefix}Inertia Estimation (Augmented UKF)", fontsize=14)

    for sp, ax in enumerate(axes.flat):
        idx = SUBPLOT_ORDER[sp]

        ax.plot(t_est, j_true[:, idx], "k--", lw=1.5, label="True")
        ax.plot(t_est, j_hat[:, idx], "b-", lw=1.0, label="Estimated")

        sigma = np.sqrt(P_J_diag[:, idx])
        ax.fill_between(t_est, j_hat[:, idx] - 2*sigma, j_hat[:, idx] + 2*sigma,
                         alpha=0.2, color="blue", label=r"$\pm 2\sigma$")

        if t_switch is not None:
            ax.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7, label="Switch")

        ax.set_ylabel(J_LABELS[idx])
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{prefix}inertia_estimation.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved {prefix}inertia estimation plot")


def plot_inertia_error(
    t_est: np.ndarray,
    j_hat: np.ndarray,
    P_J_diag: np.ndarray,
    j_true: np.ndarray,
    t_switch: float | None,
    figure_dir: str,
    prefix: str = "",
):
    """Plot inertia estimation error with +/-2sigma bands."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{prefix}Inertia Estimation Error", fontsize=14)

    for sp, ax in enumerate(axes.flat):
        idx = SUBPLOT_ORDER[sp]

        err = j_hat[:, idx] - j_true[:, idx]
        sigma = np.sqrt(P_J_diag[:, idx])

        ax.plot(t_est, err, "b-", lw=0.8)
        ax.fill_between(t_est, err - 2*sigma, err + 2*sigma, alpha=0.2, color="blue", label=r"$\pm 2\sigma$")
        ax.axhline(0, color="k", ls="-", lw=0.5)

        if t_switch is not None:
            ax.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7)

        ax.set_ylabel(J_LABELS[idx])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{prefix}inertia_error.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved {prefix}inertia error plot")


def plot_P_J_convergence(
    t_est: np.ndarray,
    P_J_diag: np.ndarray,
    t_switch: float | None,
    figure_dir: str,
    prefix: str = "",
):
    """Plot P_J diagonal convergence (log scale)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{prefix}Inertia Covariance (P_J diagonal)", fontsize=14)

    for idx, ax in enumerate(axes.flat):
        ax.semilogy(t_est, P_J_diag[:, idx], "b-", lw=1.0)
        ax.set_ylabel(f"P_{{{J_LABELS[idx]}}}")
        ax.grid(True, alpha=0.3, which="both")
        if t_switch is not None:
            ax.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{prefix}P_J_convergence.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved {prefix}P_J convergence plot")


def plot_state_error(
    t_log: np.ndarray,
    x_est: np.ndarray,
    x_true: np.ndarray,
    t_switch: float | None,
    figure_dir: str,
    prefix: str = "",
):
    """Plot state estimation error (attitude angle + omega norm)."""
    N = min(len(x_est), len(x_true))
    t = t_log[:N]
    x_e = x_est[:N]
    x_t = x_true[:N]

    # Quaternion geodesic error
    q_err_deg = np.zeros(N)
    for i in range(N):
        dot = np.clip(np.abs(np.dot(x_e[i, :4], x_t[i, :4])), 0.0, 1.0)
        q_err_deg[i] = np.degrees(2.0 * np.arccos(dot))

    w_err = np.linalg.norm(x_e[:, 4:] - x_t[:, 4:], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{prefix}State Estimation Error (UKF vs True)", fontsize=14)

    ax1.plot(t, q_err_deg, "b-", lw=0.8)
    ax1.set_ylabel("Attitude error [deg]")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    if t_switch is not None:
        ax1.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7, label="Switch")
        ax1.legend()

    ax2.plot(t, np.degrees(w_err), "b-", lw=0.8)
    ax2.set_ylabel("omega error [deg/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    if t_switch is not None:
        ax2.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{prefix}state_error.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved {prefix}state error plot")


def print_summary(
    j_hat_final: np.ndarray,
    P_J_diag_final: np.ndarray,
    j_true: np.ndarray,
    desc: str = "",
):
    """Print final estimation statistics."""
    log.info(f"\n{'='*60}")
    log.info(f"  {desc}  Final inertia estimation")
    log.info(f"{'='*60}")
    log.info(f"  {'':>6s}  {'True':>10s}  {'Est':>10s}  {'Error':>10s}  {'2s':>10s}")
    log.info(f"  {'-'*54}")
    for i, label in enumerate(J_LABELS):
        err = j_hat_final[i] - j_true[i]
        sigma2 = 2.0 * np.sqrt(P_J_diag_final[i])
        within = "|" if abs(err) <= sigma2 else "X"
        log.info(
            f"  {label:>6s}  {j_true[i]:10.4f}  {j_hat_final[i]:10.4f}"
            f"  {err:+10.4f}  {sigma2:10.4f}  {within}"
        )
    log.info(f"{'='*60}")


# ======================================================================
#  Single experiment runner
# ======================================================================

def run_single_experiment(
    cfg: DictConfig,
    sc_cfg: DictConfig,
    noise_cfg: DictConfig,
    constraint_cfg: DictConfig,
    output_dir: str,
    desc: str = "",
    prefix: str = "",
):
    """Run a single BSK simulation experiment and generate plots."""
    bskLogger = bskLogging.BSKLogger()
    bskLogger.setLogLevel(bskLogging.BSK_INFORMATION)

    figureDir = os.path.join(output_dir, "figure")
    dataDir = os.path.join(output_dir, "data")
    os.makedirs(figureDir, exist_ok=True)
    os.makedirs(dataDir, exist_ok=True)

    seed = cfg.seed
    simTimeNanos = macros.sec2nano(cfg.sim.sim_time)
    simDtNanos = macros.sec2nano(cfg.sim.sim_dt)
    logDtNanos = macros.sec2nano(cfg.sim.log_dt)
    ctrlDtNanos = macros.sec2nano(cfg.sim.ctrl_dt)

    simTask = "simTask"
    simProcess = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(False)

    dynProcess = scSim.CreateNewProcess(simProcess)
    dynProcess.addTask(scSim.CreateNewTask(simTask, simDtNanos))

    # --- Gravity ---
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('earth', 'sun')
    earth = gravBodies['earth']
    earth.isCentralBody = True

    gravity_model_path = 'data/supportData/LocalGravData/GGM03S.txt'
    if os.path.exists(gravity_model_path):
        earth.useSphericalHarmonicsGravityModel(gravity_model_path, cfg.env.gravity_order)
    else:
        raise FileNotFoundError(f"Gravity model file not found: {gravity_model_path}")

    mu = earth.mu

    timeInitString = '2026 FEBRUARY 06 01:23:45.0'
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTask, gravFactory.spiceObject, 100)

    # --- Spacecraft ---
    sc = set_sc(sc_cfg, mu)
    scSim.AddModelToTask(simTask, sc, 100)
    gravFactory.addBodiesTo(sc)

    pre_inertia = np.array(sc_cfg.inertia, dtype=float).flatten()
    post_inertia = np.array(sc_cfg.post_inertia, dtype=float).flatten()
    inertia_transition = InertiaTransition(pre_inertia, post_inertia, sc_cfg.inertia_change)

    # --- Environment ---
    facets = set_facets(sc_cfg.geometry.hub, sc_cfg.geometry.panel)

    if cfg.env.enable_drag:
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        simSetPlanetEnvironment.exponentialAtmosphere(atmo, 'earth')
        scSim.AddModelToTask(simTask, atmo, 100)
        atmo.addSpacecraftToModel(sc.scStateOutMsg)

        dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        dragEffector.ModelTag = "DragEffector"
        dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
        for facet in facets:
            dragEffector.addFacet(facet['area'], facet['coeff'], facet['normal'], facet['location'])
        scSim.AddModelToTask(simTask, dragEffector, 100)

    if cfg.env.enable_srp:
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])

        srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
        srpEffector.ModelTag = "SRPEffector"
        numFacets = 10 if sc_cfg.geometry.panel.use_panel else 6
        srpEffector.setNumFacets(numFacets)
        srpEffector.setNumArticulatedFacets(0)
        srpEffector.sunInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[1])
        for facet in facets:
            dcm_F0B = normal2dcm_F0B(facet['normal'])
            srpEffector.addFacet(facet['area'], dcm_F0B, nHat_F, rotHat_F,
                                  facet['location'], facet['diff'], facet['spec'])
        sc.addDynamicEffector(srpEffector)
        scSim.AddModelToTask(simTask, srpEffector, 100)

    if cfg.env.enable_gg:
        ggEffector = GravityGradientEffector.GravityGradientEffector()
        ggEffector.ModelTag = "GravityGradientEffector"
        ggEffector.addPlanetName(earth.planetName)
        sc.addDynamicEffector(ggEffector)
        scSim.AddModelToTask(simTask, ggEffector, 100)

    # --- Navigation ---
    sNav = simpleNav.SimpleNav()
    sNav.ModelTag = "SimpleNavigation"
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    scSim.AddModelToTask(simTask, sNav, 90)

    if sc_cfg.sensors.use_nav_noise:
        attErrorSigma = float(sc_cfg.sensors.nav_noise.att) * macros.D2R
        omegaErrorSigma = float(sc_cfg.sensors.nav_noise.omega) * macros.D2R

        PMatrix = np.zeros((18, 18))
        errorBounds = np.zeros(18)
        np.fill_diagonal(PMatrix[6:9, 6:9], attErrorSigma)
        np.fill_diagonal(PMatrix[9:12, 9:12], omegaErrorSigma)
        errorBounds[6:9] = 3 * attErrorSigma
        errorBounds[9:12] = 3 * omegaErrorSigma

        sNav.PMatrix = PMatrix
        sNav.walkBounds = errorBounds
        sNav.RNGSeed = seed

    # --- Actuator ---
    ctrlFT = extForceTorque.ExtForceTorque()
    ctrlFT.ModelTag = "ControlForceTorque"
    sc.addDynamicEffector(ctrlFT)
    scSim.AddModelToTask(simTask, ctrlFT, 80)

    # --- Guidance & Controller ---
    attGuidance = inertial3D.inertial3D()
    attGuidance.ModelTag = "AttitudeGuidance"
    q_goal = sc_cfg.attitude.quat_goal
    attGuidance.sigma_R0N = rbk.EP2MRP(np.array(q_goal))
    scSim.AddModelToTask(simTask, attGuidance)

    q_init = sc_cfg.attitude.quat_init
    omega_init = sc_cfg.attitude.omega_init
    x_init = np.concatenate([q_init, omega_init])

    # Build controller config with per-experiment noise
    ctrl_cfg = copy.deepcopy(cfg.controller)
    # Override noise-dependent fields
    ctrl_cfg.cc.Sigma_w_diag = list(noise_cfg.Sigma_w_diag)
    ctrl_cfg.estimator.Q_state_diag = list(noise_cfg.Sigma_w_diag)
    # Override inertia from this experiment's sc
    ctrl_cfg.inertia = list(sc_cfg.inertia)

    attController = hydra.utils.instantiate(
        ctrl_cfg,
        x_init=x_init,
        constraint_cfg=constraint_cfg,
    )

    warmup_time = float(getattr(attController, "warmup_time", 0.0))
    warmupTimeNanos = macros.sec2nano(warmup_time) if warmup_time > 0.0 else 0
    simEndTimeNanos = simTimeNanos + warmupTimeNanos

    ctrlTask = "ctrlTask"
    dynProcess.addTask(scSim.CreateNewTask(ctrlTask, ctrlDtNanos))
    scSim.AddModelToTask(ctrlTask, attController, 70)

    # --- Message Linking ---
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    attController.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)
    attController.attNavInMsg.subscribeTo(sNav.attOutMsg)
    ctrlFT.cmdTorqueInMsg.subscribeTo(attController.cmdTorqueOutMsg)

    # --- Logging ---
    sNavAttLog = sNav.attOutMsg.recorder(logDtNanos)
    scLog = sc.scStateOutMsg.recorder(logDtNanos)
    ctrlTorqueLog = attController.cmdTorqueOutMsg.recorder(logDtNanos)

    scSim.AddModelToTask(simTask, sNavAttLog, 60)
    scSim.AddModelToTask(simTask, scLog, 60)
    scSim.AddModelToTask(simTask, ctrlTorqueLog, 60)

    # --- Simulation Run ---
    cmdTorque0 = messaging.CmdTorqueBodyMsgPayload()
    attController.cmdTorqueOutMsg.write(cmdTorque0)

    log.info(f"=== Starting: {desc} ===")
    scSim.InitializeSimulation()

    displayDtNanos = macros.sec2nano(cfg.sim.ctrl_dt)
    currentSimTimeNanos = 0
    simEndTimeSec = simEndTimeNanos * macros.NANO2SEC

    q_goal_np = np.array(q_goal, dtype=float)
    q_goal_np = q_goal_np / np.linalg.norm(q_goal_np)

    while currentSimTimeNanos < simEndTimeNanos:
        currentSimTimeNanos += displayDtNanos
        if currentSimTimeNanos > simEndTimeNanos:
            currentSimTimeNanos = simEndTimeNanos

        current_sec = currentSimTimeNanos * macros.NANO2SEC
        new_inertia = inertia_transition.get_inertia(current_sec)
        if new_inertia is not None:
            sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(new_inertia.tolist())
            if inertia_transition.is_complete:
                log.info(
                    f"Inertia transition ({inertia_transition.transition_type}) "
                    f"complete at t={current_sec:.3f}s"
                )

        scSim.ConfigureStopTime(currentSimTimeNanos)
        scSim.ExecuteSimulation()

        # Live progress
        currentSec = currentSimTimeNanos * macros.NANO2SEC
        j_hat = getattr(attController, '_j_hat_vec', np.zeros(6))
        failures = getattr(attController, 'solve_failures', 0)
        n_solves = len(getattr(attController, 'wall_times', []))

        navData = sNav.attOutMsg.read()
        sigma_BN = np.array(navData.sigma_BN)
        s2 = np.dot(sigma_BN, sigma_BN)
        q0 = (1.0 - s2) / (1.0 + s2)
        q_vec = 2.0 * sigma_BN / (1.0 + s2)
        q_now = np.array([q0, *q_vec])
        q_now = q_now / (np.linalg.norm(q_now) + 1e-12)
        dot_q = np.clip(np.abs(np.dot(q_now, q_goal_np)), 0.0, 1.0)
        q_err_deg = float(np.degrees(2.0 * np.arccos(dot_q)))

        print(
            f"\r  [{desc}] t={currentSec:6.1f}/{simEndTimeSec:.0f}s "
            f"err={q_err_deg:5.1f}\u00b0 "
            f"J=[{j_hat[0]:.1f},{j_hat[3]:.1f},{j_hat[5]:.1f}] "
            f"solves={n_solves} fail={failures}",
            end="", flush=True,
        )

    print()
    log.info(f"=== Completed: {desc} ===")

    # --- Post-Processing ---
    timeData = sNavAttLog.times() * macros.NANO2SEC
    ctrlTorqueData = ctrlTorqueLog.torqueRequestBody

    # True state from scLog (no sensor noise)
    scSigmaBN = scLog.sigma_BN
    scOmegaBN = scLog.omega_BN_B

    # Drop warmup
    if warmup_time > 0.0:
        valid_mask = timeData >= warmup_time
        timeData = timeData[valid_mask] - warmup_time
        ctrlTorqueData = ctrlTorqueData[valid_mask]
        scSigmaBN = scSigmaBN[valid_mask]
        scOmegaBN = scOmegaBN[valid_mask]

    # Convert MRP to quaternion (true state)
    sigma_sq = np.sum(scSigmaBN**2, axis=1, keepdims=True)
    denom = 1.0 + sigma_sq
    q0_true = (1.0 - sigma_sq) / denom
    q_vec_true = 2.0 * scSigmaBN / denom
    quatTrue = np.hstack([q0_true, q_vec_true])
    x_true = np.hstack([quatTrue, scOmegaBN])

    # Estimator history from controller
    j_hat_hist = attController.get_j_hat_history()
    P_J_diag_hist = attController.get_P_J_diag_history()
    x_est_hist = attController.get_x_est_history()

    N_est = len(j_hat_hist)
    est_dt = float(cfg.sim.ctrl_dt)
    t_est = np.arange(N_est) * est_dt

    # True inertia profile (handles both step and continuous drift)
    j_true_profile = np.array([
        j_mat_to_vec(inertia_transition.compute_inertia(t).reshape(3, 3))
        for t in t_est
    ])

    t_switch = inertia_transition.transition_start_sec

    # Summary
    if N_est > 0:
        print_summary(j_hat_hist[-1], P_J_diag_hist[-1], j_true_profile[-1], desc)

    if hasattr(attController, "get_wall_time_stats"):
        wall_stats = attController.get_wall_time_stats()
        log.info(f"Controller wall-time: {format_time_stats_ms(wall_stats, decimals=2)}")

    # Plotting
    if N_est > 0:
        plot_inertia_estimation(
            t_est, j_hat_hist, P_J_diag_hist,
            j_true_profile, t_switch, figureDir, prefix)

        plot_inertia_error(
            t_est, j_hat_hist, P_J_diag_hist,
            j_true_profile, t_switch, figureDir, prefix)

        plot_P_J_convergence(t_est, P_J_diag_hist, t_switch, figureDir, prefix)

        # Align true state to estimator time grid
        log_dt = float(cfg.sim.log_dt)
        est_stride = max(1, int(round(est_dt / log_dt)))
        x_true_aligned = x_true[::est_stride]
        t_true_aligned = timeData[::est_stride]
        plot_state_error(t_true_aligned, x_est_hist, x_true_aligned, t_switch, figureDir, prefix)

    # Control torque plot
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, label in enumerate(["u_x", "u_y", "u_z"]):
        ax.plot(timeData, ctrlTorqueData[:, i], lw=0.8, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title(f"{prefix}Control Torque (UKF-SMPC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if t_switch is not None:
        ax.axvline(t_switch, color="red", ls=":", lw=1, alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(figureDir, f"{prefix}control_torque.png"), dpi=150)
    plt.close(fig)

    # Save data
    np.savez(
        os.path.join(dataDir, f"{prefix}estimator_data.npz"),
        t_est=t_est,
        j_hat=j_hat_hist,
        P_J_diag=P_J_diag_hist,
        x_est=x_est_hist,
        j_true_profile=j_true_profile,
        t_switch=np.array([t_switch if t_switch is not None else -1.0]),
        time=timeData,
        x_true=x_true,
        control=ctrlTorqueData,
    )
    log.info(f"Data saved to {dataDir}/{prefix}estimator_data.npz")


# ======================================================================
#  Main
# ======================================================================

@hydra.main(config_path="../configs", config_name="calibrate_ukf", version_base=None)
def run(cfg: DictConfig):
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception:
        output_dir = "."

    experiments = cfg.get("experiments", None)

    if experiments is not None and len(experiments) > 0:
        # Batch mode: run all experiments
        for exp_idx, exp in enumerate(experiments):
            exp_sc_name = exp.sc
            exp_noise_name = exp.noise
            exp_constraint_name = exp.constraint
            exp_desc = exp.get("desc", f"exp{exp_idx}")

            log.info(f"\n{'#'*60}")
            log.info(f"  Experiment {exp_idx+1}/{len(experiments)}: {exp_desc}")
            log.info(f"  sc={exp_sc_name}, noise={exp_noise_name}, constraint={exp_constraint_name}")
            log.info(f"{'#'*60}")

            # Load per-experiment configs via OmegaConf
            sc_cfg = OmegaConf.load(f"configs/sc/{exp_sc_name}.yaml")
            # Merge with sc_default for inherited fields
            sc_default = OmegaConf.load("configs/sc/sc_default.yaml")
            sc_cfg = OmegaConf.merge(sc_default, sc_cfg)

            noise_cfg = OmegaConf.load(f"configs/noise/{exp_noise_name}.yaml")
            constraint_cfg = OmegaConf.load(f"configs/constraint/{exp_constraint_name}.yaml")

            exp_output_dir = os.path.join(output_dir, exp_sc_name)
            prefix = f"{exp_sc_name}_"

            run_single_experiment(
                cfg=cfg,
                sc_cfg=sc_cfg,
                noise_cfg=noise_cfg,
                constraint_cfg=constraint_cfg,
                output_dir=exp_output_dir,
                desc=exp_desc,
                prefix=prefix,
            )
    else:
        # Single mode: use top-level sc/noise/constraint
        run_single_experiment(
            cfg=cfg,
            sc_cfg=cfg.sc,
            noise_cfg=cfg.noise,
            constraint_cfg=cfg.constraint,
            output_dir=output_dir,
            desc="Single run",
            prefix="",
        )

    log.info("All experiments completed!")


if __name__ == "__main__":
    run()
