"""Reusable closed-loop Basilisk episode runner for PartySMPC.

Extracts the simulation setup from ``scripts/3_run_sim.py`` (sections 3-11)
into a single callable function so that both ``90_calibrate_party.py`` and
``3_run_mc.py`` can share the same physics pipeline without code
duplication.
"""

import os
import logging
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from Basilisk import __path__ as bskPath
from Basilisk.architecture import messaging, bskLogging
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.simulation import (
    exponentialAtmosphere,
    facetDragDynamicEffector,
    facetSRPDynamicEffector,
    GravityGradientEffector,
    extForceTorque,
    simpleNav,
)
from Basilisk.utilities import (
    RigidBodyKinematics as rbk,
    simSetPlanetEnvironment,
    SimulationBaseClass,
    macros,
    simIncludeGravBody,
    unitTestSupport,
)

from sim.koz.koz import KeepoutZone
from sim.utils.bsk_setup import set_sc, set_facets, normal2dcm_F0B
from sim.utils.inertia_transition import InertiaTransition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def run_episode(
    cfg: DictConfig,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    omega_init: np.ndarray,
    inertia: np.ndarray | None = None,
    controller_overrides: dict | None = None,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Run a single closed-loop Basilisk simulation with PartySMPC.

    Replicates sections 3-11 of ``scripts/3_run_sim.py`` in a lightweight,
    non-plotting form suitable for Monte-Carlo loops.

    Args:
        cfg: Full Hydra config (must contain ``sim``, ``env``, ``sc``,
            ``controller``, ``constraint`` groups).
        q_init: (4,) initial quaternion [q0, q1, q2, q3].
        q_goal: (4,) goal quaternion.
        omega_init: (3,) initial angular velocity [rad/s].
        inertia: (9,) or (3,3) inertia matrix in row-major.  ``None`` uses
            ``cfg.sc.inertia``.
        controller_overrides: Extra kwargs merged into the controller config
            before instantiation (e.g. ``{"record_linearization_error": True}``).

    Returns:
        state_traj: (T, 7) quaternion + angular velocity trajectory.
        ctrl_traj:  (T, 3) control torque trajectory.
        controller: The ``attController`` object (for accessing diagnostics).
    """
    # ------------------------------------------------------------------
    #  Config overrides (mutate a local copy)
    # ------------------------------------------------------------------
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Inject initial conditions into sc config
    OmegaConf.update(cfg, "sc.attitude.quat_init", q_init.tolist(), force_add=True)
    OmegaConf.update(cfg, "sc.attitude.omega_init", omega_init.tolist(), force_add=True)

    if inertia is not None:
        inertia_flat = np.asarray(inertia, dtype=float).flatten().tolist()
        OmegaConf.update(cfg, "sc.inertia", inertia_flat, force_add=True)
        # If no post_inertia override and no abrupt change, mirror pre-inertia
        if not cfg.sc.inertia_change.enabled:
            OmegaConf.update(cfg, "sc.post_inertia", inertia_flat, force_add=True)

    if controller_overrides:
        for key, val in controller_overrides.items():
            OmegaConf.update(cfg, f"controller.{key}", val, force_add=True)

    # ------------------------------------------------------------------
    #  BSK Logger (quiet for MC)
    # ------------------------------------------------------------------
    bskLogger = bskLogging.BSKLogger()
    bskLogger.setLogLevel(bskLogging.BSK_WARNING)

    # ------------------------------------------------------------------
    #  Timing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  Gravity
    # ------------------------------------------------------------------
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies("earth", "sun")
    earth = gravBodies["earth"]
    earth.isCentralBody = True

    gravity_model_path = "data/supportData/LocalGravData/GGM03S.txt"
    if os.path.exists(gravity_model_path):
        earth.useSphericalHarmonicsGravityModel(gravity_model_path, cfg.env.gravity_order)
    else:
        raise FileNotFoundError(f"Gravity model file not found: {gravity_model_path}")
    mu = earth.mu

    timeInitString = "2026 FEBRUARY 06 01:23:45.0"
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = "Earth"
    scSim.AddModelToTask(simTask, gravFactory.spiceObject, 100)

    # ------------------------------------------------------------------
    #  Spacecraft
    # ------------------------------------------------------------------
    sc = set_sc(cfg.sc, mu)
    scSim.AddModelToTask(simTask, sc, 100)
    gravFactory.addBodiesTo(sc)

    pre_inertia = np.array(cfg.sc.inertia, dtype=float).flatten()
    post_inertia = np.array(cfg.sc.post_inertia, dtype=float).flatten()
    inertia_transition = InertiaTransition(pre_inertia, post_inertia, cfg.sc.inertia_change)

    # ------------------------------------------------------------------
    #  Environment
    # ------------------------------------------------------------------
    facets = set_facets(cfg.sc.geometry.hub, cfg.sc.geometry.panel)

    if cfg.env.enable_drag:
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        simSetPlanetEnvironment.exponentialAtmosphere(atmo, "earth")
        scSim.AddModelToTask(simTask, atmo, 100)
        atmo.addSpacecraftToModel(sc.scStateOutMsg)

        dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        dragEffector.ModelTag = "DragEffector"
        dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
        for facet in facets:
            dragEffector.addFacet(facet["area"], facet["coeff"], facet["normal"], facet["location"])
        scSim.AddModelToTask(simTask, dragEffector, 100)

    if cfg.env.enable_srp:
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])

        srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
        srpEffector.ModelTag = "SRPEffector"
        numFacets = 10 if cfg.sc.geometry.panel.use_panel else 6
        srpEffector.setNumFacets(numFacets)
        srpEffector.setNumArticulatedFacets(0)
        srpEffector.sunInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[1])
        for facet in facets:
            dcm_F0B = normal2dcm_F0B(facet["normal"])
            srpEffector.addFacet(
                facet["area"], dcm_F0B, nHat_F, rotHat_F,
                facet["location"], facet["diff"], facet["spec"],
            )
        sc.addDynamicEffector(srpEffector)
        scSim.AddModelToTask(simTask, srpEffector, 100)

    if cfg.env.enable_gg:
        ggEffector = GravityGradientEffector.GravityGradientEffector()
        ggEffector.ModelTag = "GravityGradientEffector"
        ggEffector.addPlanetName(earth.planetName)
        sc.addDynamicEffector(ggEffector)
        scSim.AddModelToTask(simTask, ggEffector, 100)

    # ------------------------------------------------------------------
    #  Navigation
    # ------------------------------------------------------------------
    sNav = simpleNav.SimpleNav()
    sNav.ModelTag = "SimpleNavigation"
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    if rng_seed is not None:
        sNav.RNGSeed = rng_seed
    scSim.AddModelToTask(simTask, sNav, 90)

    # Sensor noise (matches 3_run_sim.py)
    if cfg.sc.sensors.use_nav_noise:
        attErrorSigma = float(cfg.sc.sensors.nav_noise.att) * macros.D2R
        omegaErrorSigma = float(cfg.sc.sensors.nav_noise.omega) * macros.D2R

        PMatrix = np.zeros((18, 18))
        errorBounds = np.zeros(18)

        np.fill_diagonal(PMatrix[6:9, 6:9], attErrorSigma)
        np.fill_diagonal(PMatrix[9:12, 9:12], omegaErrorSigma)
        errorBounds[6:9] = 3 * attErrorSigma
        errorBounds[9:12] = 3 * omegaErrorSigma

        sNav.PMatrix = PMatrix
        sNav.walkBounds = errorBounds

    # ------------------------------------------------------------------
    #  Actuators
    # ------------------------------------------------------------------
    ctrlFT = extForceTorque.ExtForceTorque()
    ctrlFT.ModelTag = "ControlForceTorque"
    sc.addDynamicEffector(ctrlFT)
    scSim.AddModelToTask(simTask, ctrlFT, 80)

    # ------------------------------------------------------------------
    #  FSW & Controller
    # ------------------------------------------------------------------
    attGuidance = inertial3D.inertial3D()
    attGuidance.ModelTag = "AttitudeGuidance"
    attGuidance.sigma_R0N = rbk.EP2MRP(np.array(q_goal))
    scSim.AddModelToTask(simTask, attGuidance)

    x_init = np.concatenate([q_init, omega_init])

    attController = hydra.utils.instantiate(
        cfg.controller,
        x_init=x_init,
        constraint_cfg=cfg.constraint,
    )

    warmup_time = float(getattr(attController, "warmup_time", 0.0))
    warmupTimeNanos = macros.sec2nano(warmup_time) if warmup_time > 0.0 else 0
    simEndTimeNanos = simTimeNanos + warmupTimeNanos

    crtlTask = "crtlTask"
    dynProcess.addTask(scSim.CreateNewTask(crtlTask, ctrlDtNanos))
    scSim.AddModelToTask(crtlTask, attController, 70)

    # ------------------------------------------------------------------
    #  Message Linking
    # ------------------------------------------------------------------
    attController.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)
    attController.attNavInMsg.subscribeTo(sNav.attOutMsg)
    ctrlFT.cmdTorqueInMsg.subscribeTo(attController.cmdTorqueOutMsg)

    # ------------------------------------------------------------------
    #  Logging (true state for evaluation, not noisy nav)
    # ------------------------------------------------------------------
    scStateLog = sc.scStateOutMsg.recorder(logDtNanos)
    ctrlTorqueLog = attController.cmdTorqueOutMsg.recorder(logDtNanos)
    scSim.AddModelToTask(simTask, scStateLog, 60)
    scSim.AddModelToTask(simTask, ctrlTorqueLog, 60)

    # ------------------------------------------------------------------
    #  Simulation Run
    # ------------------------------------------------------------------
    cmdTorque0 = messaging.CmdTorqueBodyMsgPayload()
    attController.cmdTorqueOutMsg.write(cmdTorque0)

    scSim.InitializeSimulation()

    displayDtNanos = macros.sec2nano(cfg.sim.ctrl_dt)
    currentSimTimeNanos = 0
    while currentSimTimeNanos < simEndTimeNanos:
        currentSimTimeNanos += displayDtNanos
        if currentSimTimeNanos > simEndTimeNanos:
            currentSimTimeNanos = simEndTimeNanos

        current_sec = currentSimTimeNanos * macros.NANO2SEC
        new_inertia = inertia_transition.get_inertia(current_sec)
        if new_inertia is not None:
            sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(new_inertia.tolist())

        scSim.ConfigureStopTime(currentSimTimeNanos)
        scSim.ExecuteSimulation()

    # ------------------------------------------------------------------
    #  Post-Processing
    # ------------------------------------------------------------------
    timeData = scStateLog.times() * macros.NANO2SEC
    sigmaBNData = scStateLog.sigma_BN
    omegaBNBData = scStateLog.omega_BN_B
    ctrlTorqueData = ctrlTorqueLog.torqueRequestBody

    # Drop warmup portion
    if warmup_time > 0.0:
        valid_mask = timeData >= warmup_time
        timeData = timeData[valid_mask] - warmup_time
        sigmaBNData = sigmaBNData[valid_mask]
        omegaBNBData = omegaBNBData[valid_mask]
        ctrlTorqueData = ctrlTorqueData[valid_mask]

    # MRP → quaternion
    sigma_sq_sum = np.sum(sigmaBNData ** 2, axis=1, keepdims=True)
    denom = 1.0 + sigma_sq_sum
    q0 = (1.0 - sigma_sq_sum) / denom
    q_vec = 2.0 * sigmaBNData / denom
    quatBNData = np.hstack([q0, q_vec])

    state_traj = np.hstack([quatBNData, omegaBNBData])  # (T, 7)
    ctrl_traj = ctrlTorqueData  # (T, 3)

    return state_traj, ctrl_traj, attController


# ---------------------------------------------------------------------------
#  KOZ violation checker (used by 3_run_mc_party.py)
# ---------------------------------------------------------------------------

def check_koz_violations(
    state_traj: np.ndarray,
    boresight_B_list: np.ndarray,
    koz_objects: List[KeepoutZone],
    return_traces: bool = False,
    viol_margin_deg: float = 0.0,
) -> tuple[int, int, float] | tuple[int, int, float, dict[str, np.ndarray]]:
    """Count KOZ constraint violations and minimum clearance across a state trajectory.

    Args:
        state_traj: (T, 7) with columns [q0, q1, q2, q3, wx, wy, wz].
        boresight_B_list: (L, 3) body-frame boresight unit vectors.
        koz_objects: List of ``KeepoutZone`` instances.
        return_traces: If True, also return per-KOZ min-across-boresights
            separation angle traces (avoids redundant recomputation).
        viol_margin_deg: Small margin [deg] subtracted from the KOZ half-angle
            when determining violations, so that separations within this margin
            of the boundary are *not* counted as violations (filters numerical noise).

    Returns:
        violated_steps: Number of timesteps with at least one KOZ violation.
        total_steps: Total number of timesteps checked.
        min_gap_deg: Minimum KOZ clearance angle [deg] across all timesteps,
            boresights, and KOZ zones. Negative means violation depth.
        koz_traces: (only if ``return_traces=True``) dict mapping KOZ name
            to (T,) array of min-across-boresights separation angles [deg].
    """
    T = state_traj.shape[0]
    quats = state_traj[:, :4]  # (T, 4)
    M = len(koz_objects)

    # Pre-compute KOZ direction array and cosine thresholds
    koz_dirs = np.stack([k.direction for k in koz_objects])  # (M, 3)
    koz_angles_deg = np.array([k.angle_deg for k in koz_objects])  # (M,)
    # Shrink the effective KOZ half-angle by the margin for violation counting
    koz_cos_thresh = np.cos(np.deg2rad(koz_angles_deg - viol_margin_deg))  # (M,)

    violated_steps = 0
    min_gap_deg = np.inf
    min_sep_deg = np.empty((T, M)) if return_traces else None

    for t in range(T):
        q = quats[t]
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ])

        # Rotate all boresights to inertial frame: (L, 3)
        bs_N = (R @ boresight_B_list.T).T  # (L, 3)

        # cos(angle) between each boresight and each KOZ direction: (L, M)
        cos_vals = bs_N @ koz_dirs.T

        # Violation: cos(angle) > cos(koz_angle) ⟺ angle < koz_angle
        if np.any(cos_vals > koz_cos_thresh[np.newaxis, :]):
            violated_steps += 1

        # Separation angles [deg]: (L, M)
        sep_deg = np.degrees(np.arccos(np.clip(cos_vals, -1, 1)))

        # Minimum gap: separation_angle - koz_half_angle
        gap_deg = sep_deg - koz_angles_deg[np.newaxis, :]  # (L, M)
        min_gap_deg = min(min_gap_deg, float(np.min(gap_deg)))

        if return_traces:
            min_sep_deg[t] = np.min(sep_deg, axis=0)  # (M,) min across boresights

    if return_traces:
        koz_traces = {
            koz.name: min_sep_deg[:, i] for i, koz in enumerate(koz_objects)
        }
        return violated_steps, T, min_gap_deg, koz_traces

    return violated_steps, T, min_gap_deg


def goal_diagnostics(
    state_traj: np.ndarray,
    q_goal: np.ndarray,
    att_thresh_deg: float = 5.0,
    omega_thresh: float = 0.01,
) -> tuple[bool, float, float]:
    """Compute goal-reaching diagnostics for the final state.

    Args:
        state_traj: (T, 7) with columns [q0, q1, q2, q3, wx, wy, wz].
        q_goal: (4,) goal quaternion.
        att_thresh_deg: Maximum attitude error [deg] for success.
        omega_thresh: Maximum angular velocity magnitude [rad/s] for success.

    Returns:
        reached: True if attitude error < att_thresh_deg AND ||omega|| < omega_thresh.
        att_err_deg: Attitude error [deg].
        omega_mag: Angular velocity magnitude [rad/s].
    """
    q_final = state_traj[-1, :4]
    omega_final = state_traj[-1, 4:]

    q_final = q_final / np.linalg.norm(q_final)
    q_goal_n = q_goal / np.linalg.norm(q_goal)

    # Quaternion error angle: 2 * arccos(|q_final . q_goal|)
    dot = np.abs(np.dot(q_final, q_goal_n))
    dot = np.clip(dot, 0.0, 1.0)
    att_err_deg = float(np.degrees(2.0 * np.arccos(dot)))

    omega_mag = float(np.linalg.norm(omega_final))

    reached = att_err_deg < att_thresh_deg and omega_mag < omega_thresh
    return reached, att_err_deg, omega_mag


def check_goal_reached(
    state_traj: np.ndarray,
    q_goal: np.ndarray,
    att_thresh_deg: float = 5.0,
    omega_thresh: float = 0.01,
) -> bool:
    """Check if the final state is close to the goal attitude.

    Args:
        state_traj: (T, 7) with columns [q0, q1, q2, q3, wx, wy, wz].
        q_goal: (4,) goal quaternion.
        att_thresh_deg: Maximum attitude error [deg] for success.
        omega_thresh: Maximum angular velocity magnitude [rad/s] for success.

    Returns:
        True if attitude error < att_thresh_deg AND ||omega|| < omega_thresh.
    """
    reached, _, _ = goal_diagnostics(state_traj, q_goal, att_thresh_deg, omega_thresh)
    return reached


def compute_control_effort(ctrl_traj: np.ndarray, dt: float) -> float:
    """Compute total control effort (torque impulse) via trapezoidal integration.

    Returns:
        Integral of ||u(t)||_2 over the trajectory [Nm*s].
    """
    norms = np.linalg.norm(ctrl_traj, axis=1)  # (T,)
    time_vec = np.arange(len(ctrl_traj)) * dt
    return float(np.trapz(norms, time_vec))


def compute_koz_separation_traces(
    state_traj: np.ndarray,
    boresight_B_list: np.ndarray,
    koz_objects: List[KeepoutZone],
) -> dict[str, np.ndarray]:
    """Compute per-KOZ minimum separation angle across boresights at each timestep.

    Convenience wrapper around ``check_koz_violations(..., return_traces=True)``.
    """
    _, _, _, traces = check_koz_violations(
        state_traj, boresight_B_list, koz_objects, return_traces=True,
    )
    return traces


