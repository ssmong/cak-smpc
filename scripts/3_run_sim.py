import os
import json
import numpy as np
import scipy.io as sio

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
from sim.utils.plot_utils import (
    plot_sim_results, plot_W_history, plot_uncertainty_state_space,
    plot_control_decomposition, plot_latent_diagnostics,
    plot_v_plan_diagnostics,
)
from sim.utils.terminal_utils import format_time_stats_ms
from sim.koz import KeepoutZone

bskPath = __path__[0]

@hydra.main(config_path="../configs", config_name="sim_party", version_base=None)
def run(cfg: DictConfig):
    # -------------------------------------------------------------------------
    #   1. Logger Setup
    # -------------------------------------------------------------------------
    bskLogger = bskLogging.BSKLogger()
    bskLogger.setLogLevel(bskLogging.BSK_INFORMATION)
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "Initializing Simulation in Single-Spacecraft Mode...")

    # -------------------------------------------------------------------------
    #   2. Experiment Setup (Hydra Output Dir)
    # -------------------------------------------------------------------------
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception as e:
        bskLogger.bskLog(bskLogging.BSK_WARNING, f"Failed to get Hydra output directory: {e}. Using current directory.")
        output_dir = "."
    
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

    # -------------------------------------------------------------------------
    #   3. Gravity Setup
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    #   4. Spacecraft Setup
    # -------------------------------------------------------------------------
    # Use sc config for the single spacecraft
    sc = set_sc(cfg.sc, mu)
    scSim.AddModelToTask(simTask, sc, 100)
    gravFactory.addBodiesTo(sc)

    pre_inertia = np.array(cfg.sc.inertia, dtype=float).flatten()
    post_inertia = np.array(cfg.sc.post_inertia, dtype=float).flatten()
    if pre_inertia.size != 9:
        raise ValueError(f"sc.inertia must have 9 elements, got {pre_inertia.size}")
    if post_inertia.size != 9:
        raise ValueError(f"sc.post_inertia must have 9 elements, got {post_inertia.size}")
    inertia_transition = InertiaTransition(pre_inertia, post_inertia, cfg.sc.inertia_change)

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "SPACECRAFT SETUP COMPLETED")

    # -------------------------------------------------------------------------
    #   5. Environment Setup
    # -------------------------------------------------------------------------
    facets = set_facets(cfg.sc.geometry.hub, cfg.sc.geometry.panel)
    
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
        numFacets = 10 if cfg.sc.geometry.panel.use_panel else 6
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

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "ENVIRONMENT SETUP COMPLETED")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"    - Drag: {cfg.env.enable_drag}")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"    - SRP : {cfg.env.enable_srp}")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"    - GG  : {cfg.env.enable_gg}")

    # -------------------------------------------------------------------------
    #   6. Navigation
    # -------------------------------------------------------------------------
    sNav = simpleNav.SimpleNav()
    sNav.ModelTag = "SimpleNavigation"
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    scSim.AddModelToTask(simTask, sNav, 90)

    if cfg.sc.sensors.use_nav_noise:
        bskLogger.bskLog(bskLogging.BSK_INFORMATION, "Navigation Noise Enabled.")
        posErrorSigma = float(cfg.sc.sensors.nav_noise.pos)
        velErrorSigma = float(cfg.sc.sensors.nav_noise.vel)
        attErrorSigma = float(cfg.sc.sensors.nav_noise.att) * macros.D2R
        omegaErrorSigma = float(cfg.sc.sensors.nav_noise.omega) * macros.D2R
        
        PMatrix = np.zeros((18, 18))
        errorBounds = np.zeros(18)

        np.fill_diagonal(PMatrix[0:3, 0:3], posErrorSigma)
        np.fill_diagonal(PMatrix[3:6, 3:6], velErrorSigma)
        np.fill_diagonal(PMatrix[6:9, 6:9], attErrorSigma)
        np.fill_diagonal(PMatrix[9:12, 9:12], omegaErrorSigma)

        errorBounds[0:3] = 3 * posErrorSigma
        errorBounds[3:6] = 3 * velErrorSigma
        errorBounds[6:9] = 3 * attErrorSigma
        errorBounds[9:12] = 3 * omegaErrorSigma

        sNav.PMatrix = PMatrix
        sNav.walkBounds = errorBounds
        sNav.RNGSeed = seed

    # -------------------------------------------------------------------------
    #   7. Actuators
    # -------------------------------------------------------------------------
    ctrlFT = extForceTorque.ExtForceTorque()
    ctrlFT.ModelTag = "ControlForceTorque"
    sc.addDynamicEffector(ctrlFT)
    scSim.AddModelToTask(simTask, ctrlFT, 80)

    # -------------------------------------------------------------------------
    #   8. FSW & Controller
    # -------------------------------------------------------------------------
    attGuidance = inertial3D.inertial3D()
    attGuidance.ModelTag = "AttitudeGuidance"
    q_goal = cfg.sc.attitude.quat_goal
    attGuidance.sigma_R0N = rbk.EP2MRP(np.array(q_goal))
    
    scSim.AddModelToTask(simTask, attGuidance)

    q_init = cfg.sc.attitude.quat_init
    omega_init = cfg.sc.attitude.omega_init
    x_init = np.concatenate([q_init, omega_init])
    
    attController = hydra.utils.instantiate(
        cfg.controller,
        x_init=x_init,
        constraint_cfg=cfg.constraint
    )

    # Extend simulation time to include controller warmup (if defined)
    warmup_time = float(getattr(attController, "warmup_time", 0.0))
    warmupTimeNanos = macros.sec2nano(warmup_time) if warmup_time > 0.0 else 0
    simEndTimeNanos = simTimeNanos + warmupTimeNanos

    crtlTask = "crtlTask"
    dynProcess.addTask(scSim.CreateNewTask(crtlTask, ctrlDtNanos))
    scSim.AddModelToTask(crtlTask, attController, 70)

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "CONTROLLER SETUP COMPLETED")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"    - Attitude Controller: {attController.ModelTag}")

    # -------------------------------------------------------------------------
    #   9. Message Linking
    # -------------------------------------------------------------------------    
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    
    attController.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)
    attController.attNavInMsg.subscribeTo(sNav.attOutMsg)

    ctrlFT.cmdTorqueInMsg.subscribeTo(attController.cmdTorqueOutMsg)

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "MESSAGE LINKING COMPLETED")

    # -------------------------------------------------------------------------
    #   10. Logging
    # -------------------------------------------------------------------------
    sNavAttLog = sNav.attOutMsg.recorder(logDtNanos)
    sNavLog = sNav.transOutMsg.recorder(logDtNanos)
    scLog = sc.scStateOutMsg.recorder(logDtNanos)
    ctrlTorqueLog = attController.cmdTorqueOutMsg.recorder(logDtNanos)

    scSim.AddModelToTask(simTask, sNavAttLog, 60)
    scSim.AddModelToTask(simTask, sNavLog, 60)
    scSim.AddModelToTask(simTask, scLog, 60)
    scSim.AddModelToTask(simTask, ctrlTorqueLog, 60)

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "LOGGING SETUP COMPLETED")

    # -------------------------------------------------------------------------
    #   11. Simulation Run
    # -------------------------------------------------------------------------
    
    # Initialize messages to zero to avoid read-before-write errors
    cmdTorque0 = messaging.CmdTorqueBodyMsgPayload()
    attController.cmdTorqueOutMsg.write(cmdTorque0)
    
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "SIMULATION STARTED")
    scSim.InitializeSimulation()

    displayDt = cfg.sim.ctrl_dt
    displayDtNanos = macros.sec2nano(displayDt)

    currentSimTimeNanos = 0
    simEndTimeSec = simEndTimeNanos * macros.NANO2SEC

    # Goal quaternion for live error computation
    q_goal_np = np.array(q_goal, dtype=float)
    q_goal_np = q_goal_np / np.linalg.norm(q_goal_np)

    while currentSimTimeNanos < simEndTimeNanos:
        # Simulation step forward
        currentSimTimeNanos += displayDtNanos
        if currentSimTimeNanos > simEndTimeNanos:
            currentSimTimeNanos = simEndTimeNanos

        currentSec_for_inertia = currentSimTimeNanos * macros.NANO2SEC
        new_inertia = inertia_transition.get_inertia(currentSec_for_inertia)
        if new_inertia is not None:
            sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(new_inertia.tolist())
            if inertia_transition.is_complete:
                bskLogger.bskLog(
                    bskLogging.BSK_INFORMATION,
                    f"Inertia transition ({inertia_transition.transition_type}) "
                    f"complete at t={currentSec_for_inertia:.3f}s",
                )

        scSim.ConfigureStopTime(currentSimTimeNanos)
        scSim.ExecuteSimulation()

        # Real-time progress output (every ctrl_dt step)
        currentSec = currentSimTimeNanos * macros.NANO2SEC
        failures = getattr(attController, 'solve_failures', 0)
        n_solves = len(getattr(attController, 'wall_times', []))
        u_now = getattr(attController, '_u_curr', np.zeros(3))

        # Live state from navigation message
        navData = sNav.attOutMsg.read()
        sigma_BN = np.array(navData.sigma_BN)
        omega_BN = np.array(navData.omega_BN_B)
        s2 = np.dot(sigma_BN, sigma_BN)
        q0 = (1.0 - s2) / (1.0 + s2)
        q_vec = 2.0 * sigma_BN / (1.0 + s2)
        q_now = np.array([q0, *q_vec])
        q_now = q_now / (np.linalg.norm(q_now) + 1e-12)
        dot_q = np.clip(np.abs(np.dot(q_now, q_goal_np)), 0.0, 1.0)
        q_err_deg = float(np.degrees(2.0 * np.arccos(dot_q)))
        omega_norm = float(np.linalg.norm(omega_BN))

        print(
            f"\r  t={currentSec:6.1f}/{simEndTimeSec:.0f}s "
            f"err={q_err_deg:6.1f}\u00b0 |\u03c9|={omega_norm:.4f} "
            f"u=[{u_now[0]:+.3f},{u_now[1]:+.3f},{u_now[2]:+.3f}] "
            f"solves={n_solves} fail={failures}",
            end="", flush=True,
        )

    print()

    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "SIMULATION COMPLETED")

    # -------------------------------------------------------------------------
    #   12. Post-Processing and Data Export
    # -------------------------------------------------------------------------
    timeData = sNavAttLog.times() * macros.NANO2SEC
    sigmaBNData = sNavAttLog.sigma_BN
    omegaBNBData = sNavAttLog.omega_BN_B
    ctrlTorqueData = ctrlTorqueLog.torqueRequestBody

    # Drop warmup portion from logs and re-zero time
    warmup_time = float(getattr(attController, "warmup_time", 0.0))
    if warmup_time > 0.0:
        valid_mask = timeData >= warmup_time
        timeData = timeData[valid_mask] - warmup_time
        sigmaBNData = sigmaBNData[valid_mask]
        omegaBNBData = omegaBNBData[valid_mask]
        ctrlTorqueData = ctrlTorqueData[valid_mask]

    # Convert nav attitude MRP (B/N) to quaternion (B/N) and unwrap for continuity.
    # MRP2EP always gives q0 >= 0; unwrap ensures smooth trajectory for plotting.
    from sim.montecarlo.mc_utils import unwrap_quaternions
    sigma_sq_sum = np.sum(sigmaBNData**2, axis=1, keepdims=True)
    denom = 1.0 + sigma_sq_sum
    q0 = (1.0 - sigma_sq_sum) / denom
    q_vec = 2.0 * sigmaBNData / denom
    quatBNData = unwrap_quaternions(np.hstack([q0, q_vec]))
    
    # state = [q0, q1, q2, q3, wx, wy, wz]
    stateData = np.hstack([quatBNData, omegaBNBData])
    
    # Save data to MAT file
    simData = {'time': timeData, 'state': stateData, 'control': ctrlTorqueData}
    matFilePath = os.path.join(dataDir, "sim_data.mat")
    sio.savemat(matFilePath, simData)
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"MAT data saved to {matFilePath}")

    # Save data to NPZ file
    npzFilePath = os.path.join(dataDir, "sim_data.npz")
    npz_data = dict(time=timeData, state=stateData, control=ctrlTorqueData)
    if hasattr(attController, "get_v_plan_history"):
        vp = attController.get_v_plan_history()
        if len(vp) > 0:
            npz_data["v_plan_history"] = np.array(vp)  # (n_solve, nu, N)
    np.savez(npzFilePath, **npz_data)
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"NPZ data saved to {npzFilePath}")

    # Save data to CSV file (time + state + control torque)
    csvFilePath = os.path.join(dataDir, "sim_data.csv")
    timeCol = timeData.reshape(-1, 1)
    csvData = np.hstack([timeCol, stateData, ctrlTorqueData])
    csvHeader = "time,q0,q1,q2,q3,wx,wy,wz,ux,uy,uz"
    np.savetxt(csvFilePath, csvData, delimiter=",", header=csvHeader, comments="")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"CSV data saved to {csvFilePath}")

    # -------------------------------------------------------------------------
    #   13. Plotting
    # -------------------------------------------------------------------------
    # Target state for plotting: use goal quaternion and zero angular rate
    omega_goal = np.zeros(3, dtype=float)
    x_goal = np.concatenate([q_goal, omega_goal])

    # Build KOZ objects and boresight list from config
    koz_objects = KeepoutZone.from_cfg(cfg.constraint.koz_list)
    boresight_list = list(cfg.constraint.boresight_list)

    save_prefix = getattr(attController, "ModelTag", "Controller")

    plot_sim_results(
        x_traj=stateData,
        u_traj=ctrlTorqueData,
        x_goal=x_goal,
        koz_object=koz_objects,
        boresight_list=boresight_list,
        u_min=cfg.constraint.u_min,
        u_max=cfg.constraint.u_max,
        log_dt=cfg.sim.log_dt,
        figure_dir=figureDir,
        save_prefix=save_prefix,
    )

    # Optional: plot W history if controller exposes it
    if hasattr(attController, "get_W_history"):
        W_history = attController.get_W_history()
        if W_history is not None and np.size(W_history) > 0:
            s_history = (attController.get_s_history()
                         if hasattr(attController, "get_s_history") else None)
            plot_W_history(
                W_history=W_history,
                mpc_dt=attController.mpc_dt,
                figure_dir=figureDir,
                save_prefix=save_prefix,
                s_history=s_history,
            )

            # State-space uncertainty (requires z_history, ctx_history, and model)
            if (
                hasattr(attController, "get_z_history")
                and hasattr(attController, "get_ctx_history")
            ):
                z_history = attController.get_z_history()
                ctx_history = attController.get_ctx_history()
                if z_history is not None and len(z_history) > 0 and len(ctx_history) > 0:
                    plot_uncertainty_state_space(
                        W_history=W_history,
                        model=attController.model,
                        z_history=z_history,
                        ctx_history=ctx_history,
                        mpc_dt=attController.mpc_dt,
                        figure_dir=figureDir,
                        save_prefix=save_prefix,
                        inertia_change_time=inertia_transition.transition_start_sec,
                    )

    # Optional: plot latent diagnostics (z_curr vs z_goal + decoded state)
    if (
        hasattr(attController, "get_z_history")
        and hasattr(attController, "get_z_goal_history")
        and hasattr(attController, "get_ctx_history")
    ):
        z_history = attController.get_z_history()
        z_goal_history = attController.get_z_goal_history()
        ctx_history = attController.get_ctx_history()
        if len(z_history) > 0 and len(z_goal_history) > 0 and len(ctx_history) > 0:
            z_cost_Q = (
                attController.get_z_cost_Q_history()
                if hasattr(attController, "get_z_cost_Q_history")
                else None
            )
            plot_latent_diagnostics(
                z_history=z_history,
                z_goal_history=z_goal_history,
                model=attController.model,
                ctx_history=ctx_history,
                mpc_dt=attController.mpc_dt,
                figure_dir=figureDir,
                save_prefix=save_prefix,
                z_cost_Q_history=z_cost_Q,
            )

    # Optional: plot control history
    if hasattr(attController, "get_v_history"):
        v_hist = attController.get_v_history()
        if len(v_hist) > 0:
            plot_control_decomposition(
                Kz_history=np.zeros_like(v_hist),
                v_history=v_hist,
                dt=attController.ctrl_dt,
                u_min=cfg.constraint.u_min,
                u_max=cfg.constraint.u_max,
                figure_dir=figureDir,
                save_prefix=save_prefix,
            )

    # Optional: plot v_plan diagnostics
    if hasattr(attController, "get_v_plan_history"):
        vp_hist = attController.get_v_plan_history()
        if len(vp_hist) > 1:
            plot_v_plan_diagnostics(
                v_plan_history=vp_hist,
                mpc_dt=attController.mpc_dt,
                figure_dir=figureDir,
                save_prefix=save_prefix,
            )

    # -------------------------------------------------------------------------
    #   14. Result Summary
    # -------------------------------------------------------------------------
    if hasattr(attController, "get_wall_time_stats"):
        wall_stats = attController.get_wall_time_stats()
        bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"Controller wall-time stats: {format_time_stats_ms(wall_stats, decimals=2)}",
        )
    else:
        wall_stats = {}

    if hasattr(attController, "get_total_time_stats"):
        total_stats = attController.get_total_time_stats()
        bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            f"Controller total-time stats: {format_time_stats_ms(total_stats, decimals=2)}",
        )
    else:
        total_stats = {}
    if hasattr(attController, "get_profile_summary"):
        print(attController.get_profile_summary())

    # Collect result summary data
    sim_time_actual = float(timeData[-1]) if len(timeData) > 0 else 0.0
    q_init_list = np.array(q_init, dtype=float).flatten().tolist()
    q_goal_list = np.array(q_goal, dtype=float).flatten().tolist()
    
    boresight_list = []
    for bs_cfg in cfg.constraint.boresight_list:
        bs_vec = np.array(bs_cfg.vec, dtype=float).flatten()
        bs_normalized = (bs_vec / np.linalg.norm(bs_vec)).tolist()
        boresight_list.append({
            "name": bs_cfg.name,
            "vec": bs_normalized
        })
    
    koz_objects = KeepoutZone.from_cfg(cfg.constraint.koz_list)
    koz_list = []
    for koz in koz_objects:
        koz_list.append({
            "name": koz.name,
            "direction": koz.direction.tolist(),
            "angle_deg": float(koz.angle_deg)
        })
    
    torque_integral = np.trapz(ctrlTorqueData, timeData, axis=0).tolist()
    final_quat_bn = quatBNData[-1] if len(quatBNData) > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    final_quat_bn = final_quat_bn / np.linalg.norm(final_quat_bn)
    q_goal_np = np.array(q_goal, dtype=float)
    q_goal_np = q_goal_np / np.linalg.norm(q_goal_np)
    # Quaternion geodesic error angle [deg]
    dot_q = np.clip(np.abs(np.dot(final_quat_bn, q_goal_np)), 0.0, 1.0)
    final_att_error_deg = float(np.degrees(2.0 * np.arccos(dot_q)))
    final_omega = omegaBNBData[-1].tolist() if len(omegaBNBData) > 0 else [0.0, 0.0, 0.0]
    
    result_data = {
        "simulation_time": sim_time_actual,
        "initial_attitude": q_init_list,
        "target_attitude": q_goal_list,
        "inertia_change_applied": inertia_transition.is_complete,
        "inertia_change_time_sec": inertia_transition.transition_start_sec,
        "inertia_transition_type": inertia_transition.transition_type,
        "inertia_transition_duration_sec": inertia_transition.duration_sec,
        "pre_inertia": pre_inertia.tolist(),
        "post_inertia": post_inertia.tolist(),
        "boresight_vectors": boresight_list,
        "koz_list": koz_list,
        "torque_integral": torque_integral,
        "final_attitude_error_deg": final_att_error_deg,
        "final_angular_velocity": final_omega,
        "wall_times": wall_stats,
        "total_times": total_stats
    }
    
    result_path = os.path.join(dataDir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"Result summary saved to {result_path}")
    bskLogger.bskLog(bskLogging.BSK_INFORMATION, "RUN COMPLETED")


if __name__ == "__main__":
    run()