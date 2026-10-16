"""
Monte Carlo single-run module.

Provides module-level functions ``create_scenario()`` and ``execute_scenario()``
for use with Basilisk's ``MonteCarlo.Controller``.

Usage::

    from sim.montecarlo import single_run

    single_run.init_mc(cfg)                                    # inject Hydra config
    monteCarlo.setSimulationFunction(single_run.create_scenario)
    monteCarlo.setExecutionFunction(single_run.execute_scenario)
"""

import os
import logging
import numpy as np

from Basilisk import __path__
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
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeGravBody

from sim.utils.bsk_setup import set_sc, set_facets, normal2dcm_F0B
from sim.utils.math_utils import random_quat
from sim.montecarlo.mc_utils import random_inertia, perturbed_inertia, random_omega
from sim.controller.random import Random

bskPath = __path__[0]

# ---------------------------------------------------------------------------
#   Module-level global config (set by init_mc before MC runs)
# ---------------------------------------------------------------------------
MC_CFG = None


def inertia_to_vec6(I_3x3: np.ndarray) -> np.ndarray:
    """Convert 3x3 symmetric inertia matrix to 6 components.

    Order: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """
    return np.array(
        [
            I_3x3[0, 0],
            I_3x3[0, 1],
            I_3x3[0, 2],
            I_3x3[1, 1],
            I_3x3[1, 2],
            I_3x3[2, 2],
        ],
        dtype=float,
    )


def retain_inertia_timeseries(sim):
    """Retention callback: return time-aligned inertia (T, 7) with time column."""
    rec_att = sim.msgRecList["SimpleNavigation.attOutMsg"]
    times = rec_att.times()
    inertia_vec6 = np.asarray(sim.inertia_vec6, dtype=float).reshape(1, 6)
    inertia_ts = np.repeat(inertia_vec6, times.shape[0], axis=0)
    return {"inertia": unitTestSupport.addTimeColumn(times, inertia_ts)}


def init_mc(cfg):
    """Inject Hydra DictConfig into module-level global for MC workers.

    Must be called **before** ``monteCarlo.executeSimulations()``.
    On Linux (fork-based multiprocessing) the global is inherited by workers.
    """
    global MC_CFG
    MC_CFG = cfg


# ---------------------------------------------------------------------------
#   Scenario creation & execution (module-level functions for MC Controller)
# ---------------------------------------------------------------------------

def create_scenario():
    """Create and configure a Basilisk simulation scenario for one MC run.

    Called by ``MonteCarlo.Controller`` for each run.
    Uses ``MC_CFG`` (set via :func:`init_mc`).

    Randomised per run:
        - Inertia matrix  (``random_inertia``)
        - Initial attitude (``random_quat``)
        - Initial angular velocity (``random_omega``)

    Fixed per run (from config):
        - Mass
        - Orbit
        - Environment (drag / SRP / GG)
    """
    # Silence sim.* loggers in MC workers (main-process level may not apply)
    logging.getLogger("sim").setLevel(logging.WARNING)

    cfg = MC_CFG

    # --- Time settings -------------------------------------------------------
    simDtNanos = macros.sec2nano(cfg.sim.sim_dt)
    ctrlDtNanos = macros.sec2nano(cfg.sim.ctrl_dt)
    logDtNanos = macros.sec2nano(cfg.sim.log_dt)
    simTimeNanos = macros.sec2nano(cfg.sim.sim_time)

    # --- Simulation Base -----------------------------------------------------
    scSim = SimulationBaseClass.SimBaseClass()

    simProcessName = "simProcess"
    dynTaskName = "dynTask"
    ctrlTaskName = "ctrlTask"
    logTaskName = "logTask"

    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simDtNanos))
    dynProcess.addTask(scSim.CreateNewTask(ctrlTaskName, ctrlDtNanos))
    dynProcess.addTask(scSim.CreateNewTask(logTaskName, logDtNanos))

    # --- Gravity -------------------------------------------------------------
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
    spiceObject = gravFactory.createSpiceInterface(
        time=timeInitString, epochInMsg=True,
    )
    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(dynTaskName, gravFactory.spiceObject, 100)

    # --- Spacecraft (base from config, then randomise) -----------------------
    sc = set_sc(cfg.sc, mu)

    # Override inertia (mass stays fixed from config)
    mode = cfg.data_gen.inertia_mode
    if mode == "fixed":
        I_3x3 = np.array(cfg.data_gen.fixed_inertia).reshape(3, 3)
    elif mode == "random":
        I_3x3 = random_inertia(cfg.data_gen.inertia_range, cfg.data_gen.inertia_max_angle_deg)
    elif mode == "perturb":
        nominal = np.array(cfg.sc.inertia, dtype=np.float64).reshape(3, 3)
        I_3x3 = perturbed_inertia(
            nominal, cfg.data_gen.perturb_pct,
            off_diag_pct=cfg.data_gen.off_diag_pct,
        )
    else:
        raise ValueError(f"Unknown inertia_mode: {mode}")
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_3x3.flatten().tolist())

    # Override attitude with uniform random quaternion
    quat_init = random_quat()
    sc.hub.sigma_BNInit = rbk.EP2MRP(np.array(quat_init))

    # Override angular velocity with uniform random
    omega_init = random_omega(cfg.data_gen.omega_max)
    sc.hub.omega_BN_BInit = omega_init.tolist()

    scSim.AddModelToTask(dynTaskName, sc, 100)
    gravFactory.addBodiesTo(sc)
    scSim.scObject = sc

    # --- Environment ---------------------------------------------------------
    facets = set_facets(cfg.sc.geometry.hub, cfg.sc.geometry.panel)

    if cfg.env.enable_drag:
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        simSetPlanetEnvironment.exponentialAtmosphere(atmo, 'earth')
        atmo.addSpacecraftToModel(sc.scStateOutMsg)
        scSim.AddModelToTask(dynTaskName, atmo, 100)

        dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        dragEffector.ModelTag = "DragEffector"
        dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
        for facet in facets:
            dragEffector.addFacet(
                facet['area'], facet['coeff'],
                facet['normal'], facet['location'],
            )
        sc.addDynamicEffector(dragEffector)
        scSim.AddModelToTask(dynTaskName, dragEffector, 100)

    if cfg.env.enable_srp:
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])

        srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
        srpEffector.ModelTag = "SRPEffector"
        numFacets = 10 if cfg.sc.geometry.panel.use_panel else 6
        srpEffector.setNumFacets(numFacets)
        srpEffector.setNumArticulatedFacets(0)
        srpEffector.sunInMsg.subscribeTo(
            gravFactory.spiceObject.planetStateOutMsgs[1]
        )
        for facet in facets:
            dcm_F0B = normal2dcm_F0B(facet['normal'])
            srpEffector.addFacet(
                facet['area'], dcm_F0B, nHat_F, rotHat_F,
                facet['location'], facet['diff'], facet['spec'],
            )
        sc.addDynamicEffector(srpEffector)
        scSim.AddModelToTask(dynTaskName, srpEffector, 100)

    if cfg.env.enable_gg:
        ggEffector = GravityGradientEffector.GravityGradientEffector()
        ggEffector.ModelTag = "GravityGradientEffector"
        ggEffector.addPlanetName(earth.planetName)
        sc.addDynamicEffector(ggEffector)
        scSim.AddModelToTask(dynTaskName, ggEffector, 100)

    # --- Navigation ----------------------------------------------------------
    sNav = simpleNav.SimpleNav()
    sNav.ModelTag = "SimpleNavigation"
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

    if cfg.sc.sensors.use_nav_noise:
        nav = cfg.sc.sensors.nav_noise
        att_rad = float(nav.att) * macros.D2R      # deg -> rad (1-sigma)
        omega_rad = float(nav.omega) * macros.D2R  # deg/s -> rad/s (1-sigma)
        # PMatrix: 18x18 covariance matrix (nested list), sigma^2 on diagonal
        P = [[0.0] * 18 for _ in range(18)]
        for i in range(6, 9):    # att indices
            P[i][i] = att_rad ** 2
        for i in range(9, 12):   # rate indices
            P[i][i] = omega_rad ** 2
        sNav.PMatrix = P
        # walkBounds: 18x1 nested list, random walk bounds (>3-sigma)
        wb = [[0.0]] * 18
        for i in range(6, 9):
            wb[i] = [4 * att_rad]
        for i in range(9, 12):
            wb[i] = [4 * omega_rad]
        sNav.walkBounds = wb

    scSim.AddModelToTask(dynTaskName, sNav, 90)

    # --- Random Controller ---------------------------------------------------
    randomCtrl = Random(torque_max=float(cfg.data_gen.torque_max), seed=None)
    scSim.AddModelToTask(ctrlTaskName, randomCtrl)
    scSim.randomCtrl = randomCtrl

    # --- Actuator ------------------------------------------------------------
    ctrlFT = extForceTorque.ExtForceTorque()
    ctrlFT.ModelTag = "ControlForceTorque"
    sc.addDynamicEffector(ctrlFT)
    scSim.AddModelToTask(dynTaskName, ctrlFT, 80)
    ctrlFT.cmdTorqueInMsg.subscribeTo(randomCtrl.cmdTorqueOutMsg)

    # --- Logging -------------------------------------------------------------
    scSim.msgRecList = {}

    scSim.msgRecList["SimpleNavigation.attOutMsg"] = (sNav.attOutMsg.recorder(logDtNanos))
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["SimpleNavigation.attOutMsg"])

    scSim.msgRecList["randomCtrl.cmdTorqueOutMsg"] = (randomCtrl.cmdTorqueOutMsg.recorder(logDtNanos))
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["randomCtrl.cmdTorqueOutMsg"])

    # Store simulation parameters for execute_scenario
    scSim.simulationTime = simTimeNanos
    scSim.samplingTime = logDtNanos
    scSim.inertia_vec6 = inertia_to_vec6(I_3x3)

    return scSim


def execute_scenario(scSim):
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(scSim.simulationTime)
    scSim.ExecuteSimulation()


# ---------------------------------------------------------------------------
#   Retry helper (for NaN recovery outside MC Controller)
# ---------------------------------------------------------------------------

def run_single_retry():
    """Run a single scenario outside MC Controller for NaN retry.

    Returns data in the same dict format as ``monteCarlo.getRetainedData(i)``.
    """
    sim = create_scenario()
    execute_scenario(sim)

    messages = {}

    rec_att = sim.msgRecList["SimpleNavigation.attOutMsg"]
    times = rec_att.times()
    messages["SimpleNavigation.attOutMsg.sigma_BN"] = (
        unitTestSupport.addTimeColumn(times, rec_att.sigma_BN)
    )
    messages["SimpleNavigation.attOutMsg.omega_BN_B"] = (
        unitTestSupport.addTimeColumn(times, rec_att.omega_BN_B)
    )

    rec_trq = sim.msgRecList["randomCtrl.cmdTorqueOutMsg"]
    messages["randomCtrl.cmdTorqueOutMsg.torqueRequestBody"] = (
        unitTestSupport.addTimeColumn(rec_trq.times(), rec_trq.torqueRequestBody)
    )

    inertia_vec6 = np.asarray(sim.inertia_vec6, dtype=float).reshape(1, 6)
    inertia_ts = np.repeat(inertia_vec6, times.shape[0], axis=0)
    custom = {"inertia": unitTestSupport.addTimeColumn(times, inertia_ts)}

    return {"messages": messages, "custom": custom}
