import logging
import numpy as np
from omegaconf import DictConfig

from Basilisk.simulation import spacecraft
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import macros
from Basilisk.utilities import RigidBodyKinematics as rbk

log = logging.getLogger(__name__)

def set_sc(sc_cfg: DictConfig, mu: float):
    # Frame Definition
    #   N: Inertial Frame
    #   B: Body Frame

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = sc_cfg.tag
    
    scObject.hub.mHub = sc_cfg.mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(sc_cfg.inertia)

    # COE initialization
    orbit_cfg = sc_cfg.orbit
    oe = orbitalMotion.ClassicElements()
    oe.a = (orbit_cfg.R_E + orbit_cfg.a) * 1000  # km -> m
    oe.e = orbit_cfg.e
    oe.i = orbit_cfg.i * macros.D2R
    oe.Omega = orbit_cfg.Omega * macros.D2R
    oe.omega = orbit_cfg.omega * macros.D2R
    oe.f = orbit_cfg.f * macros.D2R

    rN, vN = orbitalMotion.elem2rv(mu, oe)

    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # Attitude Initialization
    att_cfg = sc_cfg.attitude
    quat_init = att_cfg.quat_init
    scObject.hub.sigma_BNInit = rbk.EP2MRP(np.array(quat_init))
    scObject.hub.omega_BN_BInit = list(att_cfg.omega_init)

    log.info(f"[{scObject.ModelTag}] Initial attitude: {quat_init}")
    log.info(f"[{scObject.ModelTag}] Initial angular velocity: {att_cfg.omega_init}")

    return scObject


def set_facets(hub_cfg, panel_cfg):
    """
    Generate facet information based on hub and panel configurations.
    Used for both Drag and SRP effectors.
    
    Args:
        hub_cfg: Configuration object for hub (length, width, height, coeff, diff, spec)
        panel_cfg: Configuration object for panel (width, length, coeff, diff, spec, offset)
        
    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - area (float)
            - coeff (float) - drag coefficient
            - diff (float) - SRP diffuse coefficient
            - spec (float) - SRP specular coefficient
            - normal (np.array)
            - location (np.array)
    """
    facets = []
    
    # Hub Parameters
    l = hub_cfg.length
    w = hub_cfg.width
    h = hub_cfg.height
    c_hub = hub_cfg.coeff
    diff_hub = hub_cfg.diff
    spec_hub = hub_cfg.spec
    
    # Define Hub Facets (6 faces of the box)
    # Area calculation: 
    # x-face: w * h
    # y-face: l * h
    # z-face: l * w
    
    hub_facets_data = [
        # +x face
        {'norm': [1, 0, 0],  'area': w*h, 'loc': [l/2.0, 0, 0]},
        # -x face
        {'norm': [-1, 0, 0], 'area': w*h, 'loc': [-l/2.0, 0, 0]},
        # +y face
        {'norm': [0, 1, 0],  'area': l*h, 'loc': [0, w/2.0, 0]},
        # -y face
        {'norm': [0, -1, 0], 'area': l*h, 'loc': [0, -w/2.0, 0]},
        # +z face
        {'norm': [0, 0, 1],  'area': l*w, 'loc': [0, 0, h/2.0]},
        # -z face
        {'norm': [0, 0, -1], 'area': l*w, 'loc': [0, 0, -h/2.0]},
    ]
    
    for face in hub_facets_data:
        facets.append({
            'area': face['area'],
            'coeff': c_hub,
            'diff': diff_hub,
            'spec': spec_hub,
            'normal': np.array(face['norm']),
            'location': np.array(face['loc'])
        })
        
    # Panel Parameters
    if panel_cfg.use_panel:
        # Calculate panel area from width and length
        p_area = panel_cfg.width * panel_cfg.length
        p_coeff = panel_cfg.coeff
        p_diff = panel_cfg.diff
        p_spec = panel_cfg.spec
        p_offset = panel_cfg.offset
        
        # Define Panel Facets (2 wings along Y-axis, +/- Z facing)
        panel_facets_data = [
            # +Y wing (Positive Y offset)
            # +Z face
            {'norm': [0, 0, 1],  'loc': [0, p_offset, 0]},
            # -Z face
            {'norm': [0, 0, -1], 'loc': [0, p_offset, 0]},
            
            # -Y wing (Negative Y offset)
            # +Z face
            {'norm': [0, 0, 1],  'loc': [0, -p_offset, 0]},
            # -Z face
            {'norm': [0, 0, -1], 'loc': [0, -p_offset, 0]},
        ]
        
        for p in panel_facets_data:
            facets.append({
                'area': p_area,
                'coeff': p_coeff,
                'diff': p_diff,
                'spec': p_spec,
                'normal': np.array(p['norm']),
                'location': np.array(p['loc'])
            })
            
    return facets

def normal2dcm_F0B(nHat_B):
    nHat_B = np.array(nHat_B) / np.linalg.norm(nHat_B)
    if np.allclose(nHat_B, [0, 1, 0]):
        return np.eye(3)
    elif np.allclose(nHat_B, [0, -1, 0]):
        return rbk.PRV2C(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        y_axis = nHat_B
        temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x_axis = temp - np.dot(temp, y_axis) * y_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis) / np.linalg.norm(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis]).T