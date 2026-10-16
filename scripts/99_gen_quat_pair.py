"""
KOZ 제약 조건을 만족하는 q_init과 q_goal 쿼터니언 쌍을 생성하는 스크립트

q_init과 q_goal은 120도 이상 차이나도록 생성됩니다.
"""

import logging
import numpy as np
import hydra
from omegaconf import DictConfig
from typing import List

from sim.koz.koz import KeepoutZone
from sim.koz.koz_utils import generate_quat_pair
from sim.utils.math_utils import quat_error_rad

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="gen_quat_pair", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    np.random.seed(cfg.seed)
    log.info(f"Random seed: {cfg.seed}")

    boresight_list = list(cfg.constraint.boresight_list)
    koz_objects = KeepoutZone.from_cfg(cfg.constraint.koz_list)
    margin_cap_deg = cfg.margin_cap_deg

    log.info("Boresight list:")
    for bs in boresight_list:
        vec = [round(float(x), 4) for x in bs.vec]
        log.info(f"  - {bs.name}: {vec}")

    log.info("KOZ list:")
    for koz in koz_objects:
        log.info(f"  - {koz}")

    q_init, q_goal = generate_quat_pair(
        boresight_list=boresight_list,
        koz_object=koz_objects,
        min_angle_deg=cfg.min_angle_deg,
        max_angle_deg=cfg.max_angle_deg,
        max_attempts=cfg.max_attempts,
        margin_cap_deg=margin_cap_deg,
        safety_margin_deg_init=cfg.safety_margin_deg_init,
        safety_margin_deg_goal=cfg.safety_margin_deg_goal,
        koz_proximity_deg_init=cfg.koz_proximity_deg_init,
        koz_proximity_deg_goal=cfg.koz_proximity_deg_goal,
        slerp_min_gap_deg=cfg.slerp_min_gap_deg,
        slerp_max_gap_deg=cfg.slerp_max_gap_deg,
        slerp_n_samples=cfg.slerp_n_samples,
    )
    
    angle_diff_rad = quat_error_rad(q_init, q_goal)
    angle_diff_deg = np.rad2deg(angle_diff_rad)
    
    log.info("="*60)
    log.info("Generated quaternion pair:")
    log.info(f"q_init = {[round(x, 4) for x in q_init.tolist()]}")
    log.info(f"q_goal = {[round(x, 4) for x in q_goal.tolist()]}")
    log.info(f"Angle difference: {angle_diff_deg:.2f} degrees")
    log.info("="*60)


if __name__ == "__main__":
    main()
