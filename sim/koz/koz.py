import numpy as np
from typing import List
from omegaconf import DictConfig

class KeepoutZone:
    def __init__(self, direction: list, angle_deg: float, name: str = "KOZ"):
        d = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.angle_deg = angle_deg
        self.angle_rad = np.deg2rad(angle_deg)
        self.name = name

    def __repr__(self):
        d = [round(x, 4) for x in self.direction]
        return f"KOZ {self.name}: direction={d}, angle_deg={self.angle_deg:.1f}"
    
    @classmethod
    def from_cfg(cls, koz_list: List[DictConfig]) -> List['KeepoutZone']:
        return [cls(koz.direction, koz.angle_deg, koz.name) for koz in koz_list]