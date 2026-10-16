from typing import List
import numpy as np
from omegaconf import DictConfig

from sim.koz.koz import KeepoutZone

class QuadKOZ:
    def __init__(self, boresight_list: List[DictConfig], koz_list: List[DictConfig]):
        self.boresights = []
        self.boresight_names = []
        for bs_cfg in boresight_list:
            b = np.array(bs_cfg.vec, dtype=float).flatten()
            self.boresights.append(b / np.linalg.norm(b))
            self.boresight_names.append(bs_cfg.name)
        self.koz_objects = KeepoutZone.from_cfg(koz_list)
        self.num_bs = len(self.boresights)
        self.num_koz = len(self.koz_objects)

    @classmethod
    def from_cfg(cls, constraint_cfg: DictConfig) -> "QuadKOZ":
        return cls(
            boresight_list=constraint_cfg.boresight_list,
            koz_list=constraint_cfg.koz_list,
        )

    def build_koz_matrices(self) -> np.ndarray:
        """Build KOZ quadratic matrices for all (boresight, KOZ) pairs."""
        koz_matrices = []
        for boresight in self.boresights:
            for koz in self.koz_objects:
                M = self._build_koz_matrix(
                    angle_rad=koz.angle_rad, 
                    x=koz.direction, 
                    y=boresight
                )
                koz_matrices.append(M)
        
        return np.array(koz_matrices) if koz_matrices else np.empty((0, 4, 4))

    @staticmethod
    def _build_koz_matrix(angle_rad: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute KOZ quadratic constraint matrix.
        
        Args:
            angle_rad: Angle threshold between x and y [rad]
            x: Inertial frame unit vector (sun, moon, bright object, etc.)
            y: Body frame unit vector (boresight vector)
        Returns:
            M_tilde: 4x4 matrix satisfying
            q^T M_tilde q <= 2  <=>  x_N^T R(q) y_B <= cos(angle_rad)
            under the convention v_N = q ⊗ v_B ⊗ q*.
        References:
            [1] U. Lee and M. Mesbahi, "Feedback Control for Spacecraft Reorientation 
            Under Attitude Constraints Via Convex Potentials," IEEE TAES, 2014.

        NOTE:
        In [1], q is the N->B (passive) quaternion: v_N = q* ⊗ v_B ⊗ q,
        whereas Basilisk's MRP2EP returns the B->N (active, standard Hamilton)
        quaternion: v_N = q ⊗ v_B ⊗ q*.  Since q_Basilisk = conj(q_paper),
        the q0-qv cross-term sign (b = x × y) must be flipped.
        """
        x = np.array(x, dtype=float).flatten() 
        y = np.array(y, dtype=float).flatten() 

        c = np.cos(angle_rad)
        dot_xy = np.dot(x, y)
        
        m00 = dot_xy - c
        m0v = np.cross(y, x)   # The "NOTE" Point
        mvv = np.outer(x, y) + np.outer(y, x) - (dot_xy + c) * np.eye(3)
        
        M = np.zeros((4, 4))
        M[0, 0] = m00
        M[0, 1:] = m0v
        M[1:, 0] = m0v
        M[1:, 1:] = mvv

        M_tilde = M + 2 * np.eye(4)
        
        return M_tilde
