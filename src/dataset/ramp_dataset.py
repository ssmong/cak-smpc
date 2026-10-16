"""
RAMP-Net dataset: Physics collocation points (P) + Data points (D).

Reference:
    Sanyal & Roy, "RAMP-Net: A Robust Adaptive MPC for Quadrotors via
    Physics-informed Neural Network", ICRA 2023, Section III-C, V-C.

Dataset P: Randomly sampled from state-action space (no labels needed).
Dataset D: Extracted from shared HDF5 simulation data (with next-state labels).
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch

log = logging.getLogger(__name__)


def _random_unit_quaternions(n: int) -> np.ndarray:
    """Generate n uniform random unit quaternions [q0, q1, q2, q3].

    Uses the subgroup algorithm (Shoemake, 1992) — same logic as
    sim/utils/math_utils.py:random_quats but returns [w,x,y,z] convention.
    """
    u = np.random.uniform(0, 1, (3, n))
    r1 = np.sqrt(1 - u[0])
    r2 = np.sqrt(u[0])
    t1 = 2 * np.pi * u[1]
    t2 = 2 * np.pi * u[2]
    return np.stack([
        r1 * np.sin(t1),  # q0
        r1 * np.cos(t1),  # q1
        r2 * np.sin(t2),  # q2
        r2 * np.cos(t2),  # q3
    ], axis=1)


class RampDataset:
    """Manages Physics collocation points (P) and Data points (D) for RAMP-Net.

    P: Randomly generated (x, u) pairs from the state-action space.
    D: (x_k, u_k, x_{k+1}) tuples extracted from the shared HDF5 dataset.
    """

    def __init__(
        self,
        name: str,
        data_root: str,
        num_seqs: int,
        sim_time: int,
        log_dt: float,
        mpc_dt: float,
        num_data_points: int,
        num_physics_points: int,
        omega_max: float,
        u_min: float,
        u_max: float,
        seed: int = 42,
        state_key: str = "state",
        control_key: str = "control",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        device: str = "cuda",
    ):
        self.mpc_dt = mpc_dt
        self.num_data_points = num_data_points
        self.num_physics_points = num_physics_points
        self.omega_max = omega_max
        self.u_min = u_min
        self.u_max = u_max
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        rng = np.random.RandomState(seed)

        # ---- Dataset D: load from HDF5 ----
        h5_path = Path(data_root) / f"{name}_{num_seqs}_{int(sim_time)}_{log_dt}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        pred_stride = max(1, int(round(mpc_dt / log_dt)))
        x_data_all, u_data_all, y_data_all = self._load_data_points(
            h5_path, state_key, control_key, pred_stride,
        )
        log.info(
            f"Loaded {len(x_data_all)} candidate data transitions from {h5_path} "
            f"(pred_stride={pred_stride})"
        )

        # Subsample |D| points
        n_available = len(x_data_all)
        n_total = min(num_data_points, n_available)
        idx = rng.choice(n_available, size=n_total, replace=False)
        x_data_all = x_data_all[idx]
        u_data_all = u_data_all[idx]
        y_data_all = y_data_all[idx]

        # Train/val/test split (matching Party's 3-way split pattern)
        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        perm = rng.permutation(n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        self.x_data_train = torch.from_numpy(x_data_all[train_idx]).float().to(self.device)
        self.u_data_train = torch.from_numpy(u_data_all[train_idx]).float().to(self.device)
        self.y_data_train = torch.from_numpy(y_data_all[train_idx]).float().to(self.device)

        self.x_data_val = torch.from_numpy(x_data_all[val_idx]).float().to(self.device)
        self.u_data_val = torch.from_numpy(u_data_all[val_idx]).float().to(self.device)
        self.y_data_val = torch.from_numpy(y_data_all[val_idx]).float().to(self.device)

        self.x_data_test = torch.from_numpy(x_data_all[test_idx]).float().to(self.device)
        self.u_data_test = torch.from_numpy(u_data_all[test_idx]).float().to(self.device)
        self.y_data_test = torch.from_numpy(y_data_all[test_idx]).float().to(self.device)

        log.info(f"Data points |D|: train={n_train}, val={n_val}, test={n_test}")

        # ---- Dataset P: randomly generate collocation points ----
        n_phys_test = max(1, int(num_physics_points * test_ratio))
        n_phys_val = max(1, int(num_physics_points * val_ratio))
        n_phys_train = num_physics_points - n_phys_val - n_phys_test

        self.x_phys_train = self._generate_collocation(n_phys_train, rng).to(self.device)
        self.u_phys_train = self._generate_control(n_phys_train, rng).to(self.device)

        self.x_phys_val = self._generate_collocation(n_phys_val, rng).to(self.device)
        self.u_phys_val = self._generate_control(n_phys_val, rng).to(self.device)

        self.x_phys_test = self._generate_collocation(n_phys_test, rng).to(self.device)
        self.u_phys_test = self._generate_control(n_phys_test, rng).to(self.device)

        log.info(f"Physics points |P|: train={n_phys_train}, val={n_phys_val}, test={n_phys_test}")

    def _load_data_points(
        self,
        h5_path: Path,
        state_key: str,
        control_key: str,
        pred_stride: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (x_k, u_k, x_{k+1}) transitions from all sequences in HDF5."""
        x_list, u_list, y_list = [], [], []

        with h5py.File(h5_path, "r") as hf:
            ts = hf["timeseries"]
            for k in sorted(ts.keys(), key=lambda name: int(name.split("_")[-1])):
                grp = ts[k]
                states = grp[state_key][:]    # (T, 7)
                controls = grp[control_key][:]  # (T, 3)

                T = states.shape[0]
                for i in range(0, T - pred_stride, pred_stride):
                    x_list.append(states[i])
                    u_list.append(controls[i])
                    y_list.append(states[i + pred_stride])

        return np.array(x_list), np.array(u_list), np.array(y_list)

    def _generate_collocation(self, n: int, rng: np.random.RandomState) -> torch.Tensor:
        """Generate n random state collocation points."""
        q = _random_unit_quaternions(n)  # (n, 4)
        omega = rng.uniform(-self.omega_max, self.omega_max, (n, 3))  # (n, 3)
        x = np.hstack([q, omega])  # (n, 7)
        return torch.from_numpy(x).float()

    def _generate_control(self, n: int, rng: np.random.RandomState) -> torch.Tensor:
        """Generate n random control collocation points."""
        u = rng.uniform(self.u_min, self.u_max, (n, 3))
        return torch.from_numpy(u).float()

    def get_train_batch(self) -> dict[str, torch.Tensor]:
        """Return full training batch (paper uses entire dataset as single batch)."""
        return {
            "x_phys": self.x_phys_train,
            "u_phys": self.u_phys_train,
            "x_data": self.x_data_train,
            "u_data": self.u_data_train,
            "y_data": self.y_data_train,
        }

    def get_val_batch(self) -> dict[str, torch.Tensor]:
        """Return full validation batch."""
        return {
            "x_phys": self.x_phys_val,
            "u_phys": self.u_phys_val,
            "x_data": self.x_data_val,
            "u_data": self.u_data_val,
            "y_data": self.y_data_val,
        }

    def get_test_batch(self) -> dict[str, torch.Tensor]:
        """Return full test batch."""
        return {
            "x_phys": self.x_phys_test,
            "u_phys": self.u_phys_test,
            "x_data": self.x_data_test,
            "u_data": self.u_data_test,
            "y_data": self.y_data_test,
        }
