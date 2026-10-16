import torch
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset

from src.dataset.processor import DataProcessor
from src.dataset.collator import WindowCollator

log = logging.getLogger(__name__)


class PartyDataset(Dataset):
    """Loads and normalizes full sequences. Window slicing via get_collate_fn()."""
    
    def __init__(
        self, 
        name: str, data_root: str,
        state_dim: int, control_dim: int,
        state_key: str, control_key: str,
        num_seqs: int, sim_time: int,
        log_dt: float, hist_dt: float, pred_dt: float,
        hist_len: int,
        sample_stride: int = 1,
        quat_indices: List[int] = [0, 1, 2, 3],
        split: str = "train", seed: int = 42,
        train_ratio: float = 0.8, val_ratio: float = 0.1,
        use_norm: bool = True,
        stats_dir: str = "data/stats",
        use_loaded_stats: bool = False,
        storage_device: str = "cpu",
        load_inertia: bool = True,
        shared_data: Optional[Dict[str, Optional[List[torch.Tensor]]]] = None,
        shared_split_indices: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.name = name
        self.data_root = Path(data_root)
        self.x_dim, self.u_dim = state_dim, control_dim
        self.x_key, self.u_key = state_key, control_key
        self.inertia_key = "inertia"
        self.load_inertia = bool(load_inertia)
        self.num_seqs, self.sim_time, self.log_dt = num_seqs, sim_time, log_dt
        self.hist_dt, self.pred_dt, self.hist_len = hist_dt, pred_dt, hist_len
        self.sample_stride = sample_stride
        self.split = split.lower()
        self.log_dataset_info = self.split == "train"
        self.train_ratio, self.val_ratio, self.seed = train_ratio, val_ratio, seed
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.storage_device = self._resolve_storage_device(storage_device)
        
        self.processor = DataProcessor(state_dim, control_dim, quat_indices).to(self.storage_device)
        
        # Load and split
        if shared_data is None:
            self.x_traj_list, self.u_traj_list, self.i_traj_list = self._load_data()
            self._unwrap_quaternions()
            self.split_indices = self._build_splits(len(self.x_traj_list))
        else:
            if shared_split_indices is None:
                raise ValueError("shared_split_indices must be provided when shared_data is used.")
            if "x_traj_list" not in shared_data or "u_traj_list" not in shared_data:
                raise KeyError("shared_data must contain 'x_traj_list' and 'u_traj_list'.")
            self.x_traj_list = shared_data["x_traj_list"]
            self.u_traj_list = shared_data["u_traj_list"]
            self.i_traj_list = shared_data.get("i_traj_list")
            self.split_indices = shared_split_indices

            if self.x_traj_list is None or self.u_traj_list is None:
                raise ValueError("shared_data trajectories must not be None.")
            if len(self.x_traj_list) != len(self.u_traj_list):
                raise ValueError("shared_data x/u trajectory list lengths must match.")

        self.sequence_indices = self.split_indices[self.split]
        self.indices = self.sequence_indices
        self.use_window_indices = False
        if self.log_dataset_info:
            log.info(
                f"Split sizes - train: {len(self.split_indices['train'])}, "
                f"val: {len(self.split_indices['val'])}, "
                f"test: {len(self.split_indices['test'])}"
            )
            log.info(f"Dataset tensor storage device: {self.storage_device}")
        
        # Normalize
        if use_norm:
            stats_path = self.stats_dir / f"{name}_stats.json"
            if self.split == "train" and not use_loaded_stats:
                stats = self._compute_stats(self.split_indices["train"])
                self.processor.set_stats(stats)
                self.processor.save_stats(stats_path)
            else:
                self.processor.load_stats(stats_path)
            self._normalize_in_place()

    def _resolve_storage_device(self, storage_device: str) -> torch.device:
        try:
            device = torch.device(storage_device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid storage_device='{storage_device}'. Use 'cpu' or 'cuda[:index]'.") from exc

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("storage_device is CUDA but CUDA is not available.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"storage_device points to cuda:{device.index}, "
                    f"but only {torch.cuda.device_count()} CUDA device(s) are available."
                )
        elif device.type != "cpu":
            raise ValueError(f"Unsupported storage_device type '{device.type}'. Use 'cpu' or 'cuda[:index]'.")

        return device

    @property
    def state_dim(self) -> int:
        return self.x_dim
    
    @property
    def control_dim(self) -> int:
        return self.u_dim

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) tensors for denormalization."""
        return self.processor.mean, self.processor.std

    def _load_data(self):
        num_seqs = int(self.num_seqs)
        sim_time = int(self.sim_time)
        path = self.data_root / f"{self.name}_{num_seqs}_{sim_time}_{self.log_dt}.h5"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if self.log_dataset_info:
            log.info(f"Loading data from {path}")
        x_traj_list, u_traj_list = [], []
        i_traj_list = [] if self.load_inertia else None
        
        with h5py.File(path, 'r') as hf:
            ts = hf['timeseries']
            for k in sorted(ts.keys(), key=lambda name: int(name.split("_")[-1])):
                grp = ts[k]
                x_traj = torch.from_numpy(grp[self.x_key][:]).to(self.storage_device, dtype=torch.float32)
                u_traj = torch.from_numpy(grp[self.u_key][:]).to(self.storage_device, dtype=torch.float32)
                if self.load_inertia:
                    if self.inertia_key not in grp:
                        raise KeyError(f"Missing required key '{self.inertia_key}' in {k}")
                    i_traj = torch.from_numpy(grp[self.inertia_key][:]).to(self.storage_device, dtype=torch.float32)
                    if i_traj.shape[0] != x_traj.shape[0]:
                        raise ValueError(
                            f"{k}: inertia length {i_traj.shape[0]} does not match state length {x_traj.shape[0]}"
                        )

                x_traj_list.append(x_traj)
                u_traj_list.append(u_traj)
                if self.load_inertia and i_traj_list is not None:
                    i_traj_list.append(i_traj)
        
        if self.log_dataset_info:
            log.info(f"Loaded {len(x_traj_list)} trajectories")
            log.info(f"Inertia trajectories loaded: {self.load_inertia}")
        return x_traj_list, u_traj_list, i_traj_list

    def _unwrap_quaternions(self) -> None:
        """Unwrap all quaternion trajectories for global sign consistency.

        For each trajectory:
          1. Align q[0] to identity hemisphere (q_w >= 0)
          2. Sequentially unwrap for within-trajectory continuity
        Vectorized via cumulative sign product.
        """
        q_ref = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.storage_device)
        n_flipped = 0

        for x_traj in self.x_traj_list:
            q = x_traj[:, :4]  # (T, 4) view — in-place modification
            s0 = 1.0 if (q[0] @ q_ref).item() >= 0 else -1.0
            if s0 < 0:
                n_flipped += 1

            if len(q) > 1:
                dots = (q[1:] * q[:-1]).sum(-1)
                signs = torch.where(dots >= 0, 1.0, -1.0)
                cum_signs = torch.cumprod(signs, dim=0)
                q[1:] *= (s0 * cum_signs).unsqueeze(-1)

            q[0] *= s0

        if self.log_dataset_info:
            log.info(f"Quaternion unwrap: {n_flipped}/{len(self.x_traj_list)} trajectories had start flipped to q_w>=0")

    def _build_splits(self, n: int) -> Dict[str, List[int]]:
        idx = np.arange(n)
        np.random.RandomState(self.seed).shuffle(idx)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        return {
            "train": idx[:n_train].tolist(),
            "val": idx[n_train:n_train + n_val].tolist(),
            "test": idx[n_train + n_val:].tolist(),
        }

    def _compute_stats(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        x = torch.cat([self.x_traj_list[i] for i in indices])
        u = torch.cat([self.u_traj_list[i] for i in indices])
        return {
            'mean': x.mean(0), 'std': x.std(0), 'min': x.min(0)[0], 'max': x.max(0)[0],
            'u_mean': u.mean(0), 'u_std': u.std(0), 'u_min': u.min(0)[0], 'u_max': u.max(0)[0],
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.use_window_indices:
            i, start_idx = self.indices[idx]
        else:
            i = self.indices[idx]
            start_idx = self.hist_len * int(round(self.hist_dt / self.log_dt))

        return {
            'x_traj': self.x_traj_list[i],
            'u_traj': self.u_traj_list[i],
            'i_traj': None if self.i_traj_list is None else self.i_traj_list[i],
            'start_idx': start_idx,
        }

    def _normalize_in_place(self) -> None:
        """Normalize trajectories in-place to avoid duplicate full-sized tensors."""
        state_keep_mask = self.processor.skip_norm_mask
        state_norm_indices = torch.nonzero(~state_keep_mask, as_tuple=False).flatten().tolist()

        mean = self.processor.mean
        std = self.processor.std
        u_mean = self.processor.u_mean
        u_std = self.processor.u_std

        if len(state_norm_indices) > 0:
            for x in self.x_traj_list:
                for d in state_norm_indices:
                    x[:, d].sub_(mean[d]).div_(std[d])
        for u in self.u_traj_list:
            u.sub_(u_mean).div_(u_std)

    def get_shared_data(self) -> Dict[str, Optional[List[torch.Tensor]]]:
        """Expose trajectory storage for split datasets without duplicating memory."""
        return {
            "x_traj_list": self.x_traj_list,
            "u_traj_list": self.u_traj_list,
            "i_traj_list": self.i_traj_list,
        }

    def _build_window_indices(self, rollout_for_index: int) -> None:
        hist_stride = int(round(self.hist_dt / self.log_dt))
        pred_stride = int(round(self.pred_dt / self.log_dt))
        min_idx = self.hist_len * hist_stride
        sample_stride = max(1, int(self.sample_stride))
        window_step = sample_stride * pred_stride

        window_indices = []
        for seq_idx in self.sequence_indices:
            traj_len = self.x_traj_list[seq_idx].shape[0]
            max_idx = traj_len - 1 - rollout_for_index * pred_stride
            if max_idx < min_idx:
                raise ValueError(
                    f"Trajectory too short ({traj_len}) for hist_len={self.hist_len}, rollout={rollout_for_index}"
                )
            for k in range(min_idx, max_idx + 1, window_step):
                window_indices.append((seq_idx, k))

        self.indices = window_indices
        self.use_window_indices = True

        if self.log_dataset_info:
            log.info(
                f"Expanded split='{self.split}' windows: {len(self.sequence_indices)} sequences -> {len(self.indices)} samples "
                f"(rollout={rollout_for_index}, sample_stride={sample_stride}, window_step={window_step})"
            )

    def get_collate_fn(
        self,
        rollout_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> WindowCollator:
        """Returns a collate function for DataLoader."""
        if rollout_steps is not None:
            rollout_for_index = int(rollout_steps)
        elif max_steps is not None:
            rollout_for_index = int(max_steps)
        else:
            raise ValueError("rollout_steps or max_steps must be provided to build window indices")

        # Build deterministic (sequence, start_idx) samples so __len__ reflects expanded sample count.
        self._build_window_indices(rollout_for_index)

        return WindowCollator(
            log_dt=self.log_dt,
            hist_dt=self.hist_dt,
            hist_len=self.hist_len,
            pred_dt=self.pred_dt,
            rollout_steps=rollout_steps,
            min_steps=min_steps,
            max_steps=max_steps,
        )
