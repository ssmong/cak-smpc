import os
import logging
import shutil
import numpy as np
import h5py
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from Basilisk.architecture import bskLogging
from Basilisk.utilities import macros
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy

from sim.montecarlo import single_run
from sim.montecarlo.mc_utils import mrp2quat_batch, is_data_valid, unwrap_quaternions

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="gen_data", version_base=None)
def run(cfg: DictConfig):
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # --- Paths ---------------------------------------------------------------
    data_root = to_absolute_path(cfg.data_gen.data_root)
    os.makedirs(data_root, exist_ok=True)

    # Archive dir for MC Controller (temporary, deleted after H5 save)
    archive_dir = os.path.join(
        to_absolute_path("data/experiments"),
        "data_gen",
        datetime.now().strftime("%Y-%m-%d/%H-%M-%S"),
    )
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    # H5 filename: {name}_{num_seqs}_{sim_time}_{log_dt}.h5
    name = cfg.data_gen.name
    num_seqs = int(cfg.data_gen.num_seqs)
    sim_time = int(cfg.sim.sim_time)
    log_dt = float(cfg.sim.log_dt)

    h5_filename = f"{name}_{num_seqs}_{sim_time}_{log_dt}.h5"
    h5_path = os.path.join(data_root, h5_filename)

    # --- Inject config into single_run module --------------------------------
    single_run.init_mc(cfg)

    # --- Configure MonteCarlo Controller -------------------------------------
    monteCarlo = Controller()
    monteCarlo.setSimulationFunction(single_run.create_scenario)
    monteCarlo.setExecutionFunction(single_run.execute_scenario)
    monteCarlo.setExecutionCount(num_seqs)
    monteCarlo.setShouldDisperseSeeds(True)
    monteCarlo.setThreadCount(int(cfg.data_gen.num_threads))
    monteCarlo.setVerbose(False)
    monteCarlo.setShowProgressBar(True)
    monteCarlo.setArchiveDir(archive_dir)

    # --- Retention Policy ----------------------------------------------------
    retentionPolicy = RetentionPolicy()
    retentionPolicy.logRate = macros.sec2nano(log_dt)
    retentionPolicy.addMessageLog(
        "SimpleNavigation.attOutMsg", ["sigma_BN", "omega_BN_B"]
    )
    retentionPolicy.addMessageLog(
        "randomCtrl.cmdTorqueOutMsg", ["torqueRequestBody"]
    )
    retentionPolicy.addRetentionFunction(single_run.retain_inertia_timeseries)
    monteCarlo.addRetentionPolicy(retentionPolicy)

    # --- Execute simulations -------------------------------------------------
    log.info(
        f"Starting Monte Carlo data generation: {num_seqs} runs, "
        f"sim_time={sim_time}s, log_dt={log_dt}s, "
        f"threads={cfg.data_gen.num_threads}"
    )
    failures = monteCarlo.executeSimulations()
    if failures:
        log.error(f"Failed runs reported by MC controller: {failures}")

    # --- Save to HDF5 (PartyDataset-compatible) ------------------------------
    #
    # File layout:
    #   timeseries/
    #     sequence_0/
    #       state       (T, 7) float32  [q0, q1, q2, q3, wx, wy, wz]
    #       control     (T, 3) float32  [ux, uy, uz]
    #       inertia     (T, 6) float32  [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    #     sequence_1/
    #       ...
    #   metadata/
    #     (attributes)
    #
    log.info(f"Saving data to {h5_path} ...")
    saved_count = 0
    inertia_list = []  # 통계 계산용 (각 시퀀스의 관성모멘트)

    with h5py.File(h5_path, 'w') as f:
        grp_ts = f.create_group("timeseries")

        for i in range(num_seqs):
            data = monteCarlo.getRetainedData(i)

            # NaN retry
            retry_count = 0
            while (not is_data_valid(data)) and retry_count < 20:
                log.warning(
                    f"WARNING: invalid data in run {i}, "
                    f"retrying ({retry_count + 1}/20)"
                )
                data = single_run.run_single_retry()
                retry_count += 1

            if not is_data_valid(data):
                log.error(f"Run {i} invalid after {retry_count} retries, skipping.")
                continue

            # Extract data (strip time column at index 0)
            msgs = data["messages"]
            sigma_mrp = msgs["SimpleNavigation.attOutMsg.sigma_BN"][:, 1:]
            omega = msgs["SimpleNavigation.attOutMsg.omega_BN_B"][:, 1:]
            torque = data["messages"]["randomCtrl.cmdTorqueOutMsg.torqueRequestBody"][:, 1:]
            inertia = data["custom"]["inertia"][:, 1:]

            quaternion = mrp2quat_batch(sigma_mrp)
            quaternion = unwrap_quaternions(quaternion)
            state = np.hstack([quaternion, omega])

            # Align torque length with state
            if torque.shape[0] < state.shape[0]:
                pad = np.zeros((state.shape[0] - torque.shape[0], 3))
                torque = np.vstack([torque, pad])
            elif torque.shape[0] > state.shape[0]:
                torque = torque[: state.shape[0]]

            if inertia.shape[0] != state.shape[0]:
                raise ValueError(
                    f"Run {i}: inertia length {inertia.shape[0]} does not match state length {state.shape[0]}"
                )

            grp_seq = grp_ts.create_group(f"sequence_{i}")
            grp_seq.create_dataset(cfg.data_gen.state_key, data=state, dtype=np.float32)
            grp_seq.create_dataset(cfg.data_gen.control_key, data=torque, dtype=np.float32)
            grp_seq.create_dataset("inertia", data=inertia, dtype=np.float32)
            saved_count += 1

            # 관성모멘트 수집 (첫 번째 timestep, 시퀀스 내에서 동일)
            inertia_list.append(inertia[0])

        # --- Metadata 저장 -------------------------------------------------------
        grp_meta = f.create_group("metadata")
        
        # 공통 정보
        inertia_mode = cfg.data_gen.inertia_mode
        grp_meta.attrs['inertia_mode'] = inertia_mode
        grp_meta.attrs['num_seqs'] = saved_count
        grp_meta.attrs['sim_dt'] = float(cfg.sim.sim_dt)
        grp_meta.attrs['ctrl_dt'] = float(cfg.sim.ctrl_dt)
        grp_meta.attrs['log_dt'] = float(cfg.sim.log_dt)
        grp_meta.attrs['u_min'] = -float(cfg.data_gen.torque_max)
        grp_meta.attrs['u_max'] = float(cfg.data_gen.torque_max)

        if inertia_mode == "fixed":
            grp_meta.attrs['fixed_inertia'] = list(cfg.data_gen.fixed_inertia)
        elif inertia_mode == "perturb":
            grp_meta.attrs['nominal_inertia'] = list(cfg.sc.inertia)
            grp_meta.attrs['perturb_pct'] = float(cfg.data_gen.perturb_pct)
            grp_meta.attrs['off_diag_pct'] = float(cfg.data_gen.off_diag_pct)

        if inertia_mode in ("random", "perturb"):
            if inertia_mode == "random":
                grp_meta.attrs['inertia_range'] = list(cfg.data_gen.inertia_range)
                grp_meta.attrs['inertia_max_angle_deg'] = float(cfg.data_gen.inertia_max_angle_deg)

            # 관성모멘트 통계 계산 (대각/비대각 분리)
            # inertia vec6 = [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
            all_inertia = np.array(inertia_list)  # (num_seqs, 6)

            # 대각 요소: indices 0, 3, 5 (Ixx, Iyy, Izz)
            diag_inertia = all_inertia[:, [0, 3, 5]]
            grp_meta.attrs['inertia_diag_mean'] = diag_inertia.mean(axis=0)
            grp_meta.attrs['inertia_diag_std'] = diag_inertia.std(axis=0)
            grp_meta.attrs['inertia_diag_min'] = diag_inertia.min(axis=0)
            grp_meta.attrs['inertia_diag_max'] = diag_inertia.max(axis=0)

            # 비대각 요소: indices 1, 2, 4 (Ixy, Ixz, Iyz)
            offdiag_inertia = all_inertia[:, [1, 2, 4]]
            grp_meta.attrs['inertia_offdiag_mean'] = offdiag_inertia.mean(axis=0)
            grp_meta.attrs['inertia_offdiag_std'] = offdiag_inertia.std(axis=0)
            grp_meta.attrs['inertia_offdiag_min'] = offdiag_inertia.min(axis=0)
            grp_meta.attrs['inertia_offdiag_max'] = offdiag_inertia.max(axis=0)

    log.info(f"Data generation complete: {saved_count}/{num_seqs} runs → {h5_path}")

    # Ensure the MC DataWriter process is fully finished before deleting archive.
    data_writer = getattr(monteCarlo, "dataWriter", None)
    if data_writer is not None and data_writer.pid is not None:
        if data_writer.is_alive():
            log.info("Waiting for MonteCarlo DataWriter process to finish before cleanup...")
        data_writer.join()
        if data_writer.exitcode not in (0, None):
            raise RuntimeError(
                f"MonteCarlo DataWriter exited with non-zero code: {data_writer.exitcode}"
            )

    # Cleanup MC archive 
    if os.path.exists(archive_dir):
        shutil.rmtree(archive_dir)
        log.info(f"Cleaned up archive directory: {archive_dir}")
    else:
        log.warning(f"Archive directory already absent at cleanup time: {archive_dir}")


if __name__ == "__main__":
    run()
