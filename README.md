# Context-Aware Koopman Stochastic MPC for Spacecraft Attitude Control with Time-Varying Inertia

Yeonseo Lee, Minhyung Kim, Seongjae Kim, Seungmin Yang, Hyeongjun Park · *Aerospace Science and Technology*, 2026 · [**Paper**](https://doi.org/10.1016/j.ast.2026.112960)

CAK-SMPC steers a spacecraft with time-varying inertia while keeping sensor boresights out of keep-out zones (KOZs). A Mamba context encoder conditions a Koopman VAE whose linear latent dynamics turn KOZ chance constraints into a convex QP.

## Highlights
- Two-layer Mamba encoder maps a 50-step state–torque history to a 32-D context that generates the Koopman A and B matrices.
- Koopman VAE lifts quaternion and angular velocity (7-D) into a 32-D latent space with closed-form Gaussian covariance propagation.
- KOZ chance constraints with Boole risk allocation, online covariance inflation and Monte Carlo linearization margins; Clarabel QP.
- Basilisk simulation with J4 gravity, drag, SRP and gravity gradient; 100k training sequences with randomized, misaligned inertia.
- 500-trial Monte Carlo under abrupt inertia change: 0.0 % KOZ violation vs 60.8–99.6 % for NMPC, L-SMPC, UKF-SMPC and ASGP-SMPC; 59.9 ms worst-case solve.

## Stack
Python · PyTorch · mamba-ssm · Basilisk · CasADi/IPOPT · Clarabel · OSQP · Hydra · Weights & Biases · MATLAB

## Layout
```text
src/          Learned models, losses, datasets, training
sim/          Basilisk controllers, KOZ, UKF, Monte Carlo
scripts/      Data, training, calibration, evaluation, plots
configs/      Hydra configs: controllers, spacecraft, scenarios
data/         GGM03S gravity field model
```

## Usage
```bash
pip install -r requirements.txt
pip install mamba-ssm --no-build-isolation
python -m scripts.1_gen_data data_gen.name=clean data_gen.num_seqs=100000 sim.sim_time=46
python -m scripts.2_train_party wandb.mode=disabled
python -m scripts.10_calibrate_party controller.model_dir=<YYYY-MM-DD/HH-MM-SS>
python -m scripts.3_run_mc controller.model_dir=<YYYY-MM-DD/HH-MM-SS>
python -m scripts.3_run_mc --config-name mc_nmpc
```

## Citation
```bibtex
@article{lee2026cak,
  title   = {Context-aware Koopman stochastic MPC for spacecraft attitude control with time-varying inertia},
  author  = {Lee, Yeonseo and Kim, Minhyung and Kim, Seongjae and Yang, Seungmin and Park, Hyeongjun},
  journal = {Aerospace Science and Technology},
  volume  = {178},
  pages   = {112960},
  year    = {2026},
  doi     = {10.1016/j.ast.2026.112960}
}
```
