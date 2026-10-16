import os
import torch
import torch.nn as nn
import logging
import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class PartyModelConfig:
    """학습된 모델의 시간 관련 설정을 담는 데이터 클래스"""
    hist_dt: float      # History sampling interval [s]
    pred_dt: float      # Prediction timestep (= mpc_dt) [s]
    hist_len: int       # History length (number of past states)


def print_model_summary(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info("Model Parameter Summary:")
    log.info(f"  Total Parameters: {total_params:,}")
    log.info(f"  Trainable Parameters: {trainable_params:,}")
    
    log.info("  Component Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        log.info(f"    {name}: {params:,}")


def load_party_interface(model_dir: str, device: str = "cuda", use_custom_mamba: bool | None = None):
    base_dir = os.path.join("outputs/learning_party", model_dir)
    cfg_path = os.path.join(base_dir, ".hydra", "config.yaml")
    weights_path = os.path.join(base_dir, "best_model.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    cfg = OmegaConf.load(cfg_path)

    if use_custom_mamba is not None:
        OmegaConf.update(cfg, "model.use_custom_mamba", use_custom_mamba)
        log.info(f"Overriding model.use_custom_mamba={use_custom_mamba}")

    # Legacy compat: old configs may have params removed from current Party
    _legacy_model_keys = ["context_conditioning", "branch_decoder"]
    for key in _legacy_model_keys:
        if key in cfg.model:
            log.info(f"Removing legacy model config key: {key}={cfg.model[key]}")
            del cfg.model[key]

    log.info(f"Loading model from {weights_path}...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint (strict: expect our own training format)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Remap legacy key names (log_var_head -> logvar_head)
    state_dict = {
        k.replace('log_var_head', 'logvar_head'): v for k, v in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        log.warning(f"Missing keys in state_dict (initialized randomly or defaults): {missing_keys}")
    if unexpected_keys:
        log.warning(f"Unexpected keys in state_dict (ignored): {unexpected_keys}")

    model.to(device)
    model.eval()

    # Wrap with PartyInterface and load normalization stats there
    from src.model.party_interface import PartyInterface  # lazy import to avoid circular dependency

    data_name = cfg.data.name
    stats_path = os.path.join(base_dir, "stats", f"{data_name}_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}")

    party_interface = PartyInterface(model)
    party_interface.set_eval()
    party_interface.load_stats(stats_path)
    party_interface.to(device)
    party_interface.eval()

    log.info(f"Loaded stats into PartyInterface from {stats_path}")

    # Extract model config from training config
    model_cfg = PartyModelConfig(
        hist_dt=float(cfg.data.hist_dt),
        pred_dt=float(cfg.data.pred_dt),
        hist_len=int(cfg.data.hist_len),
    )
    log.info(f"Model config: hist_dt={model_cfg.hist_dt}s, pred_dt={model_cfg.pred_dt}s, hist_len={model_cfg.hist_len}")

    return party_interface, model_cfg
