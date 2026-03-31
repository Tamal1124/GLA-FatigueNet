"""Optimizer and scheduler configuration."""
import torch.optim as optim


def get_optimizer(model, config):
    cfg = config['training']['optimizer']
    if cfg['type'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=tuple(cfg['betas']))
    elif cfg['type'] == 'adam':
        return optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['type'] == 'sgd':
        return optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    raise ValueError(f"Unknown optimizer: {cfg['type']}")


def get_scheduler(optimizer, config):
    cfg = config['training']['scheduler']
    if cfg['type'] == 'cosine_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mult'], eta_min=cfg['eta_min'])
    elif cfg['type'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=cfg.get('eta_min', 1e-6))
    elif cfg['type'] == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('step_size', 30), gamma=cfg.get('gamma', 0.1))
    raise ValueError(f"Unknown scheduler: {cfg['type']}")
