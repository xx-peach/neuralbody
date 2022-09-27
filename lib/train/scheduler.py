from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR


def make_lr_scheduler(cfg, optimizer):
    """ Instantiate lc_scheduler According to cfg.train.scheduler
        MultiStepLR, ExponentialLR 都是继承自 torch.optim.lr_scheduler._LRScheduler 的
    
    Args:
        cfg.train.scheduler - specifies which schedule strategy we wonna use and its parameters
        optimizer           - optimizer for lr_scheduler class
    Returns:
        scheduler - instantiation of the actual lr scheduler
    """
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    scheduler.gamma = cfg_scheduler.gamma
