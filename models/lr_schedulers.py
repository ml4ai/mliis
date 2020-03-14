import numpy as np
from typing import Optional


class LRScheduler:

    def __init__(self, initial_lr: float, total_steps: int):
        self.initial_lr = initial_lr
        self.total_steps = total_steps

    def anneal_lr(self, cur_step: int):
        """Implemented by subclass"""
        pass

    def cur_lr(self, cur_step):
        lr = self.anneal_lr(cur_step)
        return lr


class CosineLRScheduler(LRScheduler):

    def __init__(self, initial_lr: float, total_steps: int):
        super().__init__(initial_lr, total_steps)

    def anneal_lr(self, cur_step: int, min_to_decay_to: float = 0.0):
        lr = 0.5 * self.initial_lr * (1 + np.cos(np.pi * cur_step / self.total_steps))
        lr = np.max([lr, min_to_decay_to])
        return lr


class StepDecay(LRScheduler):

    def __init__(self, initial_lr: float, total_steps: Optional[int] = None, decay_rate: float = 0.5, decay_after_n_steps: int = 5, min_lr: float = 1e-7):
        super().__init__(initial_lr, total_steps)
        assert decay_rate is not None and decay_after_n_steps is not None
        self.decay_rate = decay_rate
        self.decay_after_n_steps = decay_after_n_steps
        self.min_lr = min_lr

    def anneal_lr(self, cur_step: int, decay_rate: Optional[float] = None, decay_after_n_steps: Optional[int] = None):
        if decay_after_n_steps is None:
            decay_after_n_steps = self.decay_after_n_steps
        if decay_rate is None:
            decay_rate = self.decay_rate
        m = cur_step // decay_after_n_steps
        lr = self.initial_lr * (decay_rate ** m)
        lr = self.min_lr if lr < self.min_lr else lr
        return lr


supported_learning_rate_schedulers = {"cosine_anneal": CosineLRScheduler, "fixed": None, "constant": None,
                                      "step": StepDecay, "step_decay": StepDecay}
