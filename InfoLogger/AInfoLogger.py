import torch
from abc import ABCMeta, abstractmethod
from typing import Dict


class AInfoLogger(metaclass=ABCMeta):
    @abstractmethod
    def try_print_log(self, loss: torch.Tensor, eval_result: Dict, step: int, global_step: int, epoch_steps: int,
                      num_epochs: int, *args, **kwargs):
        pass
