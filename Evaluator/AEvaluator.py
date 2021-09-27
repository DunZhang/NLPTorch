"""
评估器的积累
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from torch.utils.data import DataLoader
from Model.AModel import AModel


class AEvaluator():
    @abstractmethod
    def evaluate(self, model: AModel, data_loader: DataLoader, *args, **kwargs) -> Dict:
        pass

    @abstractmethod
    def try_evaluate(self, model: AModel, data_loader: DataLoader, step: int, global_step: int, epoch_steps: int,
                     num_epochs: int, *args, **kwargs) -> Union[Dict, None]:
        pass
