import torch
import torch.nn as nn
from torch import Tensor
from ..model.model_wrapper import ModelWrapper

class LabelInjectedModel(ModelWrapper):
    def __init__(self, model: nn.Module, device: torch.device, name : str = "model", input_size: int =224, **model_kwargs):
        super().__init__(model, device, name, input_size, **model_kwargs)

    def forward(self, x : Tensor) -> Tensor:
        return self.model(x, labels=self.labels)