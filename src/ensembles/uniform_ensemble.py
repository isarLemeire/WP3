import torch
from torch import Tensor
import torch.nn as nn

class UniformEnsemble(nn.Module):
    def __init__(self, models_list: list):
        super(UniformEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        
        self.data = None # Unused var

    def forward(self, x: Tensor) -> Tensor:
        all_logits = [model(x) for model in self.models]
        
        stacked_logits = torch.stack(all_logits, dim=0)
        avg_logits = torch.mean(stacked_logits, dim=0)
        
        return avg_logits