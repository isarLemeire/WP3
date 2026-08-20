import torch
from torch import Tensor
import torch.nn as nn
from typing import List

class RandomEnsemble(nn.Module):
    def __init__(self, models_list : List, minimum : float = 0.0, maximum : float = 0.5):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.min = minimum
        self.max = maximum
        self.num_models = len(models_list)
        
        self.data = None # Unused var

    def _project_weights(self, w, max_iters=10):
        """Ensures sum=1 and w in [min, max] for a random distribution."""
        w_constrained = w.clone()
        for _ in range(max_iters):
            w_new = torch.clamp(w_constrained, self.min, self.max)
            current_sum = w_new.sum(dim=0, keepdim=True)
            
            if torch.allclose(current_sum, torch.ones_like(current_sum), atol=1e-5):
                return w_new
            
            diff = (1.0 - current_sum)
            not_pinned = (w_new > self.min + 1e-7) & (w_new < self.max - 1e-7)
            num_not_pinned = not_pinned.sum(dim=0, keepdim=True).float()
            
            if torch.any(num_not_pinned == 0):
                break
                
            w_constrained = w_new + (diff / num_not_pinned) * not_pinned.float()
            
        final_w = torch.clamp(w_constrained, self.min, self.max)
        return final_w / (final_w.sum(dim=0, keepdim=True) + 1e-12)

    def forward(self, x : Tensor) -> Tensor:
        batch_size = x.shape[0]
        raw_rand = torch.rand((self.num_models, batch_size), device=x.device)
        
        weights = self._project_weights(raw_rand)
        
        all_logits = [model(x) for model in self.models]
        final_logits = 0
        for i in range(self.num_models):
            final_logits += weights[i].view(-1, 1) * all_logits[i]
            
        return torch.Tensor(final_logits)