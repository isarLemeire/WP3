
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustEnsemble(nn.Module):
    def __init__(self, models_list: list, T=1.0, minimum : float =0.0, maximum : float =1.0):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.register_buffer("weights", torch.full((len(models_list),), 1.0 / len(models_list)))
        self.T = T
        self.min = max(0.0, minimum)
        self.max = min(1.0, maximum)
        self.name = "robust"
        

    def update_adaptive_weights(self, x, labels=None):
        batch_size = x.shape[0]
        num_models = len(self.models)

        if labels is None:
            self.weights = torch.full(
                (num_models, batch_size), 
                1.0 / num_models, 
                device=x.device
            )
            return [model(x) for model in self.models]

        if not x.requires_grad:
            x.requires_grad = True

        is_targeted = getattr(self, "targeted", False)

        with torch.enable_grad():
            scores = []
            all_logits = []

            for model in self.models:
                logits = model(x)
                all_logits.append(logits)
                
                one_hot = F.one_hot(labels, num_classes=logits.shape[1]).bool()
                correct_logits = logits[one_hot].view(batch_size, 1)
                other_logits_max = logits[~one_hot].view(batch_size, -1).max(dim=1, keepdim=True)[0]
                margin = (correct_logits - other_logits_max).squeeze()

                if is_targeted:
                    numerator = F.relu(-margin) + 1e-8
                else:
                    numerator = F.relu(margin) + 1e-8
                
                loss = F.cross_entropy(logits, labels)
                g_grad = torch.autograd.grad(loss, x, retain_graph=True)[0]               
                g_norm_sq = torch.norm(g_grad, p=2, dim=(1, 2, 3))

                current_score = torch.log(numerator) - 2*torch.log(g_norm_sq + 1e-8)
                scores.append(current_score)

            stacked_scores = torch.stack(scores) # [num_models, batch_size]
            raw_weights = F.softmax(stacked_scores / self.T, dim=0)
            self.weights = self._project_weights(raw_weights).detach()
            
            #print(self.weights)
            
            return all_logits
        

            
    def _project_weights(self, w, max_iters=10):
        num_models = len(self.models)
        
        # 0. Safety Guardrail: If min is too high, just go uniform
        if self.min * num_models > 1.0 or self.max * num_models < 1.0:
            return torch.full_like(w, 1.0 / num_models)

        w_constrained = w.clone()
        
        for _ in range(max_iters):
            # 1. Clamp to current bounds
            w_new = torch.clamp(w_constrained, self.min, self.max)
            
            # 2. Check sum
            current_sum = w_new.sum(dim=0, keepdim=True)
            if torch.allclose(current_sum, torch.ones_like(current_sum), atol=1e-5):
                return w_new
            
            diff = (1.0 - current_sum)
            
            # 3. Find non-pinned indices
            # Added a tiny epsilon to the comparison to avoid float precision issues
            not_pinned = (w_new > self.min + 1e-7) & (w_new < self.max - 1e-7)
            num_not_pinned = not_pinned.sum(dim=0, keepdim=True).float()
            
            if torch.any(num_not_pinned == 0):
                break
                
            w_constrained = w_new + (diff / num_not_pinned) * not_pinned.float()
            
        # Final pass: Normalize what we have as a last resort
        final_w = torch.clamp(w_constrained, self.min, self.max)
        return final_w / final_w.sum(dim=0, keepdim=True)
            

    def forward(self, x, labels=None):
        all_logits = self.update_adaptive_weights(x, labels)
        stacked_logits = torch.stack(all_logits, dim=0) 
        # Weights [num_models, batch_size] * logits [num_models, batch_size, num_classes]
        final_logits = torch.einsum('mb, mbc -> bc', self.weights, stacked_logits)
                
        return final_logits