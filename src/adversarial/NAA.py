import torch
import torch.nn.functional as F
from .FeatureAttack import FeatureAttack

class NAA(FeatureAttack):
    def __init__(self, *args, num_steps_ig=30, random_noise = 0.2, **kwargs):
        super().__init__(attack_name="NAA", *args, **kwargs)
        self.num_steps_ig = int(num_steps_ig)
        self.random_noise = random_noise

    def setup_references(self, images, labels): # type: ignore
        """Computes Neuron Attribution weights (W) and Baseline Features (B)."""
        attributions = []
        base_features = []
        
        # Baseline is typically a zero/black image
        baseline = torch.zeros_like(images).to(self.device)
        num_classes = 1000
        one_hot_labels = F.one_hot(labels, num_classes).float()

        self.model.zero_grad()
        
        # Register hooks for IG phase
        handles = [mod.register_forward_hook(self._get_hook()) 
                    for mod in self.layer_registry]
        
        # 1. Capture Baseline Features
        self.feature_outputs.clear()
        with torch.no_grad():
            self.get_logits(baseline)

        base_features = [f.detach() for f in self.feature_outputs]
        
        # 2. Integrated Gradients loop with Noise Injection
        IA = None
        for step in range(1, self.num_steps_ig + 1):
            alpha = step / self.num_steps_ig

            # interpolate new x
            noise = torch.randn_like(images) * self.random_noise # not mentioned in paper, but given in git
            x_noisy = images + noise
            x_interp = baseline + alpha * (x_noisy - baseline)
            x_interp = x_interp.detach().requires_grad_(True)
            
            self.feature_outputs.clear()
            logits = self.get_logits(x_interp)
            
            # Official NAA uses Softmax probabilities
            prob = F.softmax(logits, dim=1)
            true_class_prob = (prob * one_hot_labels).sum()
            
            layer_feats = self.feature_outputs
            grads = torch.autograd.grad(
                true_class_prob,
                layer_feats,
                retain_graph=False,
                allow_unused=True,
            )
            
            if IA is None:
                IA = [g.detach() for g in grads]
            else:
                for i in range(len(grads)):
                    IA[i] += grads[i].detach()

        for h in handles: h.remove()

        # 3. Final Weights: Negated and Normalized Average Gradient
        for g in IA: # type: ignore
            reduce_dims = tuple(range(1, g.ndim))
            norm = torch.norm(g, p=2, dim=reduce_dims, keepdim=True) + 1e-8 # not mentioned in paper, but given in git
            attributions.append(-(g / norm))
                
        return {"attributions": attributions, "baselines": base_features}

    def compute_loss(self, adv_images, labels, references):      
        self.get_logits(adv_images)
        
        model_layer_loss = 0
        current_feats = self.feature_outputs
        W_list = references["attributions"]
        B_list = references["baselines"]
        num_layers = len(current_feats)
        
        for y_1, W, y_2 in zip(current_feats, W_list, B_list):
            # Official NAA: attribution = (adv_feat - base_feat) * weights
            # Weights are already negated, thus maximize
            A_y = (y_1 - y_2) * W
            
            # W A_y = A_y
            # since the official NAA paper found the optimal values:
            # f_p and f_n to be linear
            # gamma = 1
            
            # Official git does not include l_1 normalization mentioned in paper
            model_layer_loss += (A_y.sum() / y_1.numel())
        
        loss = (model_layer_loss / (num_layers + 1e-8))
                
        return loss, 1