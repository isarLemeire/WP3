import torch
import torch.nn.functional as F
from .FeatureAttack import FeatureAttack

class FIA(FeatureAttack):
    def __init__(self, *args, drop_rate=0.3, num_ensemble=30, num_classes = 1000, **kwargs):
        super().__init__(attack_name="FIA", *args, **kwargs)
        self.drop_rate = drop_rate
        self.num_ensemble = num_ensemble
        self.num_classes = num_classes

    def setup_references(self, images, labels): # type: ignore
        """Pre-computes the aggregate gradient-based importance weights (W)."""
        importance_weights = []
        
        one_hot_labels = F.one_hot(labels, self.num_classes).float().to(self.device)

        self.model.zero_grad()

        agg_grads = None
        for _ in range(self.num_ensemble):
            # Apply random drop mask
            keep_mask = torch.bernoulli(
                torch.full_like(images, 1.0 - self.drop_rate)
            )
            x_masked = (images * keep_mask).detach()
            
            feature_refs: list[torch.Tensor] = []

            def _capture_hook(mod, inp, out,
                                _refs=feature_refs):
                _refs.append(out[0] if isinstance(out, tuple) else out)

            handles = [layer.register_forward_hook(_capture_hook)
                        for layer in self.layer_registry]

            x_in = x_masked.requires_grad_(False)
            logits = self.get_logits(x_in)

            for h in handles:
                h.remove()
            
            # FIA Logic: Gradient of (Logits * y_true) w.r.t internal features
            loss = (logits * one_hot_labels).sum()
            layer_grads = torch.autograd.grad(
                loss,
                feature_refs,           # differentiate through live tensors
                retain_graph=False,
                create_graph=False,
            )
            
            if agg_grads is None:
                agg_grads = [g.detach().clone() for g in layer_grads]
            else:
                for i, g in enumerate(layer_grads):
                    agg_grads[i] += g.detach()
        
        # Normalize
        normed: list[torch.Tensor] = []
        for g in agg_grads: # type: ignore
            l2 = g.norm(p=2)
            normed.append(-(g / (l2 + 1e-12)))

        importance_weights = normed
                
        return importance_weights

    def compute_loss(self, adv_images, labels, references): # type: ignore
        self.get_logits(adv_images) # forward fills self.feature_outputs
        
        model_layer_loss = 0

        current_feats = self.feature_outputs
        W_list = references
        num_layers = len(current_feats)
        
        for feat, w in zip(current_feats, W_list):
            n_elements = feat.numel()
            model_layer_loss += (feat * w).sum() / n_elements
        
        loss = (model_layer_loss / (num_layers))
                
        return loss, 1