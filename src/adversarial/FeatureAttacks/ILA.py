import torch
import torch.nn.functional as F
from .FeatureAttack import FeatureAttack

class ILA(FeatureAttack):
    def __init__(self, *args, **kwargs):
        # ILA does not use a momentum component
        kwargs.setdefault('momentum', 0.0)
        super().__init__(attack_name="ILA", *args, **kwargs)

    def setup_references(self, images, labels): # type: ignore
        # 1. Generate Guide via 10-step BIM
        guide_images = images.clone().detach().requires_grad_(True)
        
        for _ in range(10):
            outputs = self.get_logits(guide_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Average the loss across the ensemble
            avg_loss = loss
            grad = torch.autograd.grad(avg_loss, guide_images)[0] # type: ignore
            
            # BIM
            guide_images = (guide_images + self.alpha * grad.sign()).detach()
            delta = torch.clamp(guide_images - images, -self.eps, self.eps)
            guide_images = torch.clamp(images + delta, 0, 1).requires_grad_(True)
        
        # 2. Capture both Clean and Guide features for all models
        with torch.no_grad():
            clean_feats = self._get_features(images)
            guide_feats = self._get_features(guide_images.detach())
        
        return {"clean": clean_feats, "guide": guide_feats}

    def compute_loss(self, adv_images, labels, references): # type: ignore
        self.get_logits(adv_images)
        
        model_layer_loss = 0
        current_feats = self.feature_outputs
        num_layers = len(current_feats)
        
        for i, f_adv in enumerate(current_feats):
            f_clean = references["clean"][i]
            f_guide = references["guide"][i]
            
            # Project current feature perturbation onto guide perturbation
            dg = (f_guide - f_clean).reshape(f_clean.shape[0], -1)
            da = (f_adv - f_clean).reshape(f_clean.shape[0], -1)
            
            projection = (da * dg).sum(dim=1).sum()
            model_layer_loss += projection
        
        loss = (model_layer_loss / (num_layers + 1e-8))
                
        return loss, 1