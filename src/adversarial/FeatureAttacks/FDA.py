import torch
from .FeatureAttack import FeatureAttack

class FDA(FeatureAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(attack_name="FDA", *args, **kwargs)
        self.eps_ratio = 1e-8
        # FDA does not use a momentum component
        kwargs.setdefault('momentum', 0.0)

    def compute_loss(self, adv_images, labels, references): # type: ignore
        self.get_logits(adv_images)
        
        model_layer_loss = 0

        current_feats = self.feature_outputs
        num_layers = len(current_feats)
        
        for i, f_adv in enumerate(current_feats):
            # Using the clean reference to create mask
            f_clean = references[i]

            reduce_dims = tuple(range(1, f_clean.ndim - 1)) if f_clean.ndim == 4 else (1,)
            mu = f_clean.mean(dim=reduce_dims, keepdim=True)
            m_high = (f_clean > mu).float()
            
            dims = tuple(range(1, f_adv.ndim))
            e_high = torch.norm(f_adv * m_high, p=2, dim=dims)
            e_low = torch.norm(f_adv * (1 - m_high), p=2, dim=dims)
            
            model_layer_loss += (
                torch.log(e_low  + self.eps_ratio)
                - torch.log(e_high + self.eps_ratio)
            ).sum()
        
        loss = (model_layer_loss / (num_layers + 1e-8))
        
        return loss, 1