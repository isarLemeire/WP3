import torch
import torch.nn as nn
import torch.nn.functional as F
from src.adversarial.FeatureAttack import FeatureAttack

class DFAA(FeatureAttack):
    def __init__(self, *args, attack_name="DFAA", lam=0.3, **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.lam = lam

        self.mte_hooks = []
        self.is_ref_phase = False 


    def compute_loss(self, adv_images, labels, references):
        self.is_ref_phase = False 
        
        ensemble_ce = 0
        ensemble_dual = 0
        

        logits = self.get_logits(adv_images)
        ensemble_ce += F.cross_entropy(logits, labels)
        
        f_adv_list = self.feature_outputs
        f_clean_list = references
        num_layers = len(f_adv_list)

        model_dual = 0
        for f_adv, f_clean in zip(f_adv_list, f_clean_list):
            mse = F.mse_loss(f_adv, f_clean)
            model_dual += torch.sigmoid(-mse) 
        
        ensemble_dual += (model_dual / (num_layers + 1e-8))

        loss = (self.lam * ensemble_ce - \
                ((1 - self.lam) * ensemble_dual ))
        

        return loss, 1