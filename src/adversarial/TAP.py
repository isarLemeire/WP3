import torch
import torch.nn as nn
import torch.nn.functional as F
from src.adversarial.FeatureAttack import FeatureAttack

class TAP(FeatureAttack):
    def __init__(self, *args, attack_name="TAP", lam=0.005, power= 0.5, eta=1e3, **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.lam = lam
        self.power = power
        self.eta = eta

        self.kernel = self._get_box_kernel3x3()

    def _get_box_kernel3x3(self):
        """3x3 spatial domain box (averaging) linear filter."""
        return torch.full((3, 3), fill_value=1.0 / 9.0, dtype=torch.float32)

    def T(self, feat):
        return torch.sign(feat) * torch.float_power(torch.abs(feat) + 1e-8, self.power)

    def compute_response_map(self, perturbation, kernel):
        B, C, H, W = perturbation.shape
        padding = kernel.shape[-1] // 2

        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        return F.conv2d(
            perturbation, 
            kernel.to(perturbation.device), 
            padding=padding, 
            groups=C
        )

    def setup_references(self, images, labels): # type: ignore
        with torch.no_grad():
            return {
                'clean_images': images.clone().detach(),
                'clean_feats': self._get_features(images)
            }


    def compute_loss(self, adv_images, labels, references):
        batch_size = adv_images.shape[0]
        ce_loss = 0
        f_loss = 0

        logits = self.get_logits(adv_images)
        ce_loss += F.cross_entropy(logits, labels)

        f_adv_list = self.feature_outputs
        num_layers = len(f_adv_list)

        clean_images = references['clean_images']
        clean_feats = references['clean_feats']
        perturbation = adv_images - clean_images



        for f_adv, f_clean in zip(f_adv_list, clean_feats):
            t_clean = self.T(f_clean).reshape(batch_size, -1)
            t_adv = self.T(f_adv).reshape(batch_size, -1)
            delta = t_clean - t_adv

            f_loss += torch.sum(delta ** 2, dim=1).mean()

        avg_f_loss = (f_loss / (num_layers + 1e-8))

        R = self.compute_response_map(perturbation, self.kernel)
        R_loss = torch.abs(R).mean()

        loss = ce_loss + self.lam * avg_f_loss + self.eta * R_loss

        return loss, 1