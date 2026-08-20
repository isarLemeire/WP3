import torch
import torch.nn.functional as F
from .FeatureAttack import FeatureAttack

class GFLA(FeatureAttack):
    def __init__(self, *args, lam=0.9, N=5, beta=4.5, **kwargs):
        super().__init__(attack_name="GFLA", *args, **kwargs)
        self.lam = lam
        self.N = N  # Neighborhood samples
        self.beta = beta # Noise scale factor

    def _get_vmi_sample(self, images):
        """Generates a noisy neighbor for VMI sampling."""
        images_det = images.detach()
        if self.N > 0:
            noise = (torch.rand_like(images_det) * 2 - 1) * (self.eps * self.beta)
            return (images_det + noise).clamp(0, 1).requires_grad_(True)
        return images_det.clone().requires_grad_(True)

    def setup_references(self, images, labels): # type: ignore
        """
        Pre-computes clean features and averaged feature-space gradients.
        """
        
        # Clean features
        clean_feats = self._get_features(images)

        # Adversarial gradient (VMI-style)
        fixed_grads = []
        num_samples = max(1, self.N)       

        for _ in range(num_samples):
            x_neighbor = self._get_vmi_sample(images)
        
            self.feature_outputs.clear()

            handles = [mod.register_forward_hook(self._get_hook())
                        for mod in self.layer_registry]
           
            logits = self.get_logits(x_neighbor)
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, self.feature_outputs)

            if not fixed_grads:
                fixed_grads = [g.detach() / num_samples for g in grads]
            else:
                for i, g in enumerate(grads):
                    fixed_grads[i] += g.detach() / num_samples

            for h in handles: h.remove()

        return {
            'clean_feats': clean_feats,
            'fixed_grads': fixed_grads
        }

    def compute_loss(self, adv_images, labels, references):
        batch_size = adv_images.shape[0]
        clean_feats = references['clean_feats']
        fixed_grads = references['fixed_grads']

        self.get_logits(adv_images)

        layer_loss = 0
        num_layers = len(self.layer_registry)

        for f_adv, f_clean, g in zip(
            self.feature_outputs,
            clean_feats,
            fixed_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            # Guided component
            proj = (delta_f * g_unit).sum(dim=1)
            # Unguided component
            dist = torch.norm(delta_f, p=2, dim=1)

            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = (layer_loss / (num_layers + 1e-8))

        return loss, 1