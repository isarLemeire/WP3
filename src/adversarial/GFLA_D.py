import torch
import torch.nn.functional as F
from .FeatureAttack import FeatureAttack

class GFLA_D(FeatureAttack):
    def __init__(self, *args, lam=0.9, N=5, beta=4.5, **kwargs):
        """
        Dynamic GFLA that recalculates feature-space gradients at every step.
        """
        super().__init__(attack_name="GFLA_D", *args, **kwargs)
        self.lam = lam 
        self.N = N  
        self.beta = beta 

    def _get_vmi_sample(self, images):
        """Generates a noisy neighbor for VMI sampling."""
        images_det = images.detach() 
        if self.N > 0:
            noise = (torch.rand_like(images_det) * 2 - 1) * (self.eps * self.beta)
            return (images_det + noise).clamp(0, 1).requires_grad_(True)
        return images_det.clone().requires_grad_(True)

    def _compute_step_gradients(self, adv_images, labels):
        """
        Computes the current step's feature-space gradients using VMI neighborhood sampling.
        Uses the base class's permanent hooks to avoid duplicate registration.
        """
        step_grads = []
        num_samples = max(1, self.N)
        
        for _ in range(num_samples):
            x_neighbor = self._get_vmi_sample(adv_images)
            
            # Clear the outputs cache to capture EXACTLY this neighbor pass
            self.feature_outputs.clear()
            
            # Forward pass automatically triggers the base class permanent hooks
            logits = self.get_logits(x_neighbor)
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, self.feature_outputs)
            
            if not step_grads:
                step_grads = [g.detach() / num_samples for g in grads]
            else:
                for i, g in enumerate(grads):
                    step_grads[i] += g.detach() / num_samples
                
        return step_grads

    def compute_loss(self, adv_images, labels, references):
        batch_size = adv_images.shape[0]

        current_step_grads = self._compute_step_gradients(adv_images, labels)

        self.feature_outputs.clear()
        self.get_logits(adv_images)
        
        model_layer_loss = 0
        num_layers = len(self.layer_registry)
        
        for f_adv, f_clean, g in zip(
            self.feature_outputs, 
            references, 
            current_step_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            # Guided component
            proj = (delta_f * g_unit).sum(dim=1)
            # Unguided component
            dist = torch.norm(delta_f, p=2, dim=1)

            model_layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = (model_layer_loss / (num_layers + 1e-8))
        return loss, 1