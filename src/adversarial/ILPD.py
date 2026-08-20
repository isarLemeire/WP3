import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureAttack import FeatureAttack

class ILPD(FeatureAttack):
    def __init__(self, *args, coef=0.1, sigma=0.05, N=1, **kwargs):
        super().__init__(attack_name="ILPD", *args, **kwargs)
        self.coef = coef
        self.sigma = sigma
        self.N = max(1, N)

    def _get_ilpd_hook(self, ori_ilout):
        """Returns a hook that blends the current output with the static reference."""
        def hook_pd(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            blended = self.coef * out + (1.0 - self.coef) * ori_ilout

            if isinstance(output, tuple):
                return (blended,) + output[1:]
            return blended
        return hook_pd

    def setup_references(self, images, labels):
        return images.detach()

    def compute_loss(self, adv_images, labels, references):
        """
            L( g_i( gamma * h_i(x+delta) + (1-gamma) * h_i(x) ), y )

        computed independently for EACH layer i (every other layer runs
        un-perturbed), then averaged over layers and over N noise samples.
        """
        num_layers = len(self.layer_registry)

        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 1

        accumulated_grad = torch.zeros_like(adv_images)

        for _ in range(self.N):
            noisy_input = references + self.sigma * torch.randn_like(references)

            self.feature_outputs.clear()
            with torch.no_grad():
                self.get_logits(noisy_input)

            ori_ilouts = [f.detach() for f in self.feature_outputs]
            self.feature_outputs.clear()

            for i, mod in enumerate(self.layer_registry):
                pd_handle = mod.register_forward_hook(self._get_ilpd_hook(ori_ilouts[i]))
                try:
                    logits = self.get_logits(adv_images)
                    loss_i = F.cross_entropy(logits, labels)
                    grad_i = torch.autograd.grad(loss_i, adv_images)[0]
                    accumulated_grad += grad_i / (self.N * num_layers)
                finally:
                    pd_handle.remove()
                    self.feature_outputs.clear()

        # Summing loss_i slows down the code due to filling up the computational graph,
        # proxy_loss differs from the official ILPD paper,
        # but results in the same gradient from the optimizers POV.
        proxy_loss = (adv_images * accumulated_grad.detach()).sum()

        return proxy_loss, 1