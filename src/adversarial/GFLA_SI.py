import torch
import torch.nn.functional as F
from .GFLA import GFLA

class GFLA_SI(GFLA):
    def __init__(self, model, m=5, num_neighbors=3, beta=4.5, **kwargs):
        super().__init__(model=model, **kwargs)
        self.m = m 
        self.num_neighbors = num_neighbors
        self.beta = beta

    def _forward_with_features(self, x):
        self.feature_outputs.clear()
        handles = [m.register_forward_hook(self._get_hook()) for m in self.layer_registry]
        logits = self.get_logits(x)
        features = list(self.feature_outputs)
        for h in handles:
            h.remove()
        return logits, features

    def _sample_vmi_neighbors(self, images):
        """Generates num_neighbors noisy neighbors of `images`. Mirrors
        GFLA._get_vmi_sample, batched across num_neighbors at once."""
        if self.num_neighbors > 1:
            neighbors = [
                (images + (torch.rand_like(images) * 2 - 1) * (self.eps * self.beta)).clamp(0, 1)
                for _ in range(self.num_neighbors)
            ]
            return torch.cat(neighbors, dim=0), self.num_neighbors
        return images, 1

    def _get_clean_reference(self, x_clean_scaled):
        _, clean_feats = self._forward_with_features(x_clean_scaled)
        return [f.detach() for f in clean_feats]

    def _get_guided_direction(self, x_neighbors_scaled, labels, n_neighbors, B):
        neighbor_logits, neighbor_feats = self._forward_with_features(x_neighbors_scaled)
        neighbor_loss = F.cross_entropy(neighbor_logits, labels.repeat(n_neighbors))
        raw_grads = torch.autograd.grad(neighbor_loss, neighbor_feats, retain_graph=False)

        fixed_grads = []
        for g in raw_grads:
            if n_neighbors > 1:
                g = g.detach().reshape(n_neighbors, B, *g.shape[1:]).mean(0)
            else:
                g = g.detach()
            fixed_grads.append(g)
        return fixed_grads

    def _feature_loss(self, adv_feats, clean_feats, fixed_grads, batch_size):
        layer_loss = 0.0
        num_layers = len(self.layer_registry)

        for f_adv, f_clean, g in zip(adv_feats, clean_feats, fixed_grads):
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)
            g_unit = g_flat / (torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8)

            proj = (delta_f * g_unit).sum(dim=1)
            dist = torch.norm(delta_f, p=2, dim=1)
            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        return layer_loss / (num_layers + 1e-8)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B = images.shape[0]

        momentum = torch.zeros_like(images).detach()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)

            for i in range(self.m):
                adv_images_copy = adv_images.clone().detach().requires_grad_(True)
                scale = 1.0 / (2 ** i)

                clean_stack, n_neighbors = self._sample_vmi_neighbors(images)
                x_concat = torch.cat([images, clean_stack, adv_images_copy], dim=0)
                x_scaled = x_concat * scale  # deterministic, no permutation needed

                x_clean_scaled = x_scaled[:B]
                x_neighbors_scaled = x_scaled[B:-B]
                x_adv_scaled = x_scaled[-B:]

                clean_feats = self._get_clean_reference(x_clean_scaled)
                fixed_grads = self._get_guided_direction(x_neighbors_scaled, labels, n_neighbors, B)

                _, adv_feats = self._forward_with_features(x_adv_scaled)
                feat_loss = self._feature_loss(adv_feats, clean_feats, fixed_grads, B)

                g_step = torch.autograd.grad(feat_loss, adv_images_copy, retain_graph=False)[0] # type: ignore
                grad += g_step

            grad = grad / float(self.m)
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.momentum * momentum + grad_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images