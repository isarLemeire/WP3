import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .GFLA import GFLA

class GFLA_BSR(GFLA):

    def __init__(self, model, num_block=2, num_copies=5, max_angles=0.2,
                 num_neighbors=3, beta=4.5, **kwargs):
        super().__init__(model=model, **kwargs)
        self.num_block = num_block
        self.num_copies = num_copies
        self.max_angles = max_angles
        self.num_neighbors = num_neighbors
        self.beta = beta

    def _get_random_partition(self, length, num_block):
        """Partitions `length` into `num_block` random integer segments >= 1."""
        if length < num_block:
            raise ValueError(f"Length ({length}) must be >= num_block ({num_block})")
        rem_length = length - num_block
        rand = np.random.uniform(size=num_block)
        rand_norm = np.round(rand * rem_length / rand.sum()).astype(int)
        rand_norm[rand_norm.argmax()] += rem_length - rand_norm.sum()
        return (rand_norm + 1).tolist()

    def _rotate_blocks(self, x, n_groups, group_size):
        """Rotates each sample by an angle shared across all n_groups stacked
        copies of that sample index -- keeps clean/neighbor/adv paired."""
        angles_rad = torch.randn(group_size, device=x.device) * self.max_angles
        angles_rad = torch.clamp(angles_rad, -2 * self.max_angles, 2 * self.max_angles)
        angles_deg = torch.rad2deg(angles_rad.repeat(n_groups))
        rotated = [
            TF.rotate(x[i:i + 1], angle=angles_deg[i].item(),
                      interpolation=TF.InterpolationMode.BILINEAR)
            for i in range(x.shape[0])
        ]
        return torch.cat(rotated, dim=0)

    def _apply_bsr_transform(self, x_concat, n_groups, group_size):
        """Splits x_concat into random blocks, permutes, and rotates them --
        identically across all n_groups stacked copies (T_k)."""
        _, _, H, W = x_concat.shape
        height_lengths = self._get_random_partition(H, self.num_block)
        width_lengths = self._get_random_partition(W, self.num_block)
        width_perm = np.random.permutation(self.num_block)
        height_perm = np.random.permutation(self.num_block)

        x_split_w = torch.split(x_concat, width_lengths, dim=3)
        x_split_w_perm = [x_split_w[i] for i in width_perm]

        strips = []
        for strip in x_split_w_perm:
            strip_split_h = torch.split(strip, height_lengths, dim=2)
            blocks = [self._rotate_blocks(strip_split_h[i], n_groups, group_size)
                      for i in height_perm]
            strips.append(torch.cat(blocks, dim=2))
        return torch.cat(strips, dim=3)

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

    def _get_clean_reference(self, x_clean_bsr):
        """Clean feature baseline under T_k -- the fixed displacement
        anchor, analogous to GFLA's clean_feats."""
        _, clean_feats = self._forward_with_features(x_clean_bsr)
        return [f.detach() for f in clean_feats]

    def _get_guided_direction(self, x_neighbors_bsr, labels, n_neighbors, B):
        """Averaged feature-space CE gradient over VMI neighbors under T_k --
        analogous to GFLA's fixed_grads."""
        neighbor_logits, neighbor_feats = self._forward_with_features(x_neighbors_bsr)
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

    # ------------------------------------------------------------------ #
    # Feature-space loss -- identical formula to GFLA.compute_loss
    # ------------------------------------------------------------------ #
    def _feature_loss(self, adv_feats, clean_feats, fixed_grads, batch_size):
        layer_loss = 0.0
        num_layers = len(self.layer_registry)

        for f_adv, f_clean, g in zip(adv_feats, clean_feats, fixed_grads):
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)
            g_unit = g_flat / (torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8)

            proj = (delta_f * g_unit).sum(dim=1)    # guided component
            dist = torch.norm(delta_f, p=2, dim=1)   # unguided component
            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        return layer_loss / (num_layers + 1e-8)

    # ------------------------------------------------------------------ #
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B = images.shape[0]

        momentum = torch.zeros_like(images).detach()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)

            for _ in range(self.num_copies):
                adv_images_copy = adv_images.clone().detach().requires_grad_(True)

                # One shared T_k, applied identically to
                # [clean image, VMI neighbors, adv image].
                clean_stack, n_neighbors = self._sample_vmi_neighbors(images)
                x_concat = torch.cat([images, clean_stack, adv_images_copy], dim=0)
                n_groups = 1 + n_neighbors + 1
                x_bsr = self._apply_bsr_transform(x_concat, n_groups, B)

                x_clean_bsr = x_bsr[:B]
                x_neighbors_bsr = x_bsr[B:-B]
                x_adv_bsr = x_bsr[-B:]

                clean_feats = self._get_clean_reference(x_clean_bsr)
                fixed_grads = self._get_guided_direction(x_neighbors_bsr, labels, n_neighbors, B)

                _, adv_feats = self._forward_with_features(x_adv_bsr)
                feat_loss = self._feature_loss(adv_feats, clean_feats, fixed_grads, B)

                g_step = torch.autograd.grad(feat_loss, adv_images_copy, retain_graph=False)[0] # type: ignore
                grad += g_step

            grad = grad / float(self.num_copies)
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.momentum * momentum + grad_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images