import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchattacks.attack import Attack


class BSR(Attack):
    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=0.9,
        num_block=2,
        num_copies=5,
        max_angles=0.2,
    ):
        super().__init__("BSR", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.num_block = num_block
        self.num_copies = num_copies
        self.max_angles = max_angles  # Max rotation angle in Radians
        self.supported_mode = ["default"]

    def _get_random_partition(self, length, num_block):
        """Partitions a dimension length into `num_block` random integer segments >= 1."""
        if length < num_block:
            raise ValueError(f"Length ({length}) must be >= num_block ({num_block})")

        rem_length = length - num_block
        rand = np.random.uniform(size=num_block)
        
        rand_norm = np.round(rand * rem_length / rand.sum()).astype(int)
        rand_norm[rand_norm.argmax()] += rem_length - rand_norm.sum()
        
        return (rand_norm + 1).tolist()

    def _rotate_batch(self, x):
        """Rotates each item in batch `x` by a random angle sampled from truncated normal."""
        B = x.shape[0]
        angles_rad = torch.randn(B, device=x.device) * self.max_angles
        angles_rad = torch.clamp(angles_rad, -2 * self.max_angles, 2 * self.max_angles)
        angles_deg = torch.rad2deg(angles_rad)

        rotated_list = []
        for b in range(B):
            rot_img = TF.rotate(
                x[b : b + 1],
                angle=angles_deg[b].item(),
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            rotated_list.append(rot_img)
        return torch.cat(rotated_list, dim=0)

    def _shuffle_rotate(self, x):
        """Splits image tensor (B, C, H, W) into random blocks, permutes, and rotates them."""
        _, _, H, W = x.shape

        height_lengths = self._get_random_partition(H, self.num_block)
        width_lengths = self._get_random_partition(W, self.num_block)

        width_perm = np.random.permutation(self.num_block)
        height_perm = np.random.permutation(self.num_block)

        x_split_w = torch.split(x, width_lengths, dim=3)
        x_split_w_perm = [x_split_w[i] for i in width_perm]

        processed_strips = []
        for strip in x_split_w_perm:
            strip_split_h = torch.split(strip, height_lengths, dim=2)
            rotated_h_blocks = []
            for i in height_perm:
                block = strip_split_h[i]
                rotated_block = self._rotate_batch(block)
                rotated_h_blocks.append(rotated_block)

            processed_strip = torch.cat(rotated_h_blocks, dim=2)
            processed_strips.append(processed_strip)

        return torch.cat(processed_strips, dim=3)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)

            for _ in range(self.num_copies):
                adv_images_copy = adv_images.clone().detach().requires_grad_(True)

                x_bsr = self._shuffle_rotate(adv_images_copy)
                outputs = self.get_logits(x_bsr)
                loss = loss_fn(outputs, labels)

                g = torch.autograd.grad(
                    loss, adv_images_copy, retain_graph=False, create_graph=False
                )[0]
                grad += g

            grad = grad / float(self.num_copies)

            grad_norm = grad / (
                torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            momentum = self.decay * momentum + grad_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images