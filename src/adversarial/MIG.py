import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack

class MIG(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=0.9, num_samples=20):
        super().__init__("MIG", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.num_samples = num_samples
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            IG = torch.zeros_like(adv_images)

            # b = 0
            for k in range(1, self.num_samples + 1):
                mix = (k / float(self.num_samples)) * adv_images
                mix = mix.detach().requires_grad_(True)

                outputs = self.get_logits(mix)
                cost = loss_fn(outputs, labels)

                G = torch.autograd.grad(
                    cost, mix, retain_graph=False, create_graph=False
                )[0]

                IG += G

            IG = adv_images * (IG / float(self.num_samples))

            IG_norm = IG / (torch.mean(torch.abs(IG), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + IG_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images