import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack

class PGN(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=0.9, beta=3.0, gamma=0.5, num_samples=5):
        super().__init__("PGN", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.num_samples = num_samples
        self.beta = beta
        self.gamma = gamma
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            g_flat = torch.zeros_like(adv_images)

            for _ in range(self.num_samples):

                noise_rand = torch.zeros_like(adv_images).uniform_(
                                    -self.eps * self.beta, self.eps * self.beta
                                )
                x1 = (adv_images + noise_rand).detach().requires_grad_(True)

                output1 = self.get_logits(x1)
                ce_loss1 = loss_fn(output1, labels)
                g1 = torch.autograd.grad(
                    ce_loss1, x1, retain_graph=False, create_graph=False
                )[0]
                g1_norm = g1 / (torch.mean(torch.abs(g1), dim=(1, 2, 3), keepdim=True) + 1e-8)

                x_star = x1 - self.alpha * g1_norm
                output_star = self.get_logits(x_star)
                ce_loss_star = loss_fn(output_star, labels)
                g_star = torch.autograd.grad(
                    ce_loss_star, x_star, retain_graph=False, create_graph=False
                )[0]

                g_flat += ((1-self.gamma)*g1 + self.gamma*g_star)

            g_flat = g_flat / self.num_samples

            g_flat_norm = g_flat / (torch.mean(torch.abs(g_flat), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + g_flat_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images