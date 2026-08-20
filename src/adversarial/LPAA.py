import torch
import torch.nn as nn
from torchattacks.attack import Attack


class LPAA(Attack):
    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        momentum=0.9,
        masking_ratio=0.95,
        beta=35,
        num_samples=5,
        eta=3.0,
        init_steps=5,
    ):
        super().__init__("LPAA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.masking_ratio = masking_ratio
        self.beta = beta
        self.num_samples = num_samples
        self.eta = eta
        self.init_steps = init_steps
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        perturbation = torch.zeros_like(images)
        momentum_buf = torch.zeros_like(images)

        # -------------------------------------------------------------------
        # Phase 1: MIM Warm-Start Initialization
        # -------------------------------------------------------------------
        for _ in range(self.init_steps):
            adv = (images + perturbation).detach().requires_grad_(True)
            outputs = self.get_logits(adv)
            cost = loss_fn(outputs, labels)

            g = torch.autograd.grad(
                cost, adv, retain_graph=False, create_graph=False
            )[0]
            g_norm = g / (
                torch.mean(torch.abs(g), dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            momentum_buf = self.momentum * momentum_buf + g_norm

            perturbation = perturbation + self.alpha * momentum_buf.sign()
            perturbation = torch.clamp(
                perturbation, min=-self.eps, max=self.eps
            )
            perturbation = (
                torch.clamp(images + perturbation, min=0.0, max=1.0) - images
            )

        # Projection to eta * eps L_inf ball & Momentum Reset
        perturbation = torch.clamp(
            perturbation, min=-self.eta * self.eps, max=self.eta * self.eps
        )
        momentum_buf = torch.zeros_like(images)

        # -------------------------------------------------------------------
        # Phase 2: LPAA Stochastic Masking Refinement
        # -------------------------------------------------------------------
        for _ in range(self.steps):
            g_accum = torch.zeros_like(perturbation)

            for _ in range(self.num_samples):
                # Random binary mask matrix
                mask = (
                    torch.rand_like(perturbation) > self.masking_ratio
                ).float()
                mix = images + (self.beta * perturbation) * mask
                mix = (
                    torch.clamp(mix, min=0.0, max=1.0)
                    .detach()
                    .requires_grad_(True)
                )

                outputs = self.get_logits(mix)
                cost = loss_fn(outputs, labels)

                g = torch.autograd.grad(
                    cost, mix, retain_graph=False, create_graph=False
                )[0]
                g_accum += g

            g_avg = g_accum / float(self.num_samples)
            g_norm = g_avg / (
                torch.mean(torch.abs(g_avg), dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            momentum_buf = self.momentum * momentum_buf + g_norm

            perturbation = (
                perturbation.detach() + self.alpha * momentum_buf.sign()
            )
            perturbation = torch.clamp(
                perturbation, min=-self.eps, max=self.eps
            )
            perturbation = (
                torch.clamp(images + perturbation, min=0.0, max=1.0) - images
            )

        return images + perturbation