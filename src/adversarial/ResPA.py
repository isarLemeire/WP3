import torch
import torch.nn as nn
from torchattacks.attack import Attack

class ResPA(Attack):
    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=0.9,
        N=5,
        rho=0.5,
        gamma=0.6,
        theta=0.6,
        beta=1.5,
    ):
        super().__init__("ResPA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.N = N
        self.rho = rho
        self.gamma = gamma
        self.theta = theta
        self.beta = beta
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        e = torch.zeros_like(images).detach().to(self.device)
        momentum = torch.zeros_like(images).detach().to(self.device)

        for _ in range(self.steps):
            avg_grad = torch.zeros_like(images)

            for _ in range(self.N):
                noise_rand = torch.zeros_like(adv_images).uniform_(
                    -self.eps * self.beta, self.eps * self.beta
                )
                x_near = (adv_images + noise_rand).detach().requires_grad_(True)

                output = self.get_logits(x_near)
                ce_loss = loss_fn(output, labels)
                g1 = torch.autograd.grad(
                    ce_loss, x_near, retain_graph=False, create_graph=False
                )[0]

                g_res = g1 - e
                g_res_norm = g_res / (
                    torch.mean(torch.abs(g_res), dim=(1, 2, 3), keepdim=True) + 1e-8
                )

                x_star = (x_near.detach() - self.rho * g_res_norm).requires_grad_(True)

                outputs_star = self.get_logits(x_star)
                loss_star = loss_fn(outputs_star, labels)
                g2 = torch.autograd.grad(
                    loss_star, x_star, retain_graph=False, create_graph=False
                )[0]

                avg_grad += (1.0 - self.gamma) * g1 + self.gamma * g2

            avg_grad = avg_grad / float(self.N)

            e = self.theta * e + (1.0 - self.theta) * avg_grad

            grad_norm = avg_grad / (
                torch.mean(torch.abs(avg_grad), dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            momentum = self.decay * momentum + grad_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta_img = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta_img, min=0.0, max=1.0).detach()

        return adv_images