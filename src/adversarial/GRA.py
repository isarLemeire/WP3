import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack


class GRA(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=0.9, beta=3.5, eta_m=0.94, num_samples=20):
        super().__init__("GRA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.beta = beta
        self.eta_m = eta_m
        self.num_samples = num_samples
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        M = torch.full_like(images, fill_value=1.0 / self.eta_m).detach().to(self.device)
        prev_sign = torch.zeros_like(images).detach().to(self.device)

        for _ in range(self.steps):

            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost = loss_fn(outputs, labels)
            
            G = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # G_flat = E_{\eta ~ U(-\beta, \beta)} [\nabla L(x + \eta \cdot \epsilon, y)]
            G_flat = torch.zeros_like(adv_images)
            for _ in range(self.num_samples):
                eta = torch.empty_like(adv_images).uniform_(-self.beta, self.beta)
                neighbor = torch.clamp(adv_images + eta * self.eps, min=0.0, max=1.0).detach()
                neighbor.requires_grad = True
                
                outputs_n = self.get_logits(neighbor)
                cost_n = loss_fn(outputs_n, labels)
                grad_n = torch.autograd.grad(
                    cost_n, neighbor, retain_graph=False, create_graph=False
                )[0]
                G_flat += grad_n
                
            G_flat = G_flat / self.num_samples

            # s = cosine(G, G_flat)
            G_vec = G.reshape(batch_size, -1)
            G_flat_vec = G_flat.reshape(batch_size, -1)
            s = F.cosine_similarity(G_vec, G_flat_vec, dim=1, eps=1e-8).reshape(batch_size, 1, 1, 1)

            # WG = s * G + (1 - s) * G_flat
            WG = s * G + (1.0 - s) * G_flat

            # g_{t+1} = \mu * g_t + WG / ||WG||_1
            WG_norm = WG / (torch.mean(torch.abs(WG), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + WG_norm
            curr_sign = momentum.sign()

            # Decay Indicator
            M_e = (curr_sign == prev_sign).float()
            M_d = 1.0 - M_e
            M = M * (M_e + self.eta_m * M_d)

            prev_sign = curr_sign.clone().detach()

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images