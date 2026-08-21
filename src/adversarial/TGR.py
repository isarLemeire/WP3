import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from functools import partial

from torchattacks.attack import Attack

class TGR(Attack):
    def __init__(self, model, eps=16 / 255, steps=10, decay=1.0,
                 attn_gamma=0.25, qkv_gamma=0.75, mlp_gamma=0.5,
                 targeted=False):
        super().__init__("TGR", model)
        self.eps = eps
        self.steps = steps
        self.step_size = eps / steps
        self.decay = decay
        self.attn_gamma = attn_gamma
        self.qkv_gamma = qkv_gamma
        self.mlp_gamma = mlp_gamma
        self.targeted = targeted
        self.loss_flag = -1 if targeted else 1

        self._hook_handles = []
        self._register_hooks()

    def _attn_tgr(self, module, grad_in, grad_out, gamma):
        out_grad = grad_in[0].clone() * gamma
        B, C, H, W = out_grad.shape  # C = num_heads, H = W = num_tokens
        out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
        max_all_H, max_all_W = max_all // H, max_all % H
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
        min_all_H, min_all_W = min_all // H, min_all % H

        out_grad[:, range(C), max_all_H, :] = 0.0
        out_grad[:, range(C), :, max_all_W] = 0.0
        out_grad[:, range(C), min_all_H, :] = 0.0
        out_grad[:, range(C), :, min_all_W] = 0.0
        return (out_grad,)

    def _qkv_tgr(self, module, grad_in, grad_out, gamma):
        """Hooked on attn.qkv -- grad_in[0] is the gradient w.r.t. the qkv
        Linear's INPUT (token embeddings), shape [B, N, C]. NOT the QKV
        projection's output; nothing here isolates V specifically."""
        out_grad = grad_in[0].clone() * gamma
        c = out_grad.shape[2]
        out_grad_cpu = out_grad.data.clone().cpu().numpy()

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)  # per-channel extreme token
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
        out_grad[:, max_all, range(c)] = 0.0
        out_grad[:, min_all, range(c)] = 0.0
        return (out_grad,) + tuple(grad_in[1:])

    def _mlp_tgr(self, module, grad_in, grad_out, gamma):
        """Hooked on mlp -- grad_in[0] is the gradient w.r.t. the mlp
        block's INPUT, shape [B, N, C]."""
        out_grad = grad_in[0].clone() * gamma
        c = out_grad.shape[2]
        out_grad_cpu = out_grad.data.clone().cpu().numpy()

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
        out_grad[:, max_all, range(c)] = 0.0
        out_grad[:, min_all, range(c)] = 0.0
        return (out_grad,) + tuple(grad_in[1:])

    def _register_hooks(self):
        attn_hook = partial(self._attn_tgr, gamma=self.attn_gamma)
        qkv_hook = partial(self._qkv_tgr, gamma=self.qkv_gamma)
        mlp_hook = partial(self._mlp_tgr, gamma=self.mlp_gamma)

        for block in self.model.blocks:
            self._hook_handles.append(
                block.attn.attn_drop.register_full_backward_hook(attn_hook)
            )
            self._hook_handles.append(
                block.attn.qkv.register_full_backward_hook(qkv_hook)
            )
            self._hook_handles.append(
                block.mlp.register_full_backward_hook(mlp_hook)
            )

    def remove_hooks(self):
        """Not called automatically -- the reference registers hooks once
        for the lifetime of the attack object too. Call this explicitly if
        you need to reuse the same underlying model without TGR's gradient
        surgery active afterward."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------ #
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(images).detach()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad_(True)
            outputs = self.get_logits(adv_images)
            cost = self.loss_flag * loss_fn(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.step_size * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images