import torch
import torch.nn as nn
from torch.nn import functional as F

from .FeatureAttack import FeatureAttack

class TGR(FeatureAttack):
    def __init__(
        self,
        model,
        momentum=0.9,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        s=2.0,
        k=3,
        vit_layer_substrings=[
            "mlp.fc2",
            "norm2",
            "attn.proj",
            "attn.qkv",
            "attn.attn_drop",
        ],
        **kwargs,
    ):
        super().__init__(
            model=model,
            momentum=momentum,
            eps=eps,
            alpha=alpha,
            steps=steps,
            vit_layer_substrings=vit_layer_substrings,
            attack_name="TGR",
            **kwargs,
        )
        self.s = s
        self.k = k

    def _mlp_qkv_grad_hook(self, grad):
        """Applies gradient scaling and top-k/bottom-k zeroing across channels."""
        if grad is None:
            return grad

        g = grad.clone() * self.s

        # Handle 3D token representations [B, N, C]
        if g.dim() == 3:
            B, N, C = g.shape
            if N <= 2 * self.k:
                return g

            # Top-k and Bottom-k along token dimension N for each channel C
            _, top_idx = torch.topk(g, self.k, dim=1, largest=True)
            _, bot_idx = torch.topk(g, self.k, dim=1, largest=False)

            g.scatter_(1, top_idx, 0.0)
            g.scatter_(1, bot_idx, 0.0)

        # Handle 4D CNN tensor fallback [B, C, H, W]
        elif g.dim() == 4:
            B, C, H, W = g.shape
            N = H * W
            if N <= 2 * self.k:
                return g

            g_flat = g.flatten(2)  # [B, C, N]
            _, top_idx = torch.topk(g_flat, self.k, dim=2, largest=True)
            _, bot_idx = torch.topk(g_flat, self.k, dim=2, largest=False)

            g_flat.scatter_(2, top_idx, 0.0)
            g_flat.scatter_(2, bot_idx, 0.0)
            g = g_flat.reshape(B, C, H, W)

        return g

    def _attn_grad_hook(self, grad):
        """Applies gradient scaling and row/column zeroing on attention matrices."""
        if grad is None:
            return grad

        g = grad.clone() * self.s

        # Handle 4D Attention map tensors [B, Head, N, N]
        if g.dim() == 4:
            B, Head, N, N_col = g.shape
            if N != N_col or N <= 2 * self.k:
                return g

            # Calculate token importance by summing row and column gradient magnitudes
            token_scores = g.abs().sum(dim=3) + g.abs().sum(dim=2)  # [B, Head, N]

            _, top_tok = torch.topk(token_scores, self.k, dim=-1, largest=True)
            _, bot_tok = torch.topk(
                token_scores, self.k, dim=-1, largest=False
            )

            mask = torch.ones_like(g)
            for b in range(B):
                for h in range(Head):
                    prune_indices = torch.cat(
                        [top_tok[b, h], bot_tok[b, h]]
                    ).unique()
                    mask[b, h, prune_indices, :] = 0.0
                    mask[b, h, :, prune_indices] = 0.0

            g = g * mask

        return g

    def _get_hook(self):
        def hook(m, i, o):
            out_tensor = o[0] if isinstance(o, tuple) else o

            if out_tensor.requires_grad:
                # Distinguish attention matrix outputs [B, Head, N, N] from feature projections
                if out_tensor.dim() == 4 and out_tensor.shape[2] == out_tensor.shape[3]:
                    out_tensor.register_hook(self._attn_grad_hook)
                else:
                    out_tensor.register_hook(self._mlp_qkv_grad_hook)

            self.feature_outputs.append(out_tensor)

        return hook

    def compute_loss(self, adv_images, labels, references):
        logits = self.get_logits(adv_images)
        loss = F.cross_entropy(logits, labels)
        # Direction = 1 for Gradient Ascent (maximizing CE Loss)
        return loss, 1