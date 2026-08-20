import torch
import torch.nn as nn
from torchattacks.attack import Attack
import random

import torch
from torchattacks.attack import Attack

class FeatureAttack(Attack):
    def __init__(self, model, momentum=0.9,
                eps=8/255, alpha=2/255, steps=10,
                layer_target_types=(torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d),
                cnn_layer_types=(torch.nn.Conv2d, torch.nn.BatchNorm2d),
                vit_layer_substrings=['mlp.fc2', 'norm2', 'attn.proj', 'attn.qkv', 'pool'],
                attack_name="FeatureAttack",
                lower_percentile = 0.0, upper_percentile = 1.0):
        super().__init__(attack_name, model)

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        
        # Extraction configuration
        self.layer_target_types = layer_target_types
        self.cnn_layer_types = cnn_layer_types
        self.vit_layer_substrings = vit_layer_substrings


        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.feature_outputs = []
        self.layer_registry = self._select_layers()



    def _select_layers(self):
        """
        Extracts layers based on instance-specific architectural components.
        Selection logic: (Type ∈ cnn_types OR Name ∈ vit_substrings) AND Type ∈ target_types
        """

        self.model.eval()
        candidate_modules = []       

        for n, m in self.model.named_modules():
            n_lower = n.lower()

            is_allowed_type = isinstance(m, self.layer_target_types)
            is_cnn_match = isinstance(m, self.cnn_layer_types)
            is_vit_match = any(sub in n_lower for sub in self.vit_layer_substrings)

            if is_allowed_type and (is_cnn_match or is_vit_match):
                candidate_modules.append((n, m))

        total = len(candidate_modules)
        if total == 0:
            return []
        
        if total == 1:
            return [m for (_, m) in candidate_modules]
        
        selected_modules = []
        for idx, (n, m) in enumerate(candidate_modules):
            # Compute relative depth percentile (0.0 to 1.0)
            relative_depth = idx / (total - 1)
            
            if self.lower_percentile <= relative_depth <= self.upper_percentile:
                selected_modules.append(m)
                
        return selected_modules

    def _get_hook(self):
        def hook(m, i, o):
            self.feature_outputs.append(o[0] if isinstance(o, tuple) else o)
        return hook

    def _get_features(self, input_t):      
        handles = []
        self.feature_outputs.clear()

        try:
            for mod in self.layer_registry:
                handles.append(mod.register_forward_hook(self._get_hook()))
            with torch.no_grad():
                self.get_logits(input_t)

        finally:
            for h in handles: h.remove()

        return [f.detach() for f in self.feature_outputs]

    def setup_references(self, images, labels):
        with torch.no_grad():
            return self._get_features(images)

    def compute_loss(self, adv_images, labels, references):
        raise NotImplementedError

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        # 1. Setup static references
        references = self.setup_references(images, labels)

        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).to(self.device)

        # 2. Permanent Hooks
        all_handles = []
        for mod in self.layer_registry:
            all_handles.append(mod.register_forward_hook(self._get_hook()))

        # 3. Optimization Loop
        for _ in range(self.steps):
            adv_images.requires_grad = True
            self.feature_outputs.clear()
            
            total_loss, direction = self.compute_loss(adv_images, labels, references)
            
            grad = torch.autograd.grad(total_loss, adv_images)[0]
            grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)
            momentum = self.momentum * momentum + grad

            # direction: 1 for Ascent, -1 for Descent
            adv_images = adv_images.detach() + (direction * self.alpha * momentum.sign())
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1)

        for h in all_handles: h.remove()
        return adv_images