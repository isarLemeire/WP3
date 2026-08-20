import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


import pandas as pd
import gc
import json
import argparse
from types import SimpleNamespace
from typing import Optional, List, Type

from src.data.nips_2017_dataset import NIPS2017Dataset

from src.models.model_wrapper import ModelWrapper


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class FeatureAttack(Attack):
    def __init__(self, model, momentum=0.9,
                eps=8/255, alpha=2/255, steps=10,
                layer_target_types=(torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d),
                cnn_layer_types=(torch.nn.Conv2d, torch.nn.BatchNorm2d),
                vit_layer_substrings=['mlp.fc2', 'norm2', 'attn.proj', 'attn.qkv', 'pool'],
                attack_name="FeatureAttack"):
        super().__init__(attack_name, model)

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        
        # Extraction configuration
        self.layer_target_types = layer_target_types
        self.cnn_layer_types = cnn_layer_types
        self.vit_layer_substrings = vit_layer_substrings       

        self.feature_outputs = []
        self.layer_registry = self._select_layers()
        
        self.trajectory = []

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
        
        return [m for (_, m) in candidate_modules]

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
        
        self.trajectory.clear()

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
            
            self.trajectory.append(adv_images.detach().cpu())
            
        for h in all_handles: h.remove()
        return adv_images
    
class GFLA(FeatureAttack):
    def __init__(self, *args, lam=0.9, N=5, beta=4.5, **kwargs):
        super().__init__(attack_name="GFLA", *args, **kwargs)
        self.lam = lam
        self.N = N  # Neighborhood samples
        self.beta = beta # Noise scale factor

    def _get_vmi_sample(self, images):
        """Generates a noisy neighbor for VMI sampling."""
        images_det = images.detach()
        if self.N > 0:
            noise = (torch.rand_like(images_det) * 2 - 1) * (self.eps * self.beta)
            return (images_det + noise).clamp(0, 1).requires_grad_(True)
        return images_det.clone().requires_grad_(True)

    def setup_references(self, images, labels): # type: ignore
        """
        Pre-computes clean features and averaged feature-space gradients.
        """
        
        # Clean features
        clean_feats = self._get_features(images)

        # Adversarial gradient (VMI-style)
        fixed_grads = []
        num_samples = max(1, self.N)       

        for _ in range(num_samples):
            x_neighbor = self._get_vmi_sample(images)
        
            self.feature_outputs.clear()

            handles = [mod.register_forward_hook(self._get_hook())
                        for mod in self.layer_registry]
           
            logits = self.get_logits(x_neighbor)
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, self.feature_outputs)

            if not fixed_grads:
                fixed_grads = [g.detach() / num_samples for g in grads]
            else:
                for i, g in enumerate(grads):
                    fixed_grads[i] += g.detach() / num_samples

            for h in handles: h.remove()

        return {
            'clean_feats': clean_feats,
            'fixed_grads': fixed_grads
        }

    def compute_loss(self, adv_images, labels, references):
        batch_size = adv_images.shape[0]
        clean_feats = references['clean_feats']
        fixed_grads = references['fixed_grads']

        self.get_logits(adv_images)

        layer_loss = 0
        num_layers = len(self.layer_registry)

        for f_adv, f_clean, g in zip(
            self.feature_outputs,
            clean_feats,
            fixed_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            proj = (delta_f * g_unit).sum(dim=1)
            
            dist = torch.norm(delta_f, p=2, dim=1)

            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = (layer_loss / (num_layers + 1e-8))

        return loss, 1
    
class GFLA_D(FeatureAttack):
    def __init__(self, *args, lam=0.9, N=5, beta=4.5, **kwargs):
        """
        Dynamic GFLA that recalculates feature-space gradients at every step.
        """
        super().__init__(attack_name="GFLA_D", *args, **kwargs)
        self.lam = lam 
        self.N = N  
        self.beta = beta 

    def _get_vmi_sample(self, images):
        """Generates a noisy neighbor for VMI sampling."""
        images_det = images.detach() 
        if self.N > 0:
            noise = (torch.rand_like(images_det) * 2 - 1) * (self.eps * self.beta)
            return (images_det + noise).clamp(0, 1).requires_grad_(True)
        return images_det.clone().requires_grad_(True)

    def _compute_step_gradients(self, adv_images, labels):
        """
        Computes the current step's feature-space gradients using VMI neighborhood sampling.
        Uses the base class's permanent hooks to avoid duplicate registration.
        """
        step_grads = []
        num_samples = max(1, self.N)
        
        for _ in range(num_samples):
            x_neighbor = self._get_vmi_sample(adv_images)
            
            # Clear the outputs cache to capture EXACTLY this neighbor pass
            self.feature_outputs.clear()
            
            # Forward pass automatically triggers the base class permanent hooks
            logits = self.get_logits(x_neighbor)
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, self.feature_outputs)
            
            if not step_grads:
                step_grads = [g.detach() / num_samples for g in grads]
            else:
                for i, g in enumerate(grads):
                    step_grads[i] += g.detach() / num_samples
                
        return step_grads

    def compute_loss(self, adv_images, labels, references):
        batch_size = adv_images.shape[0]

        current_step_grads = self._compute_step_gradients(adv_images, labels)

        self.feature_outputs.clear()
        self.get_logits(adv_images)
        
        layer_loss = 0
        num_layers = len(self.layer_registry)
        
        for f_adv, f_clean, g in zip(
            self.feature_outputs, 
            references, 
            current_step_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            # Guided component
            proj = (delta_f * g_unit).sum(dim=1)
            # Unguided component
            dist = torch.norm(delta_f, p=2, dim=1)

            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = (layer_loss / (num_layers + 1e-8))
        return loss, 1
    
class AdversarialGenerator:
    @staticmethod
    def generate_perturbed_dataset(
        loader: DataLoader, 
        model: nn.Module, 
        attack_class: type,
        device: torch.device = torch.device('cpu'),
        targeted: bool = False,
        dataset_mean:List = IMAGENET_MEAN,
        dataset_std:List = IMAGENET_STD,
        **attack_kwargs 
    ) -> list[DataLoader]:
        
        # Prepare model
        model.eval()
        model.to(device)       
        
        # Prepare attack
        attack = attack_class(model, **attack_kwargs)
        attack.set_normalization_used(
            mean=dataset_mean,
            std=dataset_std
        )
        attack.set_device(device)

        if targeted:
            attack.set_mode_targeted_by_label()
        else:
            attack.set_mode_default()
        
        step_accumulators = [[] for _ in range(attack.steps)]
        all_labels = []

        for batch in tqdm(loader, desc=f"Tracking Trajectory ({attack.attack})", leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            attack_labels = batch[2].to(device) if targeted else labels
            
            # This populates attack.trajectory implicitly
            _ = attack(images, attack_labels)
            
            # Harvest states across the entire time horizon T
            for t in range(attack.steps):
                step_accumulators[t].append(attack.trajectory[t])
            
            all_labels.append(labels.cpu())
        
        trajectory_loaders = []
        flat_labels = torch.cat(all_labels)
        
        for t in range(attack.steps):
            step_images = torch.cat(step_accumulators[t])
            step_dataset = TensorDataset(step_images, flat_labels)
            step_loader = DataLoader(
                step_dataset, 
                batch_size=loader.batch_size, 
                shuffle=False
            )
            trajectory_loaders.append(step_loader)
            
        return trajectory_loaders

def get_loader(batch_size, subset_size, csv_filepath, img_dir_filepath):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    nips_dataset = NIPS2017Dataset(
        csv_file=csv_filepath, 
        img_dir=img_dir_filepath, 
        transform=transform
    )

    if subset_size is None or subset_size >= 1000:
        return DataLoader(
            nips_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    else:
        return DataLoader(
            nips_dataset.get_subset(subset_size), 
            batch_size=batch_size, 
            shuffle=False
        )
        
def get_models(models, device):
    return [
        ModelWrapper.load(
            name=model.get("name"), 
            timm_string = model.get("timm_string"), 
            path=model.get("path"), 
            input_size=model.get("input_size", 224), 
            ViT=model.get("ViT", False),
            device=device) 
        for model in models]

def perform_experiment(loader, attacks, surrogate_models, target_models, filename,  steps = 20, device: torch.device=torch.device("cuda")):
    all_rows = []
    for s_model in surrogate_models:
        
        s_name = getattr(s_model, 'name', s_model.__class__.__name__)
        s_model.to(device)
        s_model.eval()
        
        eval_models = target_models + [s_model]

        for attack_class in attacks:
            attack_name = attack_class.__name__ if isinstance(attack_class, type) else attack_class.__class__.__name__
            
            loaders = AdversarialGenerator.generate_perturbed_dataset(
                    loader, 
                    s_model, 
                    attack_class,
                    lam = 1,
                    steps = steps,
                    device=device
                )
            eval_loaders = [loader] + loaders
            
            for i, current_loader in tqdm(enumerate(eval_loaders)):
                for t_model in eval_models:
                    t_name = getattr(t_model, 'name', t_model.__class__.__name__)
                    t_model.to(device)
                    t_model.eval()
                    
                    results = t_model.evaluate_attack_metrics(s_model, loader, current_loader)
                    
                    all_rows.append({
                        "Surrogate": s_name,
                        "Attack": attack_name,
                        "Target": t_name,
                        "CE loss": results["loss"],
                        "Fooling Rate": results["fooling_rate"] * 100,
                        "Transfer Rate": results["transfer_rate"] * 100,
                        "step": i
                    })
                    
                    if t_model != s_model:
                        t_model.to("cpu")
                        
            del loaders
            del eval_loaders
            
        s_model.to("cpu")
        torch.cuda.empty_cache()
        
    df = pd.DataFrame(all_rows)
    
    if filename.endswith('.csv'):
        df.to_csv(filename, index=False)
    elif filename.endswith('.xlsx'):
        df.to_excel(filename, index=False)
            
    return df


def main(config):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    loader = get_loader(
        batch_size=config.data.batch_size, 
        subset_size=config.data.subset_size, 
        csv_filepath=config.data.csv_filepath, 
        img_dir_filepath=config.data.img_dir_filepath
    )
    
    surrogates = get_models(config.surrogates, device)
    targets = get_models(config.targets, device)
    
    
    perform_experiment(
        attacks=[GFLA, GFLA_D],
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        filename=config.output,
        steps = config.steps,
        device=device # type: ignore
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # Convert nested dicts into SimpleNamespace objects
    def to_namespace(d):
        return SimpleNamespace(**{
            k: to_namespace(v) if isinstance(v, dict) else v
            for k, v in d.items()
        })
        
    config = to_namespace(config_dict)
        
    main(config)