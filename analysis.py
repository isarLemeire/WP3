import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import pandas as pd

import json
import argparse
from types import SimpleNamespace

from src.data.nips_2017_dataset import NIPS2017Dataset

from src.models.model_wrapper import ModelWrapper

from src.adversarial.adversarial_generator import AdversarialGenerator

from src.adversarial.FDA import FDA
from src.adversarial.ILA import ILA
from src.adversarial.FIA import FIA
from src.adversarial.NAA import NAA
from src.adversarial.DFAA import DFAA
from src.adversarial.GFLA import GFLA
from src.adversarial.GFLA_D import GFLA_D


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

def extract_model_layers(
        model, 
        layer_target_types=(torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d),
        cnn_layer_types=(torch.nn.Conv2d, torch.nn.BatchNorm2d),
        vit_layer_substrings=['mlp.fc2', 'norm2', 'attn.proj', 'attn.qkv', 'pool']
    ):
    model.eval()
    candidate_modules = []
    
    for n, m in model.named_modules():
        n_lower = n.lower()
        
        is_allowed_type = isinstance(m, layer_target_types)
        is_cnn_match = isinstance(m, cnn_layer_types)
        is_vit_match = any(sub in n_lower for sub in vit_layer_substrings)

        if is_allowed_type and (is_cnn_match or is_vit_match):
            candidate_modules.append({'name': n, 'mod': m})
    
    return candidate_modules

def evaluate_feature_disruption(model, clean_loader, perturbed_loader, device=torch.device("cuda")):
    layers = extract_model_layers(model)
    if not layers:
        return 0.0, 0.0, 0.0
        
    layer_mods = [l['mod'] for l in layers]
    num_layers = len(layer_mods)
    
    total_rfd = 0.0
    total_cosine = 0.0
    total_proj = 0.0
    total_batches = 0

    for (x_clean, y, _), (x_adv, _) in zip(clean_loader, perturbed_loader):
        x_clean, x_adv, y = x_clean.to(device), x_adv.to(device), y.to(device)
        
        feature_outputs = []
        def hook_fn(module, input, output):
            if output.requires_grad:
                output.retain_grad()
            feature_outputs.append(output)

        # --- PASS 1: Clean Forward + Backward ---
        x_clean.requires_grad = True
        handles = [mod.register_forward_hook(hook_fn) for mod in layer_mods]
        
        model.zero_grad()
        logits = model(x_clean)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        clean_feats = [f.detach().clone() for f in feature_outputs]
        fixed_grads = [f.grad.detach().clone() if f.grad is not None else torch.zeros_like(f) 
                       for f in feature_outputs]
        
        for h in handles: h.remove()
        feature_outputs = []

        # --- PASS 2: Adversarial Forward ---
        handles = [mod.register_forward_hook(hook_fn) for mod in layer_mods]
        with torch.no_grad():
            model(x_adv)
        adv_feats = [f.detach().clone() for f in feature_outputs]
        
        for h in handles: h.remove()

        # --- PASS 3: Metric Calculation ---
        batch_layer_rfd = 0.0
        batch_layer_cosine = 0.0
        batch_layer_proj = 0.0
        
        for f_c, f_a, g in zip(clean_feats, adv_feats, fixed_grads):
            flat_displacement = (f_a - f_c).flatten(1)
            flat_target_grad = g.flatten(1)
            
            # 1. Cosine Similarity
            sim = F.cosine_similarity(flat_displacement, flat_target_grad, dim=1)
            batch_layer_cosine += sim.mean().item()
            
            # 2. Distance Along Gradient (Projection onto Unit Gradient)
            inner_product = (flat_displacement * flat_target_grad).sum(dim=1)
            grad_norm = torch.norm(flat_target_grad, p=2, dim=1)
            proj = inner_product / (grad_norm + 1e-8)
            batch_layer_proj += proj.mean().item()
            
            # 3. Relative Feature Disruption (RFD)
            diff_sq = (f_a - f_c) ** 2
            rms_displacement = torch.sqrt(diff_sq.flatten(1).mean(dim=1))
            
            clean_sq = f_c ** 2
            rms_clean = torch.sqrt(clean_sq.flatten(1).mean(dim=1))
            
            rfd_per_sample = (rms_displacement / (rms_clean + 1e-8)) ** 0.5
            batch_layer_rfd += rfd_per_sample.mean().item()

        total_rfd += (batch_layer_rfd / num_layers)
        total_cosine += (batch_layer_cosine / num_layers)
        total_proj += (batch_layer_proj / num_layers)
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0

    return (total_rfd / total_batches, 
            total_cosine / total_batches, 
            total_proj / total_batches)
    
def perform_experiment(attacks, surrogate_models, target_models, loader, filename, device: torch.device=torch.device("cuda")):
    all_rows = []


    for s_model in surrogate_models:
        s_model.to(device)
        s_model.eval()
        s_name = s_model.__class__.__name__
        
        for attack_class in attacks:
            attack_name = attack_class.__name__ if isinstance(attack_class, type) else attack_class.__class__.__name__
            
            # 1. Generate Perturbed Dataset
            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader, 
                s_model, 
                attack_class,
                device=device,
            )
            
            # Define models to evaluate against (Target models + White-box surrogate)
            eval_models = target_models + [s_model]


            for model in eval_models:
                m_name = getattr(model, 'name', model.__class__.__name__)
                model.to(device)
                model.eval()
                
                # 2. Get Fooling Rate
                displacement, alignment, projection = evaluate_feature_disruption(
                    model=model, 
                    clean_loader=loader, 
                    perturbed_loader = adv_loader, 
                    device=device) # type: ignore
                
                results = model.evaluate_attack_metrics(s_model, loader, adv_loader)

                
                # Append data
                all_rows.append({
                    "Surrogate": s_name,
                    "Attack": attack_name,
                    "Target": m_name,
                    "Fooling Rate": results["fooling_rate"] * 100,
                    "Transfer Rate": results["transfer_rate"] * 100,
                    
                    "CE Loss": results["loss"], 
                    "Feature disruption distance": results["feature_distance"],
                    "Cosine alignment" : results["cosine_alignment"],
                    "Feature projection": results["projection"],
                    "Loss over displacement":  results["loss_over_feature_distance"],
                })

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
    
    attacks = [globals()[name] for name in config.attacks]
    
    perform_experiment(
        attacks=attacks,
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        filename=config.output,
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