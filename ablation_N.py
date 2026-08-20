import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader



import pandas as pd

import json
import argparse
from types import SimpleNamespace

from src.data.nips_2017_dataset import NIPS2017Dataset

from src.models.model_wrapper import ModelWrapper

from src.adversarial.adversarial_generator import AdversarialGenerator

from src.adversarial.GFLA import GFLA

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
            device=device) 
        for model in models]
    
def perform_experiment(surrogate_models, target_models, loader, N, filename, targeted=False, device="cuda"):
    all_rows = []
    
    for s_model in surrogate_models:
        s_name = s_model.__class__.__name__
        
        for n in N:

            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader, 
                s_model, 
                GFLA,
                N = n,
                device=device, # type: ignore
                targeted=targeted
            )
            
            eval_models = target_models + [s_model]
            for t_model in eval_models:
                results = t_model.evaluate_attack_metrics(s_model, loader, adv_loader)

                t_name = getattr(t_model, 'name', t_model.__class__.__name__)
                
                # Append a dictionary representing one row of the table
                all_rows.append({
                    "Surrogate": s_name,
                    "Attack": "GFLA",
                    "Target": t_name,
                    "Fooling Rate": results["fooling_rate"] * 100,
                    "Transfer Rate": results["transfer_rate"] * 100,
                    "N": n
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
    
    
    perform_experiment(
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        N=config.N,
        filename=config.output
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