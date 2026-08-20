import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchattacks import BIM, MIFGSM, NIFGSM, DIFGSM, TIFGSM, SINIFGSM, VMIFGSM

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
from src.adversarial.ILPD import ILPD
from src.adversarial.GFLA_TI import GFLA_TI
from src.adversarial.BSR import BSR

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
    
def perform_experiment(attacks, surrogate_models, target_models, loader, filename, epsilon, device="cuda"):
    all_rows = []
    
    for s_model in surrogate_models:
        s_name = s_model.__class__.__name__
        
        for attack_class in attacks:
            attack_name = attack_class.__name__ if isinstance(attack_class, type) else attack_class.__class__.__name__
            
            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader, 
                s_model, 
                attack_class,
                device=device, # type: ignore
                targeted=False,
                eps=epsilon
            )
            
            eval_models = target_models + [s_model]
            for t_model in eval_models:

                results = t_model.evaluate_attack_metrics(s_model, loader, adv_loader)

                t_name = getattr(t_model, 'name', t_model.__class__.__name__)
                
                # Append a dictionary representing one row of the table
                all_rows.append({
                    "Surrogate": s_name,
                    "Attack": attack_name,
                    "Target": t_name,
                    "Fooling Rate": results["fooling_rate"] * 100,
                    "Transfer Rate": results["transfer_rate"] * 100
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
        epsilon=config.eps/255,
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