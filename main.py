"""
    Adversarial Attack Benchmarking Framework for PyTorch Models.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Type, Union

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Custom Module Imports
from src.adversarial.adversarial_generator import AdversarialGenerator
from src.data.nips_2017_dataset import NIPS2017Dataset
from src.models.model_wrapper import ModelWrapper

from src.adversarial.FeatureAttacks.FDA import FDA
from src.adversarial.FeatureAttacks.ILA import ILA
from src.adversarial.FeatureAttacks.FIA import FIA
from src.adversarial.FeatureAttacks.NAA import NAA
from src.adversarial.FeatureAttacks.DFAA import DFAA
from src.adversarial.FeatureAttacks.GFLA.GFLA import GFLA
from src.adversarial.FeatureAttacks.GFLA.GFLA_D import GFLA_D
from src.adversarial.FeatureAttacks.ILPD import ILPD
from src.adversarial.LogitAttacks.BSR import BSR
from src.adversarial.LogitAttacks.ANDA import ANDA
from src.adversarial.LogitAttacks.LPAA import LPAA
from src.adversarial.LogitAttacks.MIG import MIG
from src.adversarial.LogitAttacks.ResPA import ResPA
from src.adversarial.LogitAttacks.TGR import TGR
from src.adversarial.LogitAttacks.GRA import GRA
from src.adversarial.LogitAttacks.PGN import PGN
from src.adversarial.FeatureAttacks.TAP import TAP
from src.adversarial.FeatureAttacks.GFLA.GFLA_TI import GFLA_TI
from src.adversarial.FeatureAttacks.GFLA.GFLA_BSR import GFLA_BSR
from src.adversarial.FeatureAttacks.GFLA.GFLA_SI import GFLA_SI
from src.adversarial.FeatureAttacks.GFLA.GFLA_DI import GFLA_DI
from torchattacks import BIM, MIFGSM, NIFGSM, DIFGSM, TIFGSM, SINIFGSM, VMIFGSM

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def dict_to_namespace(data: Dict[str, Any]) -> SimpleNamespace:
    """Recursively converts a dictionary to a SimpleNamespace object."""
    return SimpleNamespace(
        **{
            k: dict_to_namespace(v) if isinstance(v, dict) else v
            for k, v in data.items()
        }
    )

def get_dataloader(
    batch_size: int,
    subset_size: Union[int, None],
    csv_filepath: str,
    img_dir_filepath: str,
) -> DataLoader:
    """Instantiates the dataset and returns a PyTorch DataLoader."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    nips_dataset = NIPS2017Dataset(
        csv_file=csv_filepath, 
        img_dir=img_dir_filepath, 
        transform=transform
    )

    if subset_size is not None and subset_size < len(nips_dataset):
            nips_dataset = nips_dataset.get_subset(subset_size)
    
    return DataLoader(
        nips_dataset,
        batch_size=batch_size,
        shuffle=False,
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

def save_df(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path.with_suffix(".csv"), index=False)


def perform_experiment(
    attacks: List[Type],
    surrogate_models: List[ModelWrapper],
    target_models: List[ModelWrapper],
    loader: DataLoader,
    epsilon: float,
    alpha: float,
    device: torch.device,
) -> pd.DataFrame:
    """Executes the benchmarking matrix across surrogate models, attacks, and targets."""
        
    all_rows = [] # Holds all the experimental results 

    for s_model in surrogate_models:
        s_name = getattr(s_model, "name", s_model.__class__.__name__)

        for attack_class in attacks:
            attack_name = (
                attack_class.__name__
                if isinstance(attack_class, type)
                else attack_class.__class__.__name__
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader,
                s_model.model,
                attack_class,
                device=device,
                targeted=False,
                eps=epsilon,
                alpha = alpha
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_time = time.perf_counter() - start_time

            all_target_models = target_models + [s_model]
            for t_model in all_target_models:

                t_name = getattr(t_model, "name", t_model.__class__.__name__)

                results = t_model.get_fooling_and_transfer_rate(
                    s_model, loader, adv_loader
                )

                all_rows.append(
                    {
                        "Surrogate": s_name,
                        "Attack": attack_name,
                        "Target": t_name,
                        "Fooling Rate": results["fooling_rate"] * 100,
                        "Transfer Rate": results["transfer_rate"] * 100,
                        "Wall Time (s)": wall_time,
                    }
                )

            del adv_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    return df

def main(config):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution on {device}")
    
    loader = get_dataloader(
        batch_size=config.data.batch_size, 
        subset_size=getattr(config.data, "subset_size", None), 
        csv_filepath=config.data.csv_filepath, 
        img_dir_filepath=config.data.img_dir_filepath
    )
    
    surrogates = get_models(config.surrogates, device)
    targets = get_models(config.targets, device)
    
    attacks = [globals()[name] for name in config.attacks]
    
    df = perform_experiment(
        attacks=attacks,
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        epsilon=config.eps/255,
        alpha=config.alpha/255,
        device=device
    )

    save_df(df, Path(config.output))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial Transferability Benchmarking Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = dict_to_namespace(config_dict)
        
    main(config)