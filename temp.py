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
import src.adversarial as custom_attacks
from src.adversarial.adversarial_generator import AdversarialGenerator
from src.data.nips_2017_dataset import NIPS2017Dataset
from src.models.model_wrapper import ModelWrapper
import torchattacks

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    nips_dataset = NIPS2017Dataset(
        csv_file=csv_filepath,
        img_dir=img_dir_filepath,
        transform=transform,
    )

    if subset_size is not None and subset_size < len(nips_dataset):
        nips_dataset = nips_dataset.get_subset(subset_size)

    return DataLoader(
        nips_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def load_wrapped_models(
    model_configs: List[SimpleNamespace], device: torch.device
) -> List[ModelWrapper]:
    """Loads and wraps target/surrogate architectures."""
    loaded_models = []
    for cfg in model_configs:
        model = ModelWrapper.load(
            name=getattr(cfg, "name", None),
            timm_string=getattr(cfg, "timm_string", None),
            path=getattr(cfg, "path", None),
            input_size=getattr(cfg, "input_size", 224),
            ViT=getattr(cfg, "ViT", False),
            device=device,
        )
        loaded_models.append(model)
    return loaded_models


def resolve_attack_classes(attack_names: List[str]) -> List[Type]:
    """Resolves attack class names from custom modules or torchattacks."""
    resolved = []
    for name in attack_names:
        if hasattr(custom_attacks, name):
            resolved.append(getattr(custom_attacks, name))
        elif hasattr(torchattacks, name):
            resolved.append(getattr(torchattacks, name))
        else:
            raise AttributeError(
                f"Attack '{name}' not found in custom attacks or torchattacks."
            )
    return resolved


def perform_experiment(
    attacks: List[Type],
    surrogate_models: List[ModelWrapper],
    target_models: List[ModelWrapper],
    loader: DataLoader,
    output_path: Path,
    epsilon: float,
    alpha: float,
    device: torch.device,
) -> pd.DataFrame:
    """Executes the benchmarking matrix across surrogate models, attacks, and targets."""
    
    all_rows = [] 

    for s_model in surrogate_models:
        s_name = getattr(s_model, "name", s_model.__class__.__name__)
        logger.info(f"Evaluating Surrogate Architecture: {s_name}")

        for attack_class in attacks:
            attack_name = (
                attack_class.__name__
                if isinstance(attack_class, type)
                else attack_class.__class__.__name__
            )
            logger.info(f"Running Attack Generation: {attack_name}")

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            # Generate perturbed dataset
            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader,
                s_model.model,
                attack_class,
                device=device,
                targeted=False,
                eps=epsilon,
                alpha=alpha,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_time = time.perf_counter() - start_time

            eval_models = target_models + [s_model]
            for t_model in eval_models:
                t_name = getattr(t_model, "name", t_model.__class__.__name__)

                results = t_model.get_fooling_and_transfer_rate(
                    s_model, loader, adv_loader
                )

                all_rows.append(
                    {
                        "Surrogate": s_name,
                        "Attack": attack_name,
                        "Target": t_name,
                        "Fooling Rate (%)": round(results["fooling_rate"] * 100, 2),
                        "Transfer Rate (%)": round(results["transfer_rate"] * 100, 2),
                        "Wall Time (s)": round(wall_time, 4),
                    }
                )

            del adv_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        # Default to CSV if no supported extension provided
        df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"Results successfully written to: {output_path}")
    return df


def main() -> None:
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

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution initialized on device: {device}")

    loader = get_dataloader(
        batch_size=config.data.batch_size,
        subset_size=getattr(config.data, "subset_size", None),
        csv_filepath=config.data.csv_filepath,
        img_dir_filepath=config.data.img_dir_filepath,
    )

    surrogates = load_wrapped_models(config.surrogates, device)
    targets = load_wrapped_models(config.targets, device)
    attacks = resolve_attack_classes(config.attacks)

    perform_experiment(
        attacks=attacks,
        surrogate_models=surrogates,
        target_models=targets,
        loader=loader,
        output_path=Path(config.output),
        epsilon=config.eps / 255.0,
        alpha=config.alpha / 255.0,
        device=device,
    )


if __name__ == "__main__":
    main()