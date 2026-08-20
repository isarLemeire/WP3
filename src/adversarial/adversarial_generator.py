import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from typing import List

from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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
    ) -> DataLoader:
        
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
        
        all_perturbed_images = []
        all_labels = []

        # Generate perturbed images
        for batch in tqdm(loader, desc=f"Attacking ({attack.attack})", leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            attack_labels = batch[2].to(device) if targeted else labels
            
            perturbed_images = attack(images, attack_labels)
            
            # Store on CPU to avoid filling up GPU memory
            all_perturbed_images.append(perturbed_images.cpu())
            all_labels.append(labels.cpu())

        dataset = TensorDataset(torch.cat(all_perturbed_images), torch.cat(all_labels))
        
        return DataLoader(
            dataset, 
            batch_size=loader.batch_size, 
            shuffle=False
        )