"""
Dataset pipeline for the NIPS 2017 Adversarial Attacks Competition benchmark dataset.
"""
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Subset, Dataset
import pandas as pd
import os
from PIL import Image
from typing import Callable, Optional, Tuple, Union

class NIPS2017Dataset(Dataset):
    """
    Dataset wrapper for NIPS 2017 Adversarial Competition images and annotations.
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        img_dir: Union[str, Path],
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        if not self.csv_file.is_file():
            raise FileNotFoundError(
                f"\n[!] Data Error: CSV file not found at: {self.csv_file.resolve()}\n"
            )

        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"\n[!] Data Error: Image directory not found at: {self.img_dir.resolve()}\n"
            )

        self.data_info = pd.read_csv(self.csv_file)

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int, int]:
        # Image filenames in the CSV are just IDs (e.g., '67890')
        img_id = str(self.data_info.iloc[idx, 0])
        img_path = self.img_dir / f"{img_id}.png"

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        with Image.open(img_path) as img:
            image = Image.open(img_path).convert('RGB')
        
        true_label = int(self.data_info.iloc[idx, 6]) - 1  # type: ignore
        target_label = int(self.data_info.iloc[idx, 7]) - 1 # type: ignore
        
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.as_tensor(image)
            
        return image_tensor, true_label, target_label
    
    def get_subset(self, samples: int, seed: Optional[int] = None) -> Subset:
        """Create a random subset."""

        if samples > len(self):
            samples = len(self)

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            shuffled_indices = torch.randperm(len(self), generator=generator).tolist()
        else:
            shuffled_indices = torch.randperm(len(self)).tolist()

        random_indices = shuffled_indices[:samples]
        return Subset(self, random_indices)
    
    def get_single_sample_dataset(self, index: int) -> Subset:
        """Extracts a single-sample Subset for testing"""
        if index < 0 or index >= len(self.data_info):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self.data_info)}.")
            
        return Subset(self, [index])
