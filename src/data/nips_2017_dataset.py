import torch
from torch import Tensor
from torch.utils.data import Subset, Dataset
import pandas as pd
import os
from PIL import Image
from typing import Tuple, Callable, Optional

class NIPS2017Dataset(Dataset):
    def __init__(self, csv_file : str, img_dir : str, transform: Optional[Callable[[Image.Image], Tensor]]=None):
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"\n[!] Data Error: CSV file not found at: {os.path.abspath(csv_file)}\n"
            )
        self.data_info = pd.read_csv(csv_file)
        
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int, int]:
        # Image filenames in the CSV are just IDs (e.g., '67890')
        img_id = self.data_info.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, f"{img_id}.png")
        
        image = Image.open(img_name).convert('RGB')
        
        true_label = int(self.data_info.iloc[idx, 6]) - 1 # type: ignore
        target_label = int(self.data_info.iloc[idx, 7]) - 1 # type: ignore
        
        if self.transform:
            image = self.transform(image)

        image = torch.as_tensor(image)
            
        return image, true_label, target_label
    
    def get_subset(self, samples : int) -> Subset:
        """Create a random subset."""
        shuffled_indices = torch.randperm(len(self.data_info)).tolist()
        random_indices = shuffled_indices[:samples]

        return Subset(self, random_indices)
    
    def get_single_sample_dataset(self, index: int) -> Subset:
        if index < 0 or index >= len(self.data_info):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self.data_info)}.")
            
        return Subset(self, [index])
