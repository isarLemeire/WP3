"""
Model Wrapper Utility for Adversarial Evaluation.
"""

from typing import Any, Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import rename_model

class ModelWrapper(nn.Module):
    """
    Standardized wrapper around PyTorch/timm architectures to handle dynamic 
    input resizing and evaluation metrics (FR, TR, Loss).
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        name: str = "model",
        input_size: int = 224,
        ViT: bool = False,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.model_kwargs = model_kwargs
        self.input_size = input_size
        self.name = name
        self.ViT = ViT

        self.to(device)
        
    def to(self, *args: Any, **kwargs: Any) -> "ModelWrapper":
        self.model.to(*args, **kwargs)
        
        for arg in args:
            if isinstance(arg, torch.device):
                self.device = arg
                
        return super().to(*args, **kwargs)

    def forward(self, x : Tensor) -> Tensor:
        """Forward pass with automated bilinear spatial resizing if required."""
        if x.shape[-1] != self.input_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), 
                mode='bilinear', align_corners=False
            )
        return self.model(x)

    @staticmethod
    def load(
        name: str,
        timm_string: str,
        path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        input_size: int = 224,
        ViT: bool = False,
    ) -> "ModelWrapper":
        """Loads the model and its metadata from a .pth file."""
        model = timm.create_model(timm_string, pretrained=(path is None))

        if path is not None:
            checkpoint = torch.load(path, map_location=device)
            
            # Check if it's a nested dictionary (common in research checkpoints)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
        model = rename_model(model, name)
        model = model.to(device).eval()
        
        wrapped_model = ModelWrapper(
            model=model,
            device=device,
            name=name,
            input_size=input_size,
            ViT=ViT,
        )
        return rename_model(wrapped_model, name)

    def save(self, path: str):
        """Saves the state dict and reconstruction metadata."""
        torch.save({
            "architecture": self.model.__class__.__name__,
            "model_kwargs": self.model_kwargs,
            "state_dict": self.model.state_dict()
        }, f"{path}")

    def evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Returns average loss and accuracy."""
        self.model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({self.name}): ", leave=False):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                predicted = outputs.argmax(dim=1)

                correct += predicted.eq(labels).sum().item()
                total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
                
        return total_loss / total_samples, correct / total_samples

    def get_accuracy(self, loader: DataLoader) -> float:
        """Returns accuracy."""
        _, acc = self.evaluate_loader(loader, nn.CrossEntropyLoss())
        return acc

    def get_loss(self, loader: DataLoader, criterion: nn.Module = nn.CrossEntropyLoss()) -> float:
        """Returns loss."""
        loss, _ = self.evaluate_loader(loader, criterion)
        return loss
    
    def get_fooling_rate(self, clean_loader: DataLoader, perturbed_loader: DataLoader) -> float:
        """Computes the fooling rate."""
        self.model.eval()
        correct_on_clean = 0
        fooled = 0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(
                zip(clean_loader, perturbed_loader),
                desc=f"Evaluating Fooling Rate ({self.name}): ",
                total=len(clean_loader),
                leave=False,
            ):
                clean_images = clean_batch[0].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                
                # Get predictions
                clean_preds = self.forward(clean_images).argmax(dim=1)
                adv_preds = self.forward(adv_images).argmax(dim=1)
                
                # Mask for samples the model actually got right initially
                clean_correct_mask = (clean_preds == labels)
                correct_on_clean += clean_correct_mask.sum().item()
                
                # Mask for samples that were correct but are now wrong
                fooled += (clean_correct_mask & (adv_preds != labels)).sum().item()
                
        return (fooled / correct_on_clean) if correct_on_clean > 0 else 0.0
    
    def get_transfer_rate(
        self,
        target_wrapper: "ModelWrapper",
        clean_loader: DataLoader,
        perturbed_loader: DataLoader,
    ) -> float:
        """
        Computes the transfer rate from the current model to a given Target model (target_wrapper).
        """
        self.model.eval()
        target_wrapper.model.eval()
        
        m_n_denominator = 0
        tr_numerator = 0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(
                zip(clean_loader, perturbed_loader),
                desc=f"Evaluating TR to {target_wrapper.name}: ",
                total=len(clean_loader),
                leave=False,
            ):
                clean_images = clean_batch[0].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                
                # Surrogate predictions
                fs_clean = self.forward(clean_images).argmax(dim=1)
                fs_adv = self.forward(adv_images).argmax(dim=1)
                
                # Target predictions
                ft_clean = target_wrapper.forward(clean_images).argmax(dim=1)
                ft_adv = target_wrapper.forward(adv_images).argmax(dim=1)
                
                # Masks
                fs_clean_correct = (fs_clean == labels)
                ft_clean_correct = (ft_clean == labels)
                fs_fooled = (fs_adv != labels)
                ft_fooled = (ft_adv != labels)
                
                # Strict transferability subset mask
                m_n_mask = fs_fooled & ft_clean_correct & fs_clean_correct
                
                # 5. Accumulate counts
                m_n_denominator += m_n_mask.sum().item()
                tr_numerator += (ft_fooled & m_n_mask).sum().item()
                
        return (tr_numerator / m_n_denominator) if m_n_denominator > 0 else 0.0

    def get_fooling_and_transfer_rate(
        self,
        surrogate: "ModelWrapper",
        clean_loader: DataLoader,
        perturbed_loader: DataLoader,
    ) -> Dict[str, float]:
        """
            Evaluates Fooling Rate (FR), Transfer Rate (TR), and CE Loss simultaneously.
        """
        # Set both models to evaluation mode
        self.model.eval()
        surrogate.model.eval()
        
        # Counters for Fooling Rate (Surrogate-centric)
        ft_correct_on_clean = 0
        ft_fooled = 0
        
        # Counters for Transfer Rate (Strict M_n condition)
        m_n_denominator = 0
        tr_numerator = 0
        
        # Loss counters
        total_samples = 0
        total_ce_loss = 0.0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(
                zip(clean_loader, perturbed_loader),
                desc=f"Evaluating FR & TR vs {surrogate.name}: ",
                total=len(clean_loader),
                leave=False,
            ):
                clean_images = clean_batch[0].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                
                # Target forward passes
                ft_clean_logits = self.forward(clean_images)
                ft_adv_logits = self.forward(adv_images)
                
                ft_clean = ft_clean_logits.argmax(dim=1)
                ft_adv = ft_adv_logits.argmax(dim=1)

                # Surrogate forward passes
                fs_clean = surrogate.forward(clean_images).argmax(dim=1)
                fs_adv = surrogate.forward(adv_images).argmax(dim=1)
                
                # Sample-weighted Loss
                # Sample-weighted Loss
                batch_loss = F.cross_entropy(ft_adv_logits, labels, reduction="sum")
                total_ce_loss += batch_loss.item()
                batch_size = clean_images.size(0)
                total_samples += batch_size
                
                fs_clean_mask = (fs_clean == labels)
                ft_clean_mask = (ft_clean == labels)
                
                fs_fooled_mask = (fs_adv != labels)
                ft_fooled_mask = (ft_adv != labels)
                
                # --- FOOLING RATE ---
                ft_correct_on_clean += ft_clean_mask.sum().item()
                ft_fooled += (ft_clean_mask & ft_fooled_mask).sum().item()
                
                # --- TRANSFER RATE ---
                m_n_mask = fs_fooled_mask & ft_clean_mask & fs_clean_mask
                m_n_denominator += m_n_mask.sum().item()
                tr_numerator += (ft_fooled_mask & m_n_mask).sum().item()
                
        fr = (ft_fooled / ft_correct_on_clean) if ft_correct_on_clean > 0 else 0.0
        tr = (tr_numerator / m_n_denominator) if m_n_denominator > 0 else 0.0
        avg_loss = (total_ce_loss / total_samples) if total_samples > 0 else 0.0
        
        return {
            "fooling_rate": fr,
            "transfer_rate": tr,
            "clean_correct": ft_correct_on_clean,
            "loss": avg_loss
        }