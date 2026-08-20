import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Self
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from tqdm import tqdm
from .utils import rename_model
import timm


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device, name : str = "model", input_size: int =224, ViT: bool = False, **model_kwargs):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.model_kwargs = model_kwargs
        self.input_size = input_size
        self.name = name
        self.ViT = ViT

        self.to(device)
        
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        
        for arg in args:
            if isinstance(arg, torch.device):
                self.device = arg
                
        return super().to(*args, **kwargs)

    def forward(self, x : Tensor) -> Tensor:
        if x.shape[-1] != self.input_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), 
                mode='bilinear', align_corners=False
            )
        return self.model(x)

    @staticmethod
    def load(name, timm_string, path, device: torch.device=torch.device("cuda"), input_size=224, ViT: bool = False):
        """Loads the model and its metadata from a .pth file."""
        model = timm.create_model(timm_string, pretrained=(path is None))
        
        
        checkpoint = torch.load(path, map_location=device)
            
        # Check if it's a nested dictionary (common in research checkpoints)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model = rename_model(model, name)
        model = model.to(device).eval()
        
        return rename_model(ModelWrapper(model=model, name = name, device=device, input_size=input_size, ViT=ViT), name) # type: ignore

    def save(self, path: str):
        """Saves the state dict and reconstruction metadata."""
        torch.save({
            "architecture": self.model.__class__.__name__,
            "model_kwargs": self.model_kwargs,
            "state_dict": self.model.state_dict()
        }, f"{path}")
        print(f"Model saved to {path}")

    def evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Returns average loss and accuracy."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating: ", leave=False):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / total, correct / total

    def get_accuracy(self, loader: DataLoader) -> float:
        """Returns accuracy."""
        _, acc = self.evaluate_loader(loader, nn.CrossEntropyLoss())
        return acc

    def get_loss(self, loader: DataLoader, criterion: nn.Module = nn.CrossEntropyLoss()) -> float:
        """Returns loss."""
        loss, _ = self.evaluate_loader(loader, criterion)
        return loss
    
    def get_fooling_rate(self, clean_loader: DataLoader, perturbed_loader: DataLoader) -> float:
        """Returns fooling rate."""
        self.model.eval()
        correct_on_clean = 0
        fooled = 0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(zip(clean_loader, perturbed_loader), desc=f"Evaluating fooling rate: ", total=len(clean_loader), leave=False ):
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
                
        return (fooled / correct_on_clean) if correct_on_clean > 0 else 0
    
    def get_transfer_rate(self, target_wrapper, clean_loader: DataLoader, perturbed_loader: DataLoader) -> float:
        """
        Computes the transfer rate (TR_A) from the current model (Surrogate: f_s) 
        to a given Target model (f_t: target_wrapper) using clean and adversarial loaders.
        """
        # Set both models to evaluation mode
        self.model.eval()
        target_wrapper.model.eval()
        
        m_n_denominator = 0
        numerator = 0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(zip(clean_loader, perturbed_loader), desc=f"Evaluating TR to {target_wrapper.name}: ", total=len(clean_loader), leave=False):
                clean_images = clean_batch[0].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                
                # 1. Gather predictions from surrogate (self)
                fs_clean = self.forward(clean_images).argmax(dim=1)
                fs_adv = self.forward(adv_images).argmax(dim=1)
                
                # 2. Gather predictions from target (target_wrapper)
                # Mapping target inputs to its own wrapper forwarding mechanism
                ft_clean = target_wrapper.forward(clean_images).argmax(dim=1)
                ft_adv = target_wrapper.forward(adv_images).argmax(dim=1)
                
                # 3. Compute condition components
                fs_clean_correct = (fs_clean == labels)
                ft_clean_correct = (ft_clean == labels)
                fs_fooled = (fs_adv != labels)
                ft_fooled = (ft_adv != labels)
                
                # 4. Mathematical mask for M_n: (f_s(A(x)) != y) & (f_t(x) == f_s(x) == y)
                m_n_mask = fs_fooled & ft_clean_correct & fs_clean_correct
                
                # 5. Accumulate counts
                m_n_denominator += m_n_mask.sum().item()
                numerator += (ft_fooled & m_n_mask).sum().item()
                
        return (numerator / m_n_denominator) if m_n_denominator > 0 else 0

    def evaluate_attack_metrics(self, surrogate, clean_loader: DataLoader, perturbed_loader: DataLoader) -> dict:
        """
            Computes both Fooling Rate (FR) and Transfer Rate (TR) in a single pass.
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
        total_batches = 0
        total_ce_loss = 0.0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(
                zip(clean_loader, perturbed_loader), 
                desc=f"Evaluating FR & TR to {surrogate.name}: ", 
                total=len(clean_loader), 
                leave=False
            ):
                clean_images = clean_batch[0].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                
                # 1. Forward passes (1 per model per batch)
                ft_clean_logits = self.forward(clean_images)
                ft_adv_logits = self.forward(adv_images)
                
                ft_clean = ft_clean_logits.argmax(dim=1)
                ft_adv = ft_adv_logits.argmax(dim=1)
                
                fs_clean = surrogate.forward(clean_images).argmax(dim=1)
                fs_adv = surrogate.forward(adv_images).argmax(dim=1)
                
                # 2. Compute Boolean Masks
                batch_loss = F.cross_entropy(ft_adv_logits, labels)
                total_ce_loss += batch_loss.item()
                total_batches += 1
                
                fs_clean_mask = (fs_clean == labels)
                ft_clean_mask = (ft_clean == labels)
                
                fs_fooled_mask = (fs_adv != labels)
                ft_fooled_mask = (ft_adv != labels)
                
                # --- FOOLING RATE ACCUMULATION ---
                ft_correct_on_clean += ft_clean_mask.sum().item()
                ft_fooled += (ft_clean_mask & ft_fooled_mask).sum().item()
                
                # --- TRANSFER RATE ACCUMULATION ---
                # M_n mask: f_s(A(x)) != y AND f_t(x) == y AND f_s(x) == y
                m_n_mask = fs_fooled_mask & ft_clean_mask & fs_clean_mask
                
                m_n_denominator += m_n_mask.sum().item()
                tr_numerator += (ft_fooled_mask & m_n_mask).sum().item()
                
        # 3. Compute final ratios safely
        fr = (ft_fooled / ft_correct_on_clean) if ft_correct_on_clean > 0 else 0.0
        tr = (tr_numerator / m_n_denominator) if m_n_denominator > 0 else 0.0
        avg_loss = (total_ce_loss / total_batches) if total_batches > 0 else 0.0
        
        return {
            "fooling_rate": fr,
            "transfer_rate": tr,
            "clean_correct": ft_correct_on_clean,
            "loss": avg_loss
        }
    
    def get_targeted_fooling_rate(self, clean_loader: DataLoader, perturbed_loader: DataLoader) -> float:
        """Returns targeted fooling rate."""
        self.model.eval()
        correct_on_clean = 0
        successful_flips = 0

        with torch.no_grad():
            for clean_batch, adv_batch in tqdm(zip(clean_loader, perturbed_loader), desc="Evaluating Targeted Success", total=len(clean_loader), leave=False):
                clean_images = clean_batch[0].to(self.device)
                labels = clean_batch[1].to(self.device)
                target_labels = clean_batch[2].to(self.device)
                adv_images = adv_batch[0].to(self.device)
                
                # Use self.forward to catch the resizing logic for Inception/Ensembles
                clean_preds = self.forward(clean_images).argmax(dim=1)
                adv_preds = self.forward(adv_images).argmax(dim=1)

                # 1. Denominator: Only images the model got right while clean
                clean_correct_mask = (clean_preds == labels)
                correct_on_clean += clean_correct_mask.sum().item()
                
                # 2. Numerator: Correct images that flipped specifically to the target
                successful_flips += (clean_correct_mask & (adv_preds == target_labels)).sum().item()
                
        return (successful_flips / correct_on_clean) if correct_on_clean > 0 else 0