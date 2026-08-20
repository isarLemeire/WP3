import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Union
from tqdm import tqdm

from model_wrapper import ModelWrapper


class Trainer:
    @staticmethod
    def train_model(
        model: Union[nn.Module, ModelWrapper], 
        epochs: int, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module = nn.CrossEntropyLoss(),
        device: torch.device = torch.device('cpu')
    ) -> None:
        
        model.to(device)

        for epoch in range(epochs):
            # Private calls to the internal logic
            t_loss, t_acc = Trainer._run_step(model, train_loader, criterion, optimizer, train=True)
            v_loss, v_acc = Trainer._run_step(model, val_loader, criterion, train=False)

            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {t_loss:.3f} Acc: {t_acc:.2f}% | "
                  f"Val Loss: {v_loss:.3f} Acc: {v_acc:.2f}%")

    @staticmethod
    def _run_step(
        model: Union[nn.Module, ModelWrapper], 
        loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer | None = None,
        train: bool = False
    ) -> Tuple[float, float]:
        
        model.train() if train else model.eval()
        context = torch.enable_grad() if train else torch.no_grad()
        
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0
        
        desc: str = "Training" if train else "Validation"
        loop = tqdm(loader, desc=desc, leave=False, unit="batch")

        with context:
            for inputs, labels in loop:
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                if train and optimizer:
                    optimizer.zero_grad()

                outputs: torch.Tensor = model(inputs)
                loss: torch.Tensor = criterion(outputs, labels)

                if train and optimizer:
                    loss.backward()
                    optimizer.step()

                batch_size: int = int(inputs.size(0))
                total_loss += float(loss.item() * batch_size)
                
                predicted = outputs.argmax(dim=1)
                total += batch_size
                correct += int(predicted.eq(labels).sum().item())
                
                if train:
                    loop.set_postfix(loss=float(loss.item()))

        final_loss: float = total_loss / total if total > 0 else 0.0
        final_acc: float = (100.0 * correct / total) if total > 0 else 0.0
        
        return final_loss, final_acc