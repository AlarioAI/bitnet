import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bitnet.metrics import Metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        num_epochs: int,
        model_name: str) -> tuple[nn.Module, float]:

    best_model: nn.Module = model
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model = model.to(device)
        model.train()

        pbar = tqdm(total=len(train_loader), desc=f"Training {model_name}")
        running_loss: float = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(
                f'Training {model_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (i+1):.4f}'
            )
            pbar.update(1)
        pbar.close()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f'Validation {model_name} - Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    return best_model, val_loss


def test_model(model: nn.Module, test_loader: DataLoader, model_name: str) -> float:
    model.eval()
    model = model.to(device)
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy: float = 100 * correct / total
    return accuracy
