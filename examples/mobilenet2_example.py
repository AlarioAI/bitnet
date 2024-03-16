import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm

from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d
from bitnet.models.resnet import resnet18
from bitnet.models.mobilenet2 import create_mobilenet2, create_bit_mobilenet2
from seed import set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        num_epochs: int) -> None:
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader), desc=f"Training {model.name}")
        running_loss: float = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(
                f'Model: {model.name} - Epoch [{epoch+1}], Loss: {running_loss / (i+1):.4f}'
            )
            pbar.update(1)
        pbar.close()


def test_model(model: nn.Module, test_loader: DataLoader):
    model.eval()
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of {model.name}: {accuracy:.2f}%')


def main():
    num_classes: int        = 100
    learning_rate: float    = 1e-3
    num_epochs: int         = 10
    batch_size: int         = 128

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2623, 0.2513, 0.2714])
    ])

    bitnet =  create_bit_mobilenet2(num_classes=num_classes).to(device)
    floatnet = create_mobilenet2(num_classes).to(device)

    bitnet_optimizer = torch.optim.Adam(bitnet.parameters(), lr=learning_rate)
    floatnet_optimizer = torch.optim.Adam(floatnet.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.CIFAR100('./cifar_data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100('./cifar_data', train=False, download=True, transform=transform)

    print("Training BitNet")
    set_seed()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(bitnet, train_loader, bitnet_optimizer, criterion, num_epochs)
    test_model(bitnet, test_loader)

    print("Training FloatNet")
    set_seed()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(floatnet, train_loader, floatnet_optimizer, criterion, num_epochs)
    test_model(floatnet, test_loader)


if __name__ == "__main__":
    main()
