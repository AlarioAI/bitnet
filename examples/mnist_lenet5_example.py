import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm

from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d
from bitnet.models.lenet5 import LeNet
from seed import set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        num_epochs: int) -> None:
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader), desc=f"Training {model.__name__}")
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
                f'Model: {model.__name__} - Epoch [{epoch+1}], Loss: {running_loss / (i+1):.4f}'
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
    print(f'Accuracy of {model.__name__}: {accuracy:.2f}%')


def main():

    set_seed()

    num_classes: int        = 10
    learning_rate: float    = 1e-3
    num_epochs: int         = 5
    batch_size: int         = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"Testing on {device=}")
    bitnet = LeNet(BitLinear, BitConv2d, num_classes, 1, 28).to(device)
    floatnet = LeNet(nn.Linear, nn.Conv2d, num_classes, 1, 28).to(device)

    bitnet_optimizer = torch.optim.Adam(bitnet.parameters(), lr=learning_rate)
    floatnet_optimizer = torch.optim.Adam(floatnet.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)

    set_seed()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(bitnet, train_loader, bitnet_optimizer, criterion, num_epochs)
    test_model(bitnet, test_loader)

    set_seed()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(floatnet, train_loader, floatnet_optimizer, criterion, num_epochs)
    test_model(floatnet, test_loader)


if __name__ == "__main__":
    main()
