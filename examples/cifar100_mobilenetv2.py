import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from bitnet.models.mobilenet_v2 import mobilenet_v2, bit_mobilenet_v2
from bitnet.metrics import Metrics
from bitnet.seed import set_seed
from bitnet.base_experiment import train_model, test_model


def run(seed: int | None) -> tuple[dict[str, float], Metrics, int]:

    set_seed(seed)
    return_value: dict[str, float] = {}

    num_classes: int        = 100
    learning_rate: float    = 1e-3
    num_epochs: int         = 10
    batch_size: int         = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2623, 0.2513, 0.2714])
    ])

    bitnet =  bit_mobilenet_v2(num_classes=num_classes, pretrained=True)
    floatnet = mobilenet_v2(num_classes, pretrained=True)
    num_params_bitnet: int = sum(p.numel() for p in bitnet.parameters() if p.requires_grad)
    num_params_floatnet: int = sum(p.numel() for p in floatnet.parameters() if p.requires_grad)
    assert num_params_bitnet == num_params_floatnet

    bitnet_optimizer = torch.optim.Adam(bitnet.parameters(), lr=learning_rate)
    floatnet_optimizer = torch.optim.Adam(floatnet.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.CIFAR100('./cifar_data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100('./cifar_data', train=False, download=True, transform=transform)

    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(bitnet, train_loader, bitnet_optimizer, criterion, num_epochs)
    test_model(bitnet, test_loader)
    results, metrics_used = test_model(bitnet, test_loader)
    return_value.update(results)

    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_model(floatnet, train_loader, floatnet_optimizer, criterion, num_epochs)
    test_model(floatnet, test_loader)
    results, metrics_used = test_model(floatnet, test_loader)
    return_value.update(results)

    return return_value, metrics_used, num_params_bitnet


if __name__ == "__main__":
    print(run(None))