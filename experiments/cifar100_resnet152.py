import torch
from torch import nn

from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d
from bitnet.models.resnet import resnet152
from bitnet.experiments.metrics import Metrics
from bitnet.experiments.config import AvailableDatasets
from bitnet.experiments.data_preparation import get_dataloaders
from bitnet.experiments.model_training import train_model, test_model


def run(seed: int | None) -> tuple[dict[str, float], Metrics, int, int]:

    return_value: dict[str, float] = {}

    num_classes: int        = 100
    learning_rate: float    = 1e-3
    num_epochs: int         = 10
    batch_size: int         = 256

    bitnet = resnet152(BitLinear, BitConv2d, pretrained=False, num_classes=num_classes)
    floatnet = resnet152(nn.Linear, nn.Conv2d, pretrained=False, num_classes=num_classes)
    num_params_bitnet: int = sum(p.numel() for p in bitnet.parameters() if p.requires_grad)
    num_params_floatnet: int = sum(p.numel() for p in floatnet.parameters() if p.requires_grad)
    assert num_params_bitnet == num_params_floatnet

    bitnet_optimizer = torch.optim.Adam(bitnet.parameters(), lr=learning_rate)
    floatnet_optimizer = torch.optim.Adam(floatnet.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader, trainset_size = get_dataloaders(AvailableDatasets.CIFAR100, seed, batch_size)

    bitnet = train_model(bitnet, train_loader, val_loader, bitnet_optimizer, criterion, num_epochs)
    test_model(bitnet, test_loader)
    results, metrics_used = test_model(bitnet, test_loader)
    return_value.update(results)

    train_loader, val_loader, test_loader, trainset_size = get_dataloaders(AvailableDatasets.CIFAR100, seed, batch_size)

    floatnet = train_model(floatnet, train_loader, val_loader, floatnet_optimizer, criterion, num_epochs)
    results, metrics_used = test_model(floatnet, test_loader)
    return_value.update(results)

    return return_value, metrics_used, num_params_bitnet, trainset_size


if __name__ == "__main__":
    print(run(None))