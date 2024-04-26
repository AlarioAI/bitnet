import multiprocessing as mp

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d
from bitnet.models.resnet import resnet152
from bitnet.metrics import Metrics
from bitnet.seed import set_seed
from bitnet.config import ExperimentConfig
from bitnet.base_experiment import train_model, test_model


def run(seed: int | None) -> tuple[dict[str, float], Metrics, int, int]:

    set_seed(seed)
    return_value: dict[str, float] = {}

    num_classes: int        = 100
    learning_rate: float    = 1e-3
    num_epochs: int         = 10
    batch_size: int         = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    bitnet = resnet152(BitLinear, BitConv2d, pretrained=False, num_classes=num_classes)
    floatnet = resnet152(nn.Linear, nn.Conv2d, pretrained=False, num_classes=num_classes)
    num_params_bitnet: int = sum(p.numel() for p in bitnet.parameters() if p.requires_grad)
    num_params_floatnet: int = sum(p.numel() for p in floatnet.parameters() if p.requires_grad)
    assert num_params_bitnet == num_params_floatnet

    bitnet_optimizer = torch.optim.Adam(bitnet.parameters(), lr=learning_rate)
    floatnet_optimizer = torch.optim.Adam(floatnet.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.CIFAR100('./cifar_data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100('./cifar_data', train=False, download=True, transform=transform)
    val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
    train_size: int = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    set_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )

    bitnet = train_model(bitnet, train_loader, val_loader, bitnet_optimizer, criterion, num_epochs)
    test_model(bitnet, test_loader)
    results, metrics_used = test_model(bitnet, test_loader)
    return_value.update(results)

    set_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(mp.cpu_count() / ExperimentConfig.NUM_PARALLEL_EXP)
    )
    floatnet = train_model(floatnet, train_loader, val_loader, floatnet_optimizer, criterion, num_epochs)
    results, metrics_used = test_model(floatnet, test_loader)
    return_value.update(results)
    trainset_size: int = len(train_dataset)

    return return_value, metrics_used, num_params_bitnet, trainset_size


if __name__ == "__main__":
    print(run(None))