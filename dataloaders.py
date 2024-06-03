from torch.utils.data import random_split
import multiprocessing as mp

from torchvision import datasets, transforms

from bitnet.config import ExperimentConfig
from bitnet.seed import set_seed
from torch.utils.data import DataLoader



def get_loaders(seed: int | None, batch_size: int) -> tuple[DataLoader, ...]:
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Simulate ImageNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = datasets.CIFAR10('./cifar_data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./cifar_data', train=False, download=True, transform=transform)
    val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
    train_size: int = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(mp.cpu_count()))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=int(mp.cpu_count()))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(mp.cpu_count()))

    return train_loader, val_loader, test_loader