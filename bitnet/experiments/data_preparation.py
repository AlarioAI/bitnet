import multiprocessing as mp

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from bitnet.datasets.eurosat import EuroSAT
from bitnet.experiments.config import AvailableDatasets, ExperimentConfig
from bitnet.experiments.seed import set_seed


def get_dataloaders(
        dataset_name: AvailableDatasets, seed: int | None, batch_size: int
    ) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    set_seed(seed)
    match dataset_name:
        case AvailableDatasets.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            train_dataset = datasets.CIFAR10('./cifar_data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./cifar_data', train=False, download=True, transform=transform)
            val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
            train_size: int = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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

        case AvailableDatasets.CIFAR100:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2623, 0.2513, 0.2714))
            ])
            train_dataset = datasets.CIFAR100('./cifar_data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100('./cifar_data', train=False, download=True, transform=transform)
            val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
            train_size: int = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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

        case AvailableDatasets.EUROSAT:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(((0.485,0.456, 0.406)), (0.229, 0.224, 0.225))
            ])
            train_dataset = EuroSAT("./eurosat_data/", split="train", download=True, transform=transform)
            val_dataset = EuroSAT("./eurosat_data/", split="val", download=True, transform=transform)
            test_dataset = EuroSAT("./eurosat_data/", split="test", download=True, transform=transform)

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

        case AvailableDatasets.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(((0.485,0.456, 0.406)), (0.229, 0.224, 0.225))
            ])
            train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)
            val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
            train_size: int = len(train_dataset) - val_size

            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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

        case AvailableDatasets.STL10:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ])
            train_dataset = datasets.STL10('./stl10_data', split='train', download=True, transform=transform)
            test_dataset = datasets.STL10('./stl10_data', split='test', download=True, transform=transform)
            val_size: int = int(ExperimentConfig.VALIDATION_SIZE * len(train_dataset))
            train_size: int = len(train_dataset) - val_size

            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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

        case _:
            raise ValueError(f"Unrecognized dataset name: {dataset_name}")

    return train_loader, val_loader, test_loader, len(train_dataset)
