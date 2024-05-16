import torch
import torch.nn as nn


from bitnet.layer_swap import replace_layers
from bitnet.seed import set_seed
from bitnet.base_experiment import train_model, test_model
from bitnet.config import get_callable_from_string

from dataloaders import get_loaders

from dataclasses import dataclass


@dataclass
class ExperimentResult:
    best_bitnet: torch.nn.Module
    bitnet_val_loss: float
    bitnet_accuracy: float
    best_floatnet: torch.nn.Module
    floatnet_val_loss: float
    floatnet_accuracy: float


def run_single_experiment(model_name: str, seed: int | None, hyperparams: dict):
    set_seed(seed)
    model = get_callable_from_string(hyperparams["model"])
    floatnet = model(pretrained=False)
    bitnet = model(pretrained=False)
    replace_layers(bitnet)

    bitnet_opt = torch.optim.Adam(bitnet.parameters(), lr=hyperparams["learning_rate"])
    floatnet_opt = torch.optim.Adam(floatnet.parameters(), lr=hyperparams["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = get_loaders(seed, hyperparams["batch_size"])

    best_bitnet, bitnet_val_loss = train_model(bitnet, train_loader, val_loader, bitnet_opt, criterion, hyperparams["num_epochs"], f"bitnet_{model_name}")
    bitnet_accuracy = test_model(best_bitnet, test_loader, "bitnet")

    best_floatnet, floatnet_val_loss = train_model(floatnet, train_loader, val_loader, floatnet_opt, criterion, hyperparams["num_epochs"], f"floatnet_{model_name}")
    floatnet_accuracy = test_model(best_floatnet, test_loader, "bitnet")

    return ExperimentResult(
        best_bitnet, bitnet_val_loss, bitnet_accuracy, best_floatnet, floatnet_val_loss, floatnet_accuracy
    )