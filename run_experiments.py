import torch
import torch.nn as nn


from bitnet.layer_swap import replace_layers
from bitnet.seed import set_seed
from bitnet.base_experiment import train_model, test_model
from bitnet.config import HyperparameterConfig, get_callable_from_string

from dataloaders import get_loaders


def run_experiments(seed: int | None, config: dict):
    for model_name, hyperparams in config.items():
        set_seed(seed)
        model = get_callable_from_string(hyperparams["model"])
        floatnet = model(pretrained=False)
        bitnet = model(pretrained=False)
        replace_layers(bitnet)

        bitnet_opt = torch.optim.Adam(bitnet.parameters(), lr=hyperparams["learning_rate"])
        floatnet_opt = torch.optim.Adam(floatnet.parameters(), lr=hyperparams["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader, test_loader = get_loaders(seed, hyperparams["batch_size"])

        best_bitnet = train_model(bitnet, train_loader, val_loader, bitnet_opt, criterion, hyperparams["num_epochs"], f"bitnet_{model_name}")
        bitnet_results = test_model(best_bitnet, test_loader, "bitnet")

        best_floatnet = train_model(floatnet, train_loader, val_loader, floatnet_opt, criterion, hyperparams["num_epochs"], f"floatnet_{model_name}")
        floatnet_results = test_model(best_floatnet, test_loader, "bitnet")


def main():
    config_obj = HyperparameterConfig()
    config = config_obj.load_config()
    run_experiments(None, config)


if __name__ == "__main__":
    main()

