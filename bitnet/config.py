
import importlib
import os
import types
from enum import Enum
from pathlib import Path
from typing import Callable

import yaml


def get_callable_from_string(callable_string: str) -> Callable:
    module_name, func_name = callable_string.rsplit('.', 1)
    module: types.ModuleType = importlib.import_module(module_name)
    return getattr(module, func_name)


class Architectures(Enum):
    resnet = "config/resnet.yaml"
    convnext = "config/convnext.yaml"
    efficientnet = "config/efficientnet.yaml"


class ProjectConfig:
    PAPER_DIR: str = "paper"
    RESULTS_FILE: str = os.path.join(PAPER_DIR, "experiment_results.json")
    PAPER_TEX_PATH: str = os.path.join(f"{PAPER_DIR}", "main.tex")
    TABLE_TEX_PATH: str = os.path.join(f"{PAPER_DIR}", "table.tex")


class ExperimentConfig:
    NUM_RUNS: int = 10
    SEED: int = 53
    VALIDATION_SIZE: float = 0.2
    TEST_SIZE: float = 0.2
    NUM_PARALLEL_EXP: int = 1


class HyperparameterConfig:
    def __init__(self, yaml_config: str):
        self.config_path: Path = Path(yaml_config)
        self.config = self.load_config()


    def load_config(self) -> dict[str, dict]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)


    def get_hyperparameters(self, model_name: str) -> dict:
        return self.config.get(model_name, {})


    def update_hyperparameters(self, model_name: str, new_params: dict):
        self.config[model_name] = new_params
        self.save_config()


    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)
