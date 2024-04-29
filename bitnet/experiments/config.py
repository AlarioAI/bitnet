import os
from enum import Enum, auto

class ProjectConfig:
    EXAMPLES_DIR: str = "experiments"
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


class AvailableDatasets(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    MNIST = auto()
    STL10 = auto()
    EUROSAT = auto()