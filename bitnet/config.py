from enum import Enum


class ProjectConfig(Enum):
    EXAMPLES_DIR = 'examples'
    RESULTS_FILE = 'experiment_results.json'


class ExperimentConfig:
    NUM_RUNS = 10
    SEED = 53