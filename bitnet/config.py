import os


class ProjectConfig:
    EXAMPLES_DIR: str = "examples"
    RESULTS_FILE: str = "experiment_results.json"
    PAPER_DIR: str = "paper"
    PAPER_TEX_PATH: str = os.path.join(f"{PAPER_DIR}", "main.tex")


class ExperimentConfig:
    NUM_RUNS = 10
    SEED = 53