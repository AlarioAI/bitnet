from bitnet.config import HyperparameterConfig
from runner import run_single_experiment


def run_experiments(seed: int | None, config: dict):
    for model_name, hyperparams in config.items():
        run_single_experiment(model_name, seed, config)


def main():
    config_obj = HyperparameterConfig()
    config = config_obj.load_config()
    results = run_experiments(None, config)
    print(results)

if __name__ == "__main__":
    main()

