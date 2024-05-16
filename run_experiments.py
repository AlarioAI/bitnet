from bitnet.config import HyperparameterConfig
from runner import run_single_experiment


def run_experiments(seed: int | None, config: dict):
    for model_name, hyperparams in config.items():
        results = run_single_experiment(model_name, seed, hyperparams)
        print(results)

def main():
    config_obj = HyperparameterConfig()
    config = config_obj.load_config()
    run_experiments(None, config)

if __name__ == "__main__":
    main()

