import argparse
from bitnet.config import HyperparameterConfig
from bitnet.config import Architectures
from runner import run_single_experiment


def run_experiments(seed: int | None, config: dict):
    for model_name, hyperparams in config.items():
        results = run_single_experiment(model_name, seed, hyperparams)
        print(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["all"],
        choices=[arch.name for arch in Architectures],
        help="Select architectures to run experiments on (default: all)."
    )
    args = parser.parse_args()

    if "all" in args.architectures:
        selected_architectures = [arch for arch in Architectures]
    else:
        selected_architectures = [Architectures[arch] for arch in args.architectures]

    for arch in selected_architectures:
        config_obj = HyperparameterConfig(arch.value)
        config = config_obj.load_config()
        run_experiments(None, config)


if __name__ == "__main__":
    main()

