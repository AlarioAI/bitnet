import importlib
import json
from glob import glob


def run_experiment(module_name: str, num_runs: int = 2) -> dict:
    module = importlib.import_module(module_name)
    cur_experiment_results: dict = {}
    return_dict: dict = {}
    for seed in range(num_runs):
        print(f"Running experiments: `{module_name}` with seed {seed}")
        result_dict, metric = module.run(seed)
        for key, value in result_dict.items():
            if key not in cur_experiment_results:
                cur_experiment_results[key] = {"scores": [], "metric": str(metric)}
            cur_experiment_results[key]["scores"].append(value)
    model_name: str = module_name.split(".")[-1]
    return_dict[model_name] = cur_experiment_results
    return return_dict


def main():
    results: dict = {}
    experiments: list[str] = glob("examples/*py")
    experiments = [exp.replace("/", ".").replace(".py", "") for exp in experiments]
    for exp in experiments:
        results.update(run_experiment(exp))

        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()