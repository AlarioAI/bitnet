import importlib
import json
from glob import glob
from multiprocessing import Pool

from bitnet.config import ProjectConfig, ExperimentConfig


def run_experiment(module_name: str) -> dict:
    module = importlib.import_module(module_name)
    cur_experiment_results: dict = {}
    return_dict: dict = {}
    for seed in range(ExperimentConfig.NUM_RUNS):
        print(f"Running experiments: `{module_name}` with seed {seed}")
        result_dict, metric, num_parameters = module.run(seed)
        for key, value in result_dict.items():
            if key not in cur_experiment_results:
                cur_experiment_results[key] = {"scores": [], "metric": str(metric), "num_parameters": num_parameters}
            cur_experiment_results[key]["scores"].append(value)
    model_name: str = module_name.split(".")[-1]
    return_dict[model_name] = cur_experiment_results
    return return_dict


def run_experiments_chunk(experiments):
    results = {}
    for exp in experiments:
        results.update(run_experiment(exp))
    return results


def main():
    results: dict = {}
    experiments: list[str] = glob(f"{ProjectConfig.EXAMPLES_DIR}/*py")
    experiments = [exp.replace("/", ".").replace(".py", "") for exp in experiments]

    chunk_size: int = ExperimentConfig.NUM_PARALLEL_EXP

    chunks: list[list[str]] = [experiments[i:i + chunk_size] for i in range(0, len(experiments), chunk_size)]

    with Pool() as pool:
        for chunk_idx, chunk in enumerate(chunks):
            print(f"Running experiments in chunk {chunk_idx + 1}/{len(chunks)}...")
            results_list = pool.map(run_experiments_chunk, [chunk])
            for result in results_list:
                results.update(result)
            with open(f"{ProjectConfig.RESULTS_FILE}_{chunk_idx}.json", 'w') as f:
                json.dump(results, f, indent=4)
            results.clear()


if __name__ == '__main__':
    main()
