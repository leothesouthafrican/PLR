import argparse
import json
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from genetic_algorithm import GeneticAlgorithm
import numpy as np

def main(args):
    dataset = pd.read_csv("/Users/leo/Programming/PLR/Leo/data/dataset_1.csv").drop(columns=["Unnamed: 0"])
    hyperparameters = {
        "population_size": args.population_size,
        "n_generations": args.n_generations,
        "selection_rate": 0.3,
        "mutation_rate": 0.05,
        "increased_mutation_rate": 0.2,
        "num_elites": None,
        "depth_range": (1, 5),
        "hidden_dim_range": (64, 256),
        "n_epochs": 15,
        "score_metric": silhouette_score,
        "clustering_algo": "hdbscan",
        "parent_selection_method": ["roulette"], #["roulette", "tournament", "rank", "elitism"],
        "crossover_method":  ["two_point"], #["one_point", "two_point", "uniform"],
        "min_cluster_size_range": (2, 50),
        "batch_size": 64,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "mps"),
        "n_jobs": -1
    }

    results_dict = {"combinations": []}
    parent_selection_methods = hyperparameters.pop("parent_selection_method")
    crossover_methods = hyperparameters.pop("crossover_method")
    for parent_selection_method in parent_selection_methods:
        for crossover_method in crossover_methods:
            ga = GeneticAlgorithm(dataset=dataset, parent_selection_method=parent_selection_method, crossover_method=crossover_method, **hyperparameters)
            combination_results = ga.run()
            results_dict["combinations"].append(combination_results)

    with open("/Users/leo/Programming/PLR/Leo/4_gen_algo_opt/results/results.json", "w") as outfile:
        json.dump(results_dict, outfile, default=lambda o: float(o) if isinstance(o, np.float32) else o, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm")
    parser.add_argument("--n_generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population_size", type=int, default=20, help="Size of the population")

    args = parser.parse_args()
    main(args)
