import argparse
import json
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from genetic_algorithm import GeneticAlgorithm
import numpy as np
import os
from datetime import datetime
from subprocess import call

# Set a random seed for reproducibility
np.random.seed(42)

current_directory = os.path.dirname(os.path.realpath(__file__))
graphs_script_path = os.path.join(current_directory, "graphs.py")

def write_top_performers_to_txt(results, filename):
    with open(filename, 'w') as f:
        for comb in results["combinations"]:
            parent_selection = comb["parent_selection_method"]
            crossover_method = comb["crossover_method"]
            f.write(f"Parent Selection: {parent_selection}, Crossover Method: {crossover_method}\n\n")
            
            generations = comb["generations"]
            for gen in generations:
                gen_num = gen["generation_number"]
                f.write(f"Generation {gen_num}:\n")
                
                # Sort individuals by fitness and take the top 10
                top_individuals = sorted(gen["individuals"], key=lambda x: x["fitness"], reverse=True)[:10]
                
                for idx, ind in enumerate(top_individuals, 1):
                    genome = ind["genome"]
                    fitness = ind["fitness"]
                    f.write(f"  {idx}. Fitness: {fitness}, Genome: {genome}\n")
                
                f.write("\n")
            
            f.write("="*40 + "\n")
    
def main(args):
    dataset = pd.read_csv(args.dataset_path).drop(columns=["Unnamed: 0"])
    
    hyperparameters = {
        "population_size": args.population_size,
        "n_generations": args.n_generations,
        "selection_rate": 0.3,
        "mutation_rate": 0.05,
        "increased_mutation_rate": 0.2,
        "num_elites": None,
        "score_metric": silhouette_score,
        "parent_selection_method": ["tournament"],  # Adjust as needed
        "crossover_method": ["two_point"],  # Adjust as needed
        "n_jobs": -1
    }

    results_dict = {"combinations": []}
    parent_selection_methods = hyperparameters.pop("parent_selection_method")
    crossover_methods = hyperparameters.pop("crossover_method")
    
    for parent_selection_method in parent_selection_methods:
        for crossover_method in crossover_methods:
            ga = GeneticAlgorithm(
                dataset=dataset, 
                parent_selection_method=parent_selection_method, 
                crossover_method=crossover_method, 
                **hyperparameters
            )
            combination_results = ga.run()
            results_dict["combinations"].append(combination_results)

    # Create a new directory for each run
    current_time = datetime.now().strftime('%m_%d_%H_%M')
    results_dir = f"/Users/leo/Programming/PLR/Leo/4_gen_algo_opt/results/{current_time}/"
    os.makedirs(results_dir, exist_ok=True)

    # Save the JSON to this new directory
    with open(os.path.join(results_dir, "results.json"), "w") as outfile:
        json.dump(results_dict, outfile, default=lambda o: float(o) if isinstance(o, np.float32) else o, indent=4)

    # Write top performers to a text file
    txt_filename = os.path.join(results_dir, "top_performers.txt")
    write_top_performers_to_txt(results_dict, txt_filename)

    # Call the graph generation script
    call(["python", graphs_script_path, "--path", os.path.join(results_dir, "results.json")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm")
    parser.add_argument("--n_generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population_size", type=int, default=50, help="Size of the population")
    parser.add_argument("--dataset_path", type=str, default="/Users/leo/Programming/PLR/Leo/data/dataset_1.csv", help="Path to the dataset CSV file")

    args = parser.parse_args()
    main(args)
