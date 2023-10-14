# graphs.py

import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def plot_best_fitness_over_generations(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    combinations = data["combinations"]

    plt.figure(figsize=(12, 6))
    
    # For each combination, plot a line
    for comb in combinations:
        parent_selection = comb["parent_selection_method"]
        crossover_method = comb["crossover_method"]
        label = f"{parent_selection}-{crossover_method}"

        generations = comb["generations"]
        x = [gen["generation_number"] for gen in generations]
        y = [max([ind["fitness"] for ind in gen["individuals"]]) for gen in generations]
        
        # Calculate variance for shading
        y_std = [np.std([ind["fitness"] for ind in gen["individuals"]]) for gen in generations]
        y_upper = [a + b for a, b in zip(y, y_std)]
        y_lower = [a - b for a, b in zip(y, y_std)]


        # Plot
        plt.plot(x, y, label=label)
        plt.fill_between(x, y_lower, y_upper, alpha=0.2)

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness over Generations for Different Combinations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(json_file_path.replace("results.json", "fitness_plot.png"))
    plt.show()

def plot_avg_fitness_over_generations(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    combinations = data["combinations"]

    plt.figure(figsize=(12, 6))
    
    # For each combination, plot a line
    for comb in combinations:
        parent_selection = comb["parent_selection_method"]
        crossover_method = comb["crossover_method"]
        label = f"{parent_selection}-{crossover_method}"

        generations = comb["generations"]
        x = [gen["generation_number"] for gen in generations]
        y = [np.mean([ind["fitness"] for ind in gen["individuals"]]) for gen in generations]
        
        # Plot
        plt.plot(x, y, label=label)

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))  # Ensure only whole numbers on x-axis
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness over Generations for Different Combinations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(json_file_path.replace("results.json", "avg_fitness_plot.png"))
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Graphs for GA Results")
    parser.add_argument("--path", type=str, required=True, help="Path to the results JSON file")

    args = parser.parse_args()
    plot_best_fitness_over_generations(args.path)
    plot_avg_fitness_over_generations(args.path)

if __name__ == "__main__":
    # This allows for command-line use. User can provide path to JSON for visualizations.
    import argparse

    parser = argparse.ArgumentParser(description="Generate Graphs for GA Results")
    parser.add_argument("--path", type=str, required=True, help="Path to the results JSON file")

    args = parser.parse_args()
    plot_best_fitness_over_generations(args.path)
