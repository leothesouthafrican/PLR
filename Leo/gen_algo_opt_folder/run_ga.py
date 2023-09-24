#run_ga.py

import sys
sys.path.append("../")

from data_loading import load_data
from genetic_algorithm import genetic_algorithm
import matplotlib.pyplot as plt
import numpy as np

def main():
    dataset = load_data()

    # Specify other parameters as needed
    best_individual, best_fitnesses, variances = genetic_algorithm(dataset, population_size=20, n_generations=10, selection_rate=0.3, mutation_rate=0.05, increased_mutation_rate = 0.2, num_elites=None)

    # Print the best individual's genome and fitness
    print("Best Individual's Genome:", best_individual)
    print("Best Individual's Fitness:", max(best_fitnesses))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_fitnesses) + 1), best_fitnesses, label='Best Fitness')
    plt.fill_between(range(1, len(best_fitnesses) + 1), 
                     np.array(best_fitnesses) - np.array(variances), 
                     np.array(best_fitnesses) + np.array(variances), 
                     color='gray', alpha=0.5, label='Variance')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Best Fitness Over Generations')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
