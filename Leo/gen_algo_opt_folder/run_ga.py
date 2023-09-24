#run_ga.py

import sys
sys.path.append("../")

from data_loading import load_data
from genetic_algorithm import genetic_algorithm
from folder_management import create_output_folders
from file_writer import write_run_settings, write_best_genome
from plotting import save_fitness_plot

def main():
    dataset = load_data()

    best_individual, best_fitnesses, variances, best_columns, n_features = genetic_algorithm(dataset, population_size=20, n_generations=2, selection_rate=0.3, mutation_rate=0.05, increased_mutation_rate=0.2, num_elites=None)

    param_labels = ["n_neighbors", "min_dist", "min_cluster_size", "n_components"]

    sub_folder_path = create_output_folders()
    
    settings_str = "population_size=20, n_generations=10, selection_rate=0.3, mutation_rate=0.05, increased_mutation_rate=0.2, num_elites=None\n"
    write_run_settings(sub_folder_path, settings_str)

    write_best_genome(sub_folder_path, best_columns, param_labels, best_individual, n_features, best_fitnesses)
    
    save_fitness_plot(sub_folder_path, best_fitnesses, variances)


if __name__ == "__main__":
    main()