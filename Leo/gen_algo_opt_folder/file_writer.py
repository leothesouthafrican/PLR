import os

def write_run_settings(sub_folder_path, settings_str):
    """ Write the given settings to run_settings.txt. """
    with open(os.path.join(sub_folder_path, "run_settings.txt"), "w") as f:
        f.write(settings_str)

def write_best_genome(sub_folder_path, best_columns, param_labels, best_individual, n_features, best_fitnesses):
    """ Write the best genome and associated info to best_genome.txt. """
    with open(os.path.join(sub_folder_path, "best_genome.txt"), "w") as f:
        f.write("Best Individual's Genome:\n")

        # Display the feature name for each 1 in the genome
        for col in best_columns:
            f.write(f"{col}\n")
        
        f.write("\nAdditional Parameters:\n")
        for label, value in zip(param_labels, best_individual[n_features:]):
            f.write(f"{label}: {value}\n")
        
        f.write(f"\nBest Individual's Fitness: {max(best_fitnesses)}\n")
