import matplotlib.pyplot as plt
import numpy as np
import os

def save_fitness_plot(sub_folder_path, best_fitnesses, variances):
    """ Generate and save the fitness evolution plot. """
    
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
    plt.savefig(os.path.join(sub_folder_path, "fitness_evolution_plot.png"))
    plt.show()
