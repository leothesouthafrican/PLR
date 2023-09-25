#genetic_algorithm.py

import random
from cluster_comparison import perform_umap, perform_hdbscan, calculate_silhouette
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import wandb

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


run_name = f'genetic_algorithm_run_{current_time}'
wandb.init(project='PLR', name=run_name)


def genetic_algorithm(dataset, population_size=20, n_generations=100, selection_rate=0.3, mutation_rate=0.05, increased_mutation_rate=0.2, num_elites=None):
    if num_elites is None:
        num_elites = int(0.1 * population_size)

    fitness_cache = {}
    last_best_fitness = -1
    n_features = len(dataset['data_symp_groups_all'].columns)

    def fitness(params, dataset):
        params_tuple = tuple(params)
        if params_tuple in fitness_cache:
            return fitness_cache[params_tuple]

        selected_features = params[:n_features]
        n_neighbors, min_dist, min_cluster_size, n_components = params[n_features:]
        dataset_name, dataset = random.choice(list(dataset.items()))
        dataset = dataset[[col for col, keep in zip(dataset.columns, selected_features) if keep]].dropna()
        n_neighbors = max(2, int(n_neighbors))
        min_cluster_size = max(2, int(min_cluster_size))
        n_components = max(2, int(n_components))
        min_dist = min(min_dist, 1.0)

        umap_result = perform_umap(dataset, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        labels = perform_hdbscan(umap_result, min_cluster_size=min_cluster_size)
        score = calculate_silhouette(umap_result, labels)
        fitness_cache[params_tuple] = score

        return score
    
    # Best known feature vector
    best_feature_vector = (
        1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
        1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 
        1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 
        1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 
        1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 
        1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0
    )

    # Initialize population
    population = []

    for _ in range(population_size // 10):  # Adjusted to account for increased population initialization
        # Individuals close to the best performing genome
        population.append(best_feature_vector + (0.6492373060806188, 0.03733592291556887, 29, 47))
        population.append(best_feature_vector + (0.6757706356698308, 0.04030749239323905, 38, 35))

        # Slight variations of the above individuals
        population.append(best_feature_vector + (0.6757706356698308 + random.uniform(-0.1, 0.1), 0.04030749239323905 + random.uniform(-0.01, 0.01), 29 + random.randint(-2, 2), 48 + random.randint(-2, 2)))
        population.append(best_feature_vector + (0.6757706356698308 + random.uniform(-0.1, 0.1), 0.04030749239323905 + random.uniform(-0.01, 0.01), 38 + random.randint(-2, 2), 35 + random.randint(-2, 2)))

        # Diverse genome exploration - now bumped up to 4 individuals
        population.append(best_feature_vector + (random.uniform(0.1, 5), random.uniform(0.045, 0.75), random.randint(30, 45), random.randint(8, 22)))
        population.append(best_feature_vector + (random.uniform(6, 20), random.uniform(0.76, 1.0), random.randint(46, 50), random.randint(23, 25)))
        population.append(best_feature_vector + (random.uniform(21, 35), random.uniform(0.26, 0.5), random.randint(21, 35), random.randint(25, 38)))
        population.append(best_feature_vector + (random.uniform(36, 45), random.uniform(0.51, 0.75), random.randint(36, 45), random.randint(39, 65)))

    # To store best fitness and variance for each generation
    best_fitnesses = []
    variances = []

    for generation in tqdm(range(n_generations), desc="Generations"):
        scores = Parallel(n_jobs=-1)(delayed(fitness)(ind, dataset) for ind in population)

        best_fitness = max(scores)
        best_fitnesses.append(best_fitness)
        variances.append(np.var(scores))

        # Log the metrics (without the genome) to wandb
        wandb.log({
            "Best Fitness": best_fitness,
            "Variance": variances[generation]
        })

        best_idx = scores.index(best_fitness)
        best_genome = population[best_idx]
        best_genome_str = ', '.join(map(str, best_genome))
        best_columns = [col for col, keep in zip(dataset['data_symp_groups_all'].columns, best_genome[:n_features]) if keep]


        # Create and log a new table for this generation
        table_data = [(generation, best_genome_str, ', '.join(best_columns), best_fitness)]
        table = wandb.Table(columns=["Generation", "Best Genome", "Best Columns", "Fitness"], data=table_data)
        wandb.log({f"Best Genomes (Generation {generation})": table})

        if best_fitness <= last_best_fitness:
            mutation_rate = increased_mutation_rate
        else:
            mutation_rate = 0.1
        last_best_fitness = best_fitness

        sorted_indices = np.argsort(scores)[::-1]
        elites = [population[i] for i in sorted_indices[:num_elites]]

        fitness_sum = sum(scores)
        selected_population = []
        for _ in range(int(selection_rate * population_size)):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for i in range(len(scores)):
                current += scores[i]
                if current > pick:
                    selected_population.append(population[i])
                    break

        children = []
        while len(children) < population_size - len(selected_population) - num_elites:
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            children.append(child)

        mutations = 0
        for i in range(len(children)):
            if random.random() < mutation_rate:
                mutate_pos = random.randint(0, len(children[i]) - 1)
                if mutate_pos < n_features:
                    mutated_bit = 1 - children[i][mutate_pos]
                    children[i] = children[i][:mutate_pos] + (mutated_bit,) + children[i][mutate_pos+1:]
                else:
                    new_value = random.choice([
                        random.randint(5, 50),
                        random.uniform(0.0, 1.0),
                        random.randint(5, 50),
                        random.randint(2, 25)
                    ])
                    children[i] = children[i][:mutate_pos] + (new_value,) + children[i][mutate_pos+1:]
                mutations += 1

        population = elites + selected_population + children

    final_scores = Parallel(n_jobs=-1)(delayed(fitness)(ind, dataset) for ind in population)
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    best_individual = population[best_idx]
    best_columns = [col for col, keep in zip(dataset['data_symp_groups_all'].columns, best_individual[:n_features]) if keep]

    return best_individual, best_fitnesses, variances, best_columns, n_features