import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from cluster_comparison import perform_umap, perform_hdbscan, calculate_silhouette
from data_loading import load_data
from datetime import datetime
import wandb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from autoencoder import Autoencoder

class GeneticAlgorithm:
    def __init__(self, dataset, population_size=100, n_generations=20, selection_rate=0.3, mutation_rate=0.05, 
                 increased_mutation_rate=0.2, num_elites=None):
        self.dataset = dataset
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.increased_mutation_rate = increased_mutation_rate
        self.num_elites = num_elites or int(0.1 * population_size)
        self.n_features = len(dataset['data_symp_groups_all'].columns)
        self.fitness_cache = {}
        self.last_best_fitness = -1

    def fitness(self, params):
        assert len(params) == self.n_features + 4, f"Expected length: {self.n_features + 4}, but got: {len(params)} with params: {params}"
        params_tuple = tuple(params)
        if params_tuple in self.fitness_cache:
            return self.fitness_cache[params_tuple]

        selected_features = params[:self.n_features]
        n_neighbors, min_dist, min_cluster_size, n_components = params[self.n_features:]
        dataset_name, dataset = random.choice(list(self.dataset.items()))
        dataset = dataset[[col for col, keep in zip(dataset.columns, selected_features) if keep]].dropna()
        n_neighbors = max(2, int(n_neighbors))
        min_cluster_size = max(2, int(min_cluster_size))
        n_components = max(2, int(n_components))
        min_dist = min(min_dist, 1.0)

        umap_result = perform_umap(dataset, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        labels = perform_hdbscan(umap_result, min_cluster_size=min_cluster_size)
        score = calculate_silhouette(umap_result, labels)
        self.fitness_cache[params_tuple] = score

        return score

    def initialize_population(self, best_feature_vector):
        population = []
        for _ in range(self.population_size // 10): 
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

        return population

    def mutate(self, child, best_individual):
        mutate_pos = random.randint(0, len(child) - 1)
        if mutate_pos < self.n_features:
            mutated_bit = 1 - child[mutate_pos]
            child = child[:mutate_pos] + (mutated_bit,) + child[mutate_pos+1:]
        else:
            # For parameter mutations
            if mutate_pos == self.n_features:  # for n_neighbors
                new_value = int(best_individual[mutate_pos] + np.random.normal(0, 10))
            elif mutate_pos == self.n_features + 1:  # for min_dist
                new_value = best_individual[mutate_pos] + np.random.normal(0, 0.35)
                new_value = max(0, min(new_value, 1))
            elif mutate_pos == self.n_features + 2 or mutate_pos == self.n_features + 3:  # for min_cluster_size and n_components
                new_value = int(best_individual[mutate_pos] + np.random.normal(0, 10))
            child = child[:mutate_pos] + (new_value,) + child[mutate_pos+1:]
        
        return child

    def run(self):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'genetic_algorithm_run_{current_time}'
        wandb.init(project='PLR', name=run_name)

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

        best_individual = None
        best_individual_fitness = float('-inf')

        population = self.initialize_population(best_feature_vector)
        best_fitnesses = []
        variances = []

        for generation in tqdm(range(self.n_generations), desc="Generations"):
            scores = Parallel(n_jobs=-1)(delayed(self.fitness)(ind) for ind in population)
            
            # 1. Storing the best individual and its fitness value.
            max_fitness_index = scores.index(max(scores))
            current_best_fitness = scores[max_fitness_index]
            current_best_individual = population[max_fitness_index]

            # Get the best genome string and related columns
            best_genome_str = ', '.join(map(str, current_best_individual))
            try:
                best_columns = [col for col, bit in zip(self.dataset['data_symp_groups_all'].columns, current_best_individual[:self.n_features]) if bit]
            except KeyError:
                best_columns = []

            # Create a table for wandb logging
            table_data = [(generation, best_genome_str, ', '.join(best_columns), current_best_fitness)]
            table = wandb.Table(columns=["Generation", "Best Genome", "Best Columns", "Fitness"], data=table_data)
            

            if current_best_fitness > best_individual_fitness:
                best_individual = current_best_individual
                best_individual_fitness = current_best_fitness
            else:
                # 2. Increasing mutation rate if there's no improvement in the best fitness.
                self.mutation_rate = min(self.increased_mutation_rate, 1.0) 

            best_fitnesses.append(best_individual_fitness)

            print(f"Generation {generation + 1}/{self.n_generations}: Best Fitness = {best_individual_fitness}, Best Genome = {best_individual}")
            
            # 3. Selection: select potential parents with a probability related to their fitness.
            total_fitness = sum(scores)
            if total_fitness == 0:
                raise ValueError("Total fitness is zero. Please check the fitness calculation.")
            probabilities = [score / total_fitness for score in scores]
            parents_indices = np.random.choice(len(population), size=int(self.population_size * self.selection_rate), p=probabilities, replace=False)
            parents = [population[i] for i in parents_indices]

            # 4. Crossover (recombination): pairs of parents are combined to create children.
            children = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):  # Ensure we have pairs of parents
                    parent1 = parents[i]
                    parent2 = parents[i + 1]
                    crossover_point = random.randint(0, len(parent1) - 1)
                    child1 = parent1[:crossover_point] + parent2[crossover_point:]
                    child2 = parent2[:crossover_point] + parent1[crossover_point:]
                    children.extend([child1, child2])

            # 5. Mutation: apply random changes to children.
            for i in range(len(children)):
                if random.random() < self.mutation_rate:
                    children[i] = self.mutate(children[i], best_individual)

            # 6. Elitism: keep the best individuals from the current generation.
            elites = sorted(zip(scores, population), reverse=True)[:self.num_elites]
            elites = [individual for _, individual in elites]

            # 7. Replace the old population with the new population of children and elites.
            population = children + elites

            # Calculate variance for analysis purposes.
            print("Before append, variances:", variances)
            variances.append(np.var(scores))
            print("After append, variances:", variances)

            # Log the results using wandb
            wandb.log({'Best Fitness': current_best_fitness, 'Variance': variances[-1], 'Generation': generation, f"Best Genomes (Generation {generation})": table})

            # Update the mutation rate based on the progress.
            if current_best_fitness > self.last_best_fitness:
                self.mutation_rate = self.increased_mutation_rate
            self.last_best_fitness = current_best_fitness

        try:
            best_columns = [col for col, bit in zip(self.dataset['data_symp_groups_all'].columns, best_individual[:self.n_features]) if bit]
        except KeyError:
            best_columns = []

        return best_individual, best_fitnesses, variances, best_columns, self.n_features




if __name__ == "__main__":
    # Load the dataset
    dataset_path = "/Users/leo/Programming/PLR/Leo/data/non_binary_data_processed.csv"
    dataset = {
        'data_symp_groups_all': pd.read_csv(dataset_path, index_col=0)
    }

    # Setting up device for GPU usage if available
    device = torch.device("cpu")

    # Instantiate and move the autoencoder to the device
    input_size = len(dataset['data_symp_groups_all'].columns)
    autoencoder = Autoencoder(input_size=input_size, depth=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Convert dataset to tensor and create a DataLoader for it
    data = dataset['data_symp_groups_all'].values
    data_tensor = torch.FloatTensor(data).to(device)
    dataset_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)

    # Train the autoencoder with tqdm for progress visualization
    epochs = 50
    for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
        for batch_data, in dataset_loader:
            optimizer.zero_grad()
            outputs = autoencoder(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()

    # Now, you should pass the trained autoencoder or its encoder part to your GeneticAlgorithm.
    # For this example, I'm just instantiating the GA and running it.
    # Ensure that the GeneticAlgorithm uses the encoder for dimensionality reduction.
    ga = GeneticAlgorithm(load_data(), population_size=150)  # Example population size

    # Run the Genetic Algorithm
    best_individual, best_fitnesses, variances, best_columns, n_features = ga.run()

    # Display results
    print("Best Individual:", best_individual)
    print("Best Fitnesses:", best_fitnesses)
    print("Variance:", variances)
    print("Best Columns:", best_columns)
    print("Number of Features:", n_features)
