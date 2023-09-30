import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from cluster_comparison import perform_hdbscan, calculate_silhouette
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
    def __init__(self, dataset, population_size=100, n_generations=20, max_depth = 5, selection_rate=0.3, mutation_rate=0.05, 
                increased_mutation_rate=0.2, num_elites=None, depth_range=(1,5), latent_dim_range=(2, 128)):
        self.dataset = dataset
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.increased_mutation_rate = increased_mutation_rate
        self.num_elites = num_elites or int(0.1 * population_size)
        self.n_features = len(dataset['data_symp_groups_all'].columns)
        self.max_depth = max_depth
        self.depth_range = depth_range
        self.latent_dim_range = latent_dim_range
        self.fitness_cache = {}
        self.last_best_fitness = -1

    def fitness(self, params):
        selected_features = params[:self.n_features]
        depth = max(1, int(self.max_depth))
        latent_dim = max(2, int(params[self.n_features + 1]))

        dataset_name, dataset = random.choice(list(self.dataset.items()))
        dataset = dataset[[col for col, keep in zip(dataset.columns, selected_features) if keep]].dropna()

        input_size = len(dataset.columns)
        autoencoder = Autoencoder(input_size=input_size, depth=depth).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        data = dataset.values
        data_tensor = torch.FloatTensor(data).to(device)
        dataset_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)

        epochs = 5
        for epoch in range(epochs):
            for batch_data, in dataset_loader:
                optimizer.zero_grad()
                outputs = autoencoder(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()

        fitness_value = -float(loss.item())

        return fitness_value

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
            #... (continue with rest of best_feature_vector)
        )

        best_individual = None
        best_individual_fitness = float('-inf')

        population = self.initialize_population(best_feature_vector)
        best_fitness_log = []

        for generation in tqdm(range(self.n_generations)):
            # Evaluation
            fitness_values = Parallel(n_jobs=-1)(delayed(self.fitness)(ind) for ind in population)

            # Selecting the best individuals in the current generation
            sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
            elites = sorted_population[:self.num_elites]
            elite_fitness = sum([self.fitness(elite) for elite in elites]) / self.num_elites
            best_fitness_log.append(elite_fitness)
            if generation > 0 and best_fitness_log[-1] <= best_fitness_log[-2]:
                self.mutation_rate = self.increased_mutation_rate
            else:
                self.mutation_rate = 0.05

            if elite_fitness > best_individual_fitness:
                best_individual = elites[0]
                best_individual_fitness = elite_fitness

            # Selection
            selected_parents = sorted_population[:int(self.population_size * self.selection_rate)]

            # Crossover and Mutation
            children = []
            while len(children) < self.population_size - len(elites):
                parent1 = random.choice(selected_parents)
                parent2 = random.choice(selected_parents)
                crossover_point = random.randint(1, len(parent1) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1, best_individual)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2, best_individual)

                children.append(child1)
                children.append(child2)

            # Creating the new population
            population = elites + children

            wandb.log({
                'Generation': generation,
                'Best Fitness': best_individual_fitness,
                'Elite Fitness': elite_fitness
            })

        return best_individual, best_individual_fitness




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
