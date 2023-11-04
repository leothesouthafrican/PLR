import numpy as np
from joblib import Parallel, delayed
from helpers import perform_hdbscan
from datetime import datetime
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import silhouette_score
from selection import roulette_wheel_selection, tournament_selection, rank_selection, elitism_selection
from crossover import one_point_crossover, two_point_crossover, uniform_crossover
from autoencoder import Autoencoder


class GeneticAlgorithm:
    def __init__(self, dataset, population_size=100, n_generations=20, selection_rate=0.3, mutation_rate=0.15, 
                 increased_mutation_rate=0.2, num_elites=None, depth_range=(1,5), hidden_dim_range=(64, 512),
                 n_epochs=20, score_metric=silhouette_score, clustering_algo="hdbscan", parent_selection_method="roulette",
                 crossover_method="one_point",min_cluster_size_range=(2, 50), learning_rate=0.001, batch_size=64,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), n_jobs = -1):
        
        self.dataset = dataset
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.increased_mutation_rate = increased_mutation_rate
        self.generations_without_improvement = 0
        self.num_elites = num_elites or int(0.1 * population_size)
        self.n_features = len(dataset.columns)
        self.depth_range = depth_range
        self.hidden_dim_range = hidden_dim_range
        self.n_epochs = n_epochs
        self.score_metric = score_metric
        self.parent_selection_method = parent_selection_method
        self.crossover_method = crossover_method
        self.clustering_algo = clustering_algo
        self.min_cluster_size_range = min_cluster_size_range
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.n_jobs = n_jobs

    def _get_config_params(self):
        attribute_names = ["population_size", "n_generations", "selection_rate", "mutation_rate", 
                        "increased_mutation_rate", "num_elites", "n_features", 
                        "depth_range", "hidden_dim_range", "n_epochs", "score_metric", 
                        "parent_selection_method", "crossover_method", "clustering_algo", 
                        "min_cluster_size_range", "learning_rate", "batch_size", "device"]
        return {attr: str(getattr(self, attr)) for attr in attribute_names}

    def fitness(self, params):
        selected_features = params[:self.n_features]
        min_cluster_size = int(params[self.n_features + 2])
        
        selected_cols = [col for col, keep in zip(self.dataset.columns, selected_features) if keep]
        if not selected_cols:
            return -np.inf
        dataset = self.dataset[selected_cols].dropna()

        input_size = len(dataset.columns)

        # Extract depth from params
        depth = int(params[self.n_features])

        # Extract hidden dim from params
        first_hidden_dim = int(params[self.n_features + 1])

        autoencoder = Autoencoder(input_size=input_size, first_hidden_dim=first_hidden_dim, depth=depth).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        
        data = dataset.values
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset_loader = DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for batch_data, in dataset_loader:
                optimizer.zero_grad()
                outputs = autoencoder(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            latent_data = autoencoder.encode(data_tensor).cpu().numpy()

        if self.clustering_algo == "hdbscan":
            labels = perform_hdbscan(latent_data, min_cluster_size=min_cluster_size)
        else:
            pass

        if len(set(labels)) > 1:
            fitness_value = self.score_metric(latent_data, labels)
        else:
            fitness_value = -1
        return fitness_value


    def init_population(self):
        population = []
        for _ in range(self.population_size):
            features = np.random.choice([0, 1], size=self.n_features)
            
            # Get depth from depth_range
            depth = np.random.randint(self.depth_range[0], self.depth_range[1] + 1)

            # Get the hidden dim from hidden_dim_range that's a multiple of 32 (I made certain to make it multiples of 32 so that if the depth increases,
            # the final latent space will always be "predictable" or part of a set)
            possible_hidden_dims = [i for i in range(self.hidden_dim_range[0], self.hidden_dim_range[1] + 1) if i % 32 == 0]
            hidden_dim = np.random.choice(possible_hidden_dims)

            # Get the min_cluster_size from min_cluster_size_range
            min_cluster_size = np.random.randint(self.min_cluster_size_range[0], self.min_cluster_size_range[1] + 1)

            chromosome = np.concatenate((features, [depth, hidden_dim, min_cluster_size]))
            population.append(chromosome)
        return population

    def select_parents(self, population, fitness_values):
        if self.parent_selection_method == "roulette":
            return roulette_wheel_selection(population, fitness_values)
        elif self.parent_selection_method == "tournament":
            return tournament_selection(population, fitness_values)
        elif self.parent_selection_method == "rank":
            return rank_selection(population, fitness_values)
        elif self.parent_selection_method == "elitism":
            return elitism_selection(population, fitness_values)
        else:
            raise ValueError(f"Invalid selection method: {self.parent_selection_method}")


    def mutate(self, individual):
        mutation_occurred = False
        # Determine mutation rate
        current_mutation_rate = self.mutation_rate if self.generations_without_improvement < 3 else self.increased_mutation_rate
        print(f"Current mutation rate: {current_mutation_rate}")
        
        # The reason that we have chosen to break this up into 4 steps instead of just inplementing a single if statement is so that
        # don't always mutate the entire genome of the individual, mutating, by chance, only a small part. 

        # 1. Mutate features
        if np.random.rand() < current_mutation_rate:
            mutation_occurred = True
            individual = self._mutate_features(individual)
        
        # 2. Mutate depth
        if np.random.rand() < current_mutation_rate:
            mutation_occurred = True
            individual = self._mutate_depth(individual)
        
        # 3. Mutate hidden dimensions
        if np.random.rand() < current_mutation_rate:
            mutation_occurred = True
            individual = self._mutate_hidden_dim(individual)
        
        # 4. Mutate minimum cluster size
        if np.random.rand() < current_mutation_rate:
            mutation_occurred = True
            individual = self._mutate_min_cluster_size(individual)
        
        return individual, mutation_occurred

    def _mutate_features(self, individual):
        """Randomly flip a feature's inclusion/exclusion status."""
        idx = np.random.randint(0, self.n_features)
        individual[idx] = 1 - individual[idx]
        return individual

    def _mutate_depth(self, individual):
        """Change depth using Gaussian distribution centered around current depth."""
        mutation_strength = 1
        delta = int(np.random.normal(0, mutation_strength))
        individual[self.n_features] += delta
        individual[self.n_features] = np.clip(individual[self.n_features], self.depth_range[0], self.depth_range[1])
        return individual

    def _mutate_hidden_dim(self, individual):
        """Change hidden dim in steps of 32 but based on a Gaussian distribution."""
        mutation_strength = 32  # I chose 32 initially but it is possible to change this to smaller or larger jumps
        # Get delta from Gaussian distribution
        delta = int(np.round(np.random.normal(0, mutation_strength) / 32)) * 32
        # Add delta to hidden dim
        individual[self.n_features + 1] += delta
        # Make sure hidden dim is within the range
        individual[self.n_features + 1] = np.clip(individual[self.n_features + 1], self.hidden_dim_range[0], self.hidden_dim_range[1])
        return individual

    def _mutate_min_cluster_size(self, individual):
        """Change min cluster size using Gaussian distribution."""
        mutation_strength = 2  
        delta = int(np.random.normal(0, mutation_strength))
        individual[self.n_features + 2] += delta
        individual[self.n_features + 2] = np.clip(individual[self.n_features + 2], self.min_cluster_size_range[0], self.min_cluster_size_range[1])
        return individual

    def crossover(self, parent1, parent2):
        if self.crossover_method == "one_point":
            return one_point_crossover(parent1, parent2)
        elif self.crossover_method == "two_point":
            return two_point_crossover(parent1, parent2)
        elif self.crossover_method == "uniform":
            return uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")

    def run(self):
        run_name = f"GA_run_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        # wandb.init(project="PLR", name=run_name)
        # wandb.config.update(self._get_config_params())
        columns = ["Generation Number", "Best Genome", "Best Features", "Fitness"]
        # generation_table = wandb.Table(columns=columns)
        population = self.init_population()

        prev_best_fitness = -np.inf  # Keep track of best fitness from previous generation
        generations_data = []  # Store the results for each generation which will then be stored in the JSON file

        for generation in tqdm(range(self.n_generations)):
            fitness_values = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness)(individual) for individual in population)
            
            generation_results = {
                "generation_number": generation + 1,
                "individuals": []
            }
            
            for fitness_val, individual in zip(fitness_values, population):
                genome_data = {
                    "genome": individual.tolist(),
                    "fitness": fitness_val,
                    "features": [col for col, keep in zip(self.dataset.columns, individual[:self.n_features]) if keep]
                }
                generation_results["individuals"].append(genome_data)

            generations_data.append(generation_results)
            
            best_index = np.argmax(fitness_values)
            best_genome_this_gen = population[best_index]
            best_features = [col for col, keep in zip(self.dataset.columns, best_genome_this_gen[:self.n_features]) if keep]
            # generation_table.add_data(generation+1, best_genome_this_gen.tolist(), best_features, fitness_values[best_index])
            
            # Check if best fitness improved
            if fitness_values[best_index] > prev_best_fitness:
                prev_best_fitness = fitness_values[best_index]
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

            new_population = [best_genome_this_gen]
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_values)
                child1, child2 = self.crossover(parent1, parent2)
                mutated_child1, _ = self.mutate(child1)
                mutated_child2, _ = self.mutate(child2)
                new_population.extend([mutated_child1, mutated_child2])

            population = new_population[:self.population_size]

        # wandb.log({"Generations Table": generation_table})
        # wandb.finish()

        return {
            "parent_selection_method": self.parent_selection_method,
            "crossover_method": self.crossover_method,
            "generations": generations_data
        }

