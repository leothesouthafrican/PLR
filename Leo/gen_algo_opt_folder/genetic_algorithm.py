import random
import numpy as np
from joblib import Parallel, delayed
from cluster_comparison import perform_hdbscan
from data_loading import load_data
from datetime import datetime
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from selection import roulette_wheel_selection, tournament_selection, rank_selection, elitism_selection
from crossover import one_point_crossover, two_point_crossover, uniform_crossover

from autoencoder import Autoencoder


class GeneticAlgorithm:
    def __init__(self, dataset, population_size=100, n_generations=20, max_depth=5, selection_rate=0.3, mutation_rate=0.05, 
                 increased_mutation_rate=0.2, num_elites=None, depth_range=(1,5), latent_dim_range=(2, 128),
                 n_epochs=20, score_metric=silhouette_score, clustering_algo="hdbscan", parent_selection_method="roulette",
                 crossover_method="one_point",min_cluster_size_range=(2, 50), learning_rate=0.001, batch_size=64,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), n_jobs = -1):
        
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
        self.n_epochs = n_epochs
        self.score_metric = score_metric
        self.parent_selection_method = parent_selection_method
        self.crossover_method = crossover_method
        self.clustering_algo = clustering_algo
        self.min_cluster_size_range = min_cluster_size_range
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fitness_cache = {}
        self.last_best_fitness = -1
        self.device = device
        self.n_jobs = n_jobs

    def _get_config_params(self):
        # Extract the names of the attributes in order to log them to wandb
        attribute_names = ["population_size", "n_generations", "selection_rate", "mutation_rate", 
                        "increased_mutation_rate", "num_elites", "n_features", "max_depth", 
                        "depth_range", "latent_dim_range", "n_epochs", "score_metric", 
                        "parent_selection_method", "crossover_method", "clustering_algo", 
                        "min_cluster_size_range", "learning_rate", "batch_size", "device"]
                        
        # Use a dictionary comprehension to get a dict of attribute names and their values
        return {attr: str(getattr(self, attr)) for attr in attribute_names}

    def fitness(self, params):
        # Feature Selection
        selected_features = params[:self.n_features]
        depth = max(1, self.max_depth)
        min_cluster_size = int(params[self.n_features + 2])
        
        # Randomly select a dataset
        dataset_name, dataset = random.choice(list(self.dataset.items()))
        dataset = dataset[[col for col, keep in zip(dataset.columns, selected_features) if keep]].dropna()

        # Initialize and train the autoencoder
        input_size = len(dataset.columns)
        autoencoder = Autoencoder(input_size=input_size, depth=depth).to(self.device)
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
        # Extract latent features
        with torch.no_grad():
            latent_data = autoencoder.encode(data_tensor).cpu().numpy()

        # Apply the specified clustering algorithm
        if self.clustering_algo == "hdbscan":
            labels = perform_hdbscan(latent_data, min_cluster_size=min_cluster_size)
        else:
            # Add logic for other clustering algorithms if needed
            pass
        # Compute the specified score metric if more than one cluster is found
        if len(set(labels)) > 1:
            fitness_value = self.score_metric(latent_data, labels)
        else:
            fitness_value = -1  # a penalty for only one cluster
        return fitness_value


    def init_population(self):
        population = []
        
        for _ in range(self.population_size):
            # Binary feature selection
            features = np.random.choice([0, 1], size=self.n_features)
            
            # Depth and latent dimension
            depth = np.random.randint(self.depth_range[0], self.depth_range[1] + 1)
            latent_dim = np.random.randint(self.latent_dim_range[0], self.latent_dim_range[1] + 1)
            
            # Clustering hyperparameter (e.g., min_cluster_size for HDBSCAN)
            min_cluster_size = np.random.randint(self.min_cluster_size_range[0], self.min_cluster_size_range[1] + 1)
            
            chromosome = np.concatenate((features, [depth, latent_dim, min_cluster_size]))
            population.append(chromosome)
        
        return population
    
    def select_parents(self, population, fitness_values):
        
        if self.parent_selection_method == "roulette":
            return roulette_wheel_selection(population, fitness_values)
        
        elif self.parent_selection_method == "tournament":
            return tournament_selection(population, fitness_values)
        
        elif self.parent_selection_method == "rank":
            return rank_selection(population, fitness_values)
        
        elif self.parent_selection_method == "elite":
            return elitism_selection(population, fitness_values)
        
        else:
            raise ValueError("Invalid selection method specified!")

    def mutate(self, individual):

        mutation = False
        # Mutate binary feature selection
        if np.random.rand() < self.mutation_rate:
            for i in range(self.n_features):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = 1 - individual[i]  # Flip the binary value
                    mutation = True
                
        # Mutate depth
        if np.random.rand() < self.mutation_rate:
            individual[self.n_features] += np.random.choice([-1, 1])  # Increment or decrement by 1
            individual[self.n_features] = np.clip(individual[self.n_features], self.depth_range[0], self.depth_range[1])
            mutation = True
            
        # Mutate latent dimension
        if np.random.rand() < self.mutation_rate:
            individual[self.n_features + 1] += np.random.choice([-1, 1])
            individual[self.n_features + 1] = np.clip(individual[self.n_features + 1], self.latent_dim_range[0], self.latent_dim_range[1])
            mutation = True
            
        # Mutate clustering hyperparameter
        if np.random.rand() < self.mutation_rate:
            individual[self.n_features + 2] += np.random.choice([-1, 1])
            individual[self.n_features + 2] = np.clip(individual[self.n_features + 2], self.min_cluster_size_range[0], self.min_cluster_size_range[1])
            
        return individual, mutation
    
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
        # Initialize wandb
        run_name = f"GA_run_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        wandb.init(project="PLR", name=run_name)

        # Log the parameters to wandb.config
        wandb.config.update(self._get_config_params())

        # Define table columns
        columns = ["Generation Number", "Best Genome", "Best Features", "Fitness"]
        generation_table = wandb.Table(columns=columns)

        # Initialize population
        population = self.init_population()

        # Variables to keep track of variance and mutation count
        generation_variances = []

        # Main evolutionary loop
        for generation in tqdm(range(self.n_generations), desc="Generations — Best Fitness So Far {:.4f}".format(self.last_best_fitness)"):
            # Create a tqdm object for fitness evaluation progress
            fitness_eval_pbar = tqdm(population, desc=f"Eval Fitness Gen {generation+1} - Best Fitness: {best_fitness_this_gen}", leave=False)
            
            fitness_values = []
            best_fitness_this_gen = -np.inf
            best_genome_this_gen = None

            fitness_values = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness)(individual) for individual in fitness_eval_pbar)

            for fitness_val, individual in zip(fitness_values, population):
                if fitness_val > best_fitness_this_gen:
                    best_fitness_this_gen = fitness_val
                    best_genome_this_gen = individual
                    fitness_eval_pbar.set_description(f"Eval Fitness Gen {generation+1} - Best Fitness: {best_fitness_this_gen:.4f}")

            # Calculate variance for this generation
            generation_variance = np.var(fitness_values)
            generation_variances.append(generation_variance)

            # Log best results to wandb Table
            best_features = [col for col, keep in zip(self.dataset['data_symp_groups_all'].columns, best_genome_this_gen[:self.n_features]) if keep]
            generation_table.add_data(generation+1, best_genome_this_gen.tolist(), best_features, best_fitness_this_gen)

            # Update the last best fitness (across all generations)
            self.last_best_fitness = max(self.last_best_fitness, best_fitness_this_gen)

            # Generate next population with crossover and mutation
            new_population = [best_genome_this_gen]
            # Variable to keep track of mutation count for this generation
            mutation_count = 0
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_values)
                child1, child2 = self.crossover(parent1, parent2)
                mutated_child1, mutation1bool = self.mutate(child1)
                mutated_child2, mutation2bool = self.mutate(child2)
                mutation_count += int(mutation1bool) + int(mutation2bool)

                new_population.extend([mutated_child1, mutated_child2])

            # Replace the old population with the new one
            population = new_population[:self.population_size]  # Ensure the population doesn't exceed the set size

            # Extract depth and latent dimension from the best genome of this generation
            best_depth = best_genome_this_gen[self.n_features]
            best_latent_dim = best_genome_this_gen[self.n_features + 1]

            # Log metrics to wandb
            wandb.log({
                "generation": generation+1,
                "best_fitness": best_fitness_this_gen,
                "best_depth": best_depth,
                "best_features": best_features,
                "best_latent_dim": best_latent_dim,
                "variance": generation_variance,
                "mutation_count": mutation_count
            })

        # Evaluate final fitness with a progress bar
        final_fitness_values = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness)(individual) for individual in tqdm(population, desc="Final Fitness Evaluation"))

        best_index = np.argmax(final_fitness_values)

        # Log the final table to wandb
        wandb.log({"Generations Table": generation_table})

        # Close the wandb run
        wandb.finish()

        return population[best_index], final_fitness_values[best_index]



if __name__ == "__main__":
    # Load your data using load_data()
    dataset = load_data()

    # Creating a dictionary of hyperparameters
    hyperparameters = {
        "population_size": 160,
        "n_generations": 25,
        "selection_rate": 0.3,
        "mutation_rate": 0.05,
        "increased_mutation_rate": 0.2,
        "num_elites": None,
        "max_depth": 5,
        "depth_range": (1, 5),
        "latent_dim_range": (2, 128),
        "n_epochs": 15,
        "score_metric": silhouette_score,
        "clustering_algo": "hdbscan",
        "parent_selection_method": ["roulette", "tournament", "rank", "elite"],
        "crossover_method": ["one_point", "two_point", "uniform"],
        "min_cluster_size_range": (2, 50),
        "learning_rate": 0.001,
        "batch_size": 64,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "n_jobs": -1
    }

    #testing the GA for all combinations of parent selection and crossover methods
    for parent_selection_method in hyperparameters["parent_selection_method"]:
        for crossover_method in hyperparameters["crossover_method"]:
            print("Parent Selection Method:", parent_selection_method)
            print("Crossover Method:", crossover_method)
            # Create a GeneticAlgorithm instance
            ga = GeneticAlgorithm(dataset=dataset, population_size=hyperparameters["population_size"], n_generations=hyperparameters["n_generations"],
                                selection_rate=hyperparameters["selection_rate"], mutation_rate=hyperparameters["mutation_rate"], 
                                increased_mutation_rate=hyperparameters["increased_mutation_rate"], num_elites=hyperparameters["num_elites"],
                                max_depth=hyperparameters["max_depth"], depth_range=hyperparameters["depth_range"], 
                                latent_dim_range=hyperparameters["latent_dim_range"], n_epochs=hyperparameters["n_epochs"], 
                                score_metric=hyperparameters["score_metric"], clustering_algo=hyperparameters["clustering_algo"], 
                                parent_selection_method=parent_selection_method, crossover_method=crossover_method, 
                                min_cluster_size_range=hyperparameters["min_cluster_size_range"], learning_rate=hyperparameters["learning_rate"], 
                                batch_size=hyperparameters["batch_size"], device=hyperparameters["device"], n_jobs=hyperparameters["n_jobs"])

            # Run the GA
            best_individual, best_fitness = ga.run()
