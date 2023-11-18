import numpy as np
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
from datetime import datetime
# import wandb
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from selection import roulette_wheel_selection, tournament_selection, rank_selection, elitism_selection
from crossover import one_point_crossover, two_point_crossover, uniform_crossover

class GeneticAlgorithm:
    def __init__(self, dataset, population_size=100, n_generations=20, selection_rate=0.3, mutation_rate=0.15, 
                 increased_mutation_rate=0.2, num_elites=None, perplexity_range=(5, 50), 
                 learning_rate_range=(10, 1000), n_iter_range=(250, 1000), score_metric=silhouette_score,
                 parent_selection_method="roulette", crossover_method="one_point", min_cluster_size_range=(2, 50), n_jobs=-1):
        
        self.dataset = dataset
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.increased_mutation_rate = increased_mutation_rate
        self.generations_without_improvement = 0
        self.num_elites = num_elites or int(0.1 * population_size)
        self.n_features = len(dataset.columns)
        self.perplexity_range = perplexity_range
        self.learning_rate_range = learning_rate_range
        self.n_iter_range = n_iter_range
        self.score_metric = score_metric
        self.parent_selection_method = parent_selection_method
        self.crossover_method = crossover_method
        self.min_cluster_size_range = min_cluster_size_range
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
        perplexity = int(params[self.n_features])
        learning_rate = int(params[self.n_features + 1])
        n_iter = int(params[self.n_features + 2])

        selected_cols = [col for col, keep in zip(self.dataset.columns, selected_features) if keep]
        if not selected_cols:
            return -np.inf
        dataset = self.dataset[selected_cols].dropna()

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(dataset.values)

        if len(set(tsne_results)) > 1:
            fitness_value = self.score_metric(dataset.values, tsne_results)
        else:
            fitness_value = -1
        return fitness_value

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            features = np.random.choice([0, 1], size=self.n_features)
            perplexity = np.random.randint(self.perplexity_range[0], self.perplexity_range[1] + 1)
            learning_rate = np.random.randint(self.learning_rate_range[0], self.learning_rate_range[1] + 1)
            n_iter = np.random.randint(self.n_iter_range[0], self.n_iter_range[1] + 1)

            chromosome = np.concatenate((features, [perplexity, learning_rate, n_iter]))
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

