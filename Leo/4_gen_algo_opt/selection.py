import numpy as np
import random

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    pick = random.uniform(0, total_fitness)
    current = 0
    for chromosome, fitness in zip(population, fitness_values):
        current += fitness
        if current > pick:
            parent1 = chromosome
            break
    else:  # In case we exhaust the loop without breaking
        parent1 = population[-1]
    
    # Repeat the process again for the second parent
    pick = random.uniform(0, total_fitness)
    current = 0
    for chromosome, fitness in zip(population, fitness_values):
        current += fitness
        if current > pick:
            parent2 = chromosome
            break
    else:  # In case we exhaust the loop without breaking
        parent2 = population[-1]

    return parent1, parent2


def tournament_selection(population, fitness_values, tournament_size=5):
    # First parent
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_chromosomes = [population[i] for i in selected_indices]
    tournament_fitness = [fitness_values[i] for i in selected_indices]
    winner_index1 = selected_indices[np.argmax(tournament_fitness)]
    
    # Second parent
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_chromosomes = [population[i] for i in selected_indices]
    tournament_fitness = [fitness_values[i] for i in selected_indices]
    winner_index2 = selected_indices[np.argmax(tournament_fitness)]
    
    return population[winner_index1], population[winner_index2]

def rank_selection(population, fitness_values):
    sorted_indices = np.argsort(fitness_values)
    pick1, pick2 = random.randint(0, len(population)-1), random.randint(0, len(population)-1)
    return population[sorted_indices[pick1]], population[sorted_indices[pick2]]

def elitism_selection(population, fitness_values):
    elite_index = np.argmax(fitness_values)
    # Return the elite chromosome and a random one for diversity
    random_index = np.random.randint(0, len(population))
    return population[elite_index], population[random_index]
