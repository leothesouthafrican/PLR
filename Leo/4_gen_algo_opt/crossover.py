import numpy as np

def one_point_crossover(parent1, parent2):
    """Performs one-point crossover between two parents."""
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    
    return child1, child2

def two_point_crossover(parent1, parent2):
    """Performs two-point crossover between two parents."""
    crossover_point1, crossover_point2 = sorted(np.random.choice(range(len(parent1)), 2, replace=False))
    child1 = np.concatenate((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.concatenate((parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    
    return child1, child2

def uniform_crossover(parent1, parent2):
    """Performs uniform crossover between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()
    
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
            
    return child1, child2
