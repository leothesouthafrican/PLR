from genetic_algorithm import GeneticAlgorithm

import unittest
import numpy as np
import pandas as pd

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing purposes
        self.dataset = {'data_symp_groups_all': pd.DataFrame(np.random.rand(1000, 5))}
        self.ga = GeneticAlgorithm(dataset=self.dataset)
        
    def test_init_population(self):
        population = self.ga.init_population()
        self.assertEqual(len(population), self.ga.population_size)
        
        for individual in population:
            self.assertEqual(len(individual), self.ga.n_features + 3)
            self.assertTrue(set(individual[:self.ga.n_features]) <= {0, 1})

    def test_fitness_function(self):
        # Given the stochastic nature of the GA, we mainly want to ensure the fitness function runs without errors
        individual = self.ga.init_population()[0]
        fitness_val = self.ga.fitness(individual)
        self.assertIsInstance(fitness_val, float)
        
    def test_mutation(self):
        individual = self.ga.init_population()[0]
        mutated_individual = self.ga.mutate(np.copy(individual))
        self.assertNotEqual(individual.tolist(), mutated_individual.tolist())
        
    def test_crossover(self):
        parent1, parent2 = self.ga.init_population()[:2]
        child1, child2 = self.ga.crossover(parent1, parent2)
        self.assertNotEqual(parent1.tolist(), child1.tolist())
        self.assertNotEqual(parent2.tolist(), child2.tolist())
        
    def test_select_parents(self):
        population = self.ga.init_population()
        fitness_values = [self.ga.fitness(individual) for individual in population]
        parent1, parent2 = self.ga.select_parents(population, fitness_values)
        self.assertTrue(parent1 in population)
        self.assertTrue(parent2 in population)
        
    def test_entire_run(self):
        best_individual, best_fitness = self.ga.run()
        self.assertEqual(len(best_individual), self.ga.n_features + 3)
        self.assertIsInstance(best_fitness, float)
        

if __name__ == "__main__":
    unittest.main()