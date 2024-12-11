import unittest
from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

class ReplacementTest(unittest.TestCase):
    def test_replace_worst_1(self):
        np.random.seed(0)

        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fitness = ga._GeneticAlgorithm__fitness_of_pop(pop)

        # sort population by fitness
        pop_fitness, pop = ga._GeneticAlgorithm__sort_pop_by_fitness(pop_fitness, pop)
        
        # 20% hardcoded, so only 40 should remain
        actual1, actual2 = ga._GeneticAlgorithm__replacement(pop, pop_fitness)
        expected1, expected2 = pop_fitness[10:], pop[10:]
        self.assertTrue(np.array_equal(actual1, expected1))
        self.assertTrue(np.array_equal(actual2, expected2))

    def test_elitism_1(self):
        np.random.seed(0)

        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fitness = ga._GeneticAlgorithm__fitness_of_pop(pop)
        
        self.assertTrue(True)
