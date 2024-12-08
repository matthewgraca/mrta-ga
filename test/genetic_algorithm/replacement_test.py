import unittest
from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

class ReplacementTest(unittest.TestCase):
    def test_replace_worst_1(self):
        np.random.seed(0)

        env = EnvironmentInitializer(
            robots=3, tasks=12, robot_loc=[(3, 8), (5, 3), (20, 19)]
        )
        ga = GeneticAlgorithm(replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)

        # sort population by fitness
        pop_fitness, pop = ga._GeneticAlgorithm__sort_pop_by_fitness(pop)
        
        # 20% hardcoded, so only 40 should remain
        actual = ga._GeneticAlgorithm__replacement(pop, pop_fitness)
        expected = pop[10:]
        self.assertTrue(np.array_equal(actual, expected))

