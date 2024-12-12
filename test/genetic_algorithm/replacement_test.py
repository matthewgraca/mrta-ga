import unittest
from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

class ReplacementTest(unittest.TestCase):
    def test_replace_worst_1(self):
        np.random.seed(0)

        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(pop_size=50, replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fit = ga._GeneticAlgorithm__fitness_of_pop(pop)
        expected1, expected2 = ga._GeneticAlgorithm__sort_pop_by_fitness(pop_fit, pop)

        # 20% hardcoded, so only 40 should remain
        actual1, actual2 = ga._GeneticAlgorithm__replacement(pop, pop_fit)

        self.assertTrue(np.array_equal(actual1, expected1[10:]))
        self.assertTrue(np.array_equal(actual2, expected2[10:]))

    def test_elitism_old_is_in_front(self):
        np.random.seed(0)

        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(pop_size=50, replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fit = ga._GeneticAlgorithm__fitness_of_pop(pop)

        # the idea is if the fitnesses are sorted, then:
        # best is extracted, first 10 removed, last 1 removed, 
        # then best is back in front
        a, b = ga._GeneticAlgorithm__sort_pop_by_fitness(pop_fit, pop)
        a, b = np.flip(a), np.flip(b)
        best1, best2 = a[0], b[0]
        expected1 = np.append(best1, a[10:49].copy())
        expected2 = np.append([best2], b[10:49].copy(), axis=0)
        actual1, actual2 = ga._GeneticAlgorithm__elitism(b, a)
        
        self.assertTrue(np.array_equal(actual1, expected1))
        self.assertTrue(np.array_equal(actual2, expected2))

    def test_elitism_old_is_not_in_front(self):
        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(pop_size=50, replacement='replace_worst', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fit = ga._GeneticAlgorithm__fitness_of_pop(pop)

        # idea is if sorted, then best fits are at the back, and it will 
        # simply replace the first 10
        a, b = ga._GeneticAlgorithm__sort_pop_by_fitness(pop_fit, pop)
        expected1, expected2 = a[10:].copy(), b[10:].copy()
        actual1, actual2 = ga._GeneticAlgorithm__elitism(b, a)

        self.assertTrue(np.array_equal(actual1, expected1))
        self.assertTrue(np.array_equal(actual2, expected2))
