import unittest
from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

class SelectionTest(unittest.TestCase):
    def test_rws_1(self):
        np.random.seed(0)
        env = EnvironmentInitializer(
            robots=3, tasks=12, robot_loc=[(3, 8), (5, 3), (20, 19)]
        )
        ga = GeneticAlgorithm(pop_init='random', selection='rws', env=env)
        pop = ga._GeneticAlgorithm__pop_init(50)
        pop_fitness = ga._GeneticAlgorithm__fitness_of_pop(pop, constraint=True)
        pop_fit, pop = ga._GeneticAlgorithm__sort_pop_by_fitness(pop_fitness, pop)
        
        actual = ga._GeneticAlgorithm__selection(pop, pop_fit)
        expected = [
            [ 6,  8, 11, 12,  4,  7,  1,  3,  5,  2,  9, 10,  2,  8,  2], 
            [ 8,  3,  5, 12,  7,  4, 11, 10,  2,  9,  1,  6,  5,  1,  6], 
            [ 2,  5,  8, 10,  7, 12, 11,  6,  4,  9,  3,  1,  2,  1,  9], 
            [ 6,  1,  2,  9,  5, 11,  3,  7,  4,  8, 12, 10,  8,  2,  2], 
            [ 6,  8, 11, 12,  4,  7,  1,  3,  5,  2,  9, 10,  2,  8,  2], 
            [ 3,  7,  6, 11, 10, 12,  8,  4,  5,  1,  9,  2,  9,  1,  2], 
            [ 1,  9, 12,  5,  2,  6,  7,  3, 11,  8,  4, 10,  5,  6,  1], 
            [ 2,  1,  3, 11, 10, 12,  6,  4,  8,  5,  7,  9,  8,  1,  3], 
            [ 8,  1,  9,  2,  5,  3,  4, 11, 12, 10,  6,  7,  3,  5,  4], 
            [ 9, 11,  5,  3, 12,  4, 10,  1,  6,  7,  2,  8,  2,  7,  3]
        ]
        self.assertTrue(np.array_equal(actual, expected))

    def test_lambda_normal(self):
        ga = GeneticAlgorithm()
        actual = ga._GeneticAlgorithm__get_lambda(
            pop_size=100, mating_pool_prop=0.2
        )
        expected = 20
        self.assertEqual(actual, expected)

    def test_lambda_floor(self):
        ga = GeneticAlgorithm()
        actual = ga._GeneticAlgorithm__get_lambda(
            pop_size=10, mating_pool_prop=0.1
        )
        expected = 2
        self.assertEqual(actual, expected)
