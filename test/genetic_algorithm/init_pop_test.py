import numpy as np
import unittest
from src.genetic_algorithm import GeneticAlgorithm

class InitPopTest(unittest.TestCase):
    def test_chromosome(self):
        np.random.seed(0)
        ga = GeneticAlgorithm()
        actual = ga._GeneticAlgorithm__create_two_part_chromosome(10, 3).tolist()
        expected = [3, 9, 5, 10, 2, 7, 8, 4, 1, 6, 5, 4, 1]
        self.assertEqual(expected, actual)
