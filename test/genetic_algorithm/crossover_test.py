import unittest
from src.genetic_algorithm import GeneticAlgorithm
import numpy as np

class CrossoverTest(unittest.TestCase):
    def test_xover1(self):
        ga = GeneticAlgorithm()
        actual1, actual2 = ga._GeneticAlgorithm__two_part_crossover(np.array([0]), np.array([0]))
        actual1.tolist()
        actual2.tolist()
        expected1, expected2 = [0], [0]
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
