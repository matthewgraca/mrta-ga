import unittest
from src.genetic_algorithm import GeneticAlgorithm

class FitnessTest(unittest.TestCase):
    def test_x(self):
        ga = GeneticAlgorithm()
        actual = ga._GeneticAlgorithm__fitness(0, 0, 0, 0, 0)
        self.assertEqual(actual, 0)
