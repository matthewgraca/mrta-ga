import unittest
from src.genetic_algorithm import GeneticAlgorithm

class InitPopTest(unittest.TestCase):
    def test_func(self):
        ga = GeneticAlgorithm()
        self.assertTrue(
            ga._GeneticAlgorithm__pop_init(method='random', pop_size=100) == 0
        )
