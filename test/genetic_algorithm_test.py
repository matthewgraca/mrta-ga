import unittest
from src.genetic_algorithm import GeneticAlgorithm

class GeneticAlgorithmTest(unittest.TestCase):
    def test_1(self):
        ga = GeneticAlgorithm()
        self.assertTrue(ga.test() == 0)
