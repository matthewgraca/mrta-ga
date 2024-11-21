import unittest
from src.genetic_algorithm import GeneticAlgorithm

class GeneticAlgorithmTest(unittest.TestCase):
    '''
        Test initialization, parameters
    '''
    def test_init(self):
        ga = GeneticAlgorithm()
        actual = ga.get_parameters()
        expected = {
            'pop_size' : 100,
            'crossover' : 'pmx',
            'mutation' : 'inverse',
            'pc' : 0.4,
            'pm' : 0.6
        }
        self.assertEqual(actual, expected)

