import unittest
from src.genetic_algorithm import GeneticAlgorithm

class ConstructorTest(unittest.TestCase):
    '''
    Test initialization, parameters
    '''
    def test_init(self):
        ga = GeneticAlgorithm()
        actual = ga.get_parameters()
        expected = {
            'pop_size' : 100,
            'pop_init' : 'random',
            'selection': 'sus',
            'crossover' : 'pmx',
            'mutation' : 'inverse',
            'pc' : 0.4,
            'pm' : 0.6,
            'replacement' : 'none' 
        }
        self.assertEqual(actual, expected)
   
    def test_valid_parameters(self):
        ga = GeneticAlgorithm(pop_size=1000, pc=0.1, pm=0.3)
        actual = ga.get_parameters()
        expected = {
            'pop_size' : 1000,
            'pop_init' : 'random',
            'selection': 'sus',
            'crossover' : 'pmx',
            'mutation' : 'inverse',
            'pc' : 0.1,
            'pm' : 0.3,
            'replacement' : 'none'
        }
        self.assertEqual(actual, expected)

    def test_invalid_pop_size_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, pop_size=0)

    def test_invalid_pc_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, pc=-1.0)

    def test_invalid_mutation_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, mutation='buh')

    def test_invalid_pop_init_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, pop_init='buh')

    def test_invalid_selection_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, selection='buh')

    def test_invalid_replacement_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, replacement='buh')
