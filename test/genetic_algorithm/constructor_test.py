import unittest
from src.environment_initializer import EnvironmentInitializer
from src.genetic_algorithm import GeneticAlgorithm

class ConstructorTest(unittest.TestCase):
    '''
    Test initialization, parameters
    '''
    def test_init(self):
        ga = GeneticAlgorithm()
        actual = ga.get_params()
        expected = {
            'objective_func':'flow_time',
            'pop_size' : 100,
            'pop_init' : 'random',
            'selection': 'rws',
            'crossover' : 'tcx',
            'mutation' : 'inverse',
            'pc' : 0.4,
            'pm' : 0.6,
            'replacement' : 'replace_worst' 
        }
        self.assertEqual(actual, expected)
   
    def test_valid_parameters(self):
        ga = GeneticAlgorithm(pop_size=1000, pc=0.1, pm=0.3)
        actual = ga.get_params()
        expected = {
            'objective_func':'flow_time',
            'pop_size' : 1000,
            'pop_init' : 'random',
            'selection': 'rws',
            'crossover' : 'tcx',
            'mutation' : 'inverse',
            'pc' : 0.1,
            'pm' : 0.3,
            'replacement' : 'replace_worst'
        }
        self.assertEqual(actual, expected)

    def test_pop_size_past_max(self):
        self.assertRaises(ValueError, GeneticAlgorithm, pop_size=1000000)
        ga = GeneticAlgorithm

    def test_pop_size_past_min(self):
        self.assertRaises(ValueError, GeneticAlgorithm, pop_size=1)
        ga = GeneticAlgorithm

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

    def test_invalid_objective_func_parameter(self):
        self.assertRaises(ValueError, GeneticAlgorithm, objective_func='buh')

    def test_all_valid_params(self):
        expected = {
            'objective_func': {'makespan', 'flow_time'},
            'pop_init'      : {'random', 'greedy'},
            'selection'     : {'rws'},
            'crossover'     : {'tcx'},
            'mutation'      : {'inverse', 'swap'},
            'replacement'   : {'replace_worst'}
        }  

        actual = GeneticAlgorithm().get_valid_params()
        self.assertEqual(actual, expected)
