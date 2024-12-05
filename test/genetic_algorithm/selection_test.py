import unittest
from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer

class SelectionTest(unittest.TestCase):
    def test_rws_1(self):
        env = EnvironmentInitializer(
            robots=3, tasks=12, robot_loc=[(3, 8), (5, 3), (20, 19)]
        )
        ga = GeneticAlgorithm(pop_init='random', selection='rws', env=env)
        pop = ga._GeneticAlgorithm__pop_init(500)
        
        actual = ga._GeneticAlgorithm__selection(pop)
        expected = 0
        self.assertEqual(actual, expected)
