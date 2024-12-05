import unittest
from src.environment_initializer import EnvironmentInitializer
from src.genetic_algorithm import GeneticAlgorithm

class FitnessTest(unittest.TestCase):
    def test_flow_time(self):
        env = EnvironmentInitializer(
            robots=3, tasks=12, robot_loc=[(3, 8), (5, 3), (20, 19)],
            task_path = 'maps/task_list.json',
            grid_path = 'maps/warehouse_small.txt'
        )
        ga = GeneticAlgorithm(objective_func='flow_time', env=env)

        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]

        actual = ga._GeneticAlgorithm__fitness(chromosome)
        expected = 1 / 257
        self.assertAlmostEqual(actual, expected, delta=1/257)

    def test_makespan(self):
        env = EnvironmentInitializer(
            robots=3, tasks=12, robot_loc=[(3, 8), (5, 3), (20, 19)],
            task_path = 'maps/task_list.json',
            grid_path = 'maps/warehouse_small.txt'
        )
        ga = GeneticAlgorithm(objective_func='makespan', env=env)

        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]

        actual = ga._GeneticAlgorithm__fitness(chromosome)
        expected = 1 / 325 
        self.assertEqual(actual, expected)
