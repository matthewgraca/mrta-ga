import unittest
from src.environment_initializer import EnvironmentInitializer
from src.genetic_algorithm import GeneticAlgorithm

class FitnessTest(unittest.TestCase):
    def test_flow_time(self):
        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(objective_func='flow_time', env=env)

        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]

        actual = ga._GeneticAlgorithm__flow_time(chromosome)
        expected = 770 
        self.assertEqual(actual, expected)

    def test_makespan(self):
        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(objective_func='makespan', env=env)

        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]

        actual = ga._GeneticAlgorithm__makespan(chromosome)
        expected = 330 
        self.assertEqual(actual, expected)

    def test_collision_detection_simple_no_collision(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,1)],[(0,2)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 0
        self.assertEqual(actual, expected)

    def test_collision_detection_with_padding(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,1), (2,2)],[(0,2)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 0
        self.assertEqual(actual, expected)

    def test_collision_detection_simple_yes_collision_1(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,1)],[(0,0)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 1
        self.assertEqual(actual, expected)

    def test_collision_detection_simple_yes_collision_2(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,0)],[(0,0)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 2
        self.assertEqual(actual, expected)

    def test_collision_detection_yes_collision_with_padding_1(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,1), (2,2)],[(0,2), (2,2)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 1
        self.assertEqual(actual, expected)

    def test_collision_detection_with_padding_same_cell_diff_loc(self):
        ga = GeneticAlgorithm()
        tours = [[(0,0)],[(0,1), (2,2)],[(2,2)]]
        actual = ga._GeneticAlgorithm__constraint_collision(tours)
        expected = 0
        self.assertEqual(actual, expected)

    def test_collision_detection_large_example(self):
        env = EnvironmentInitializer(robots=3, tasks=12)
        ga = GeneticAlgorithm(objective_func='flow_time', env=env)

        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]
        subtours = ga._GeneticAlgorithm__fitness_get_all_subtours(chromosome)
        actual = ga._GeneticAlgorithm__constraint_collision(subtours)
        expected = 1 
        self.assertEqual(actual, expected)

        
