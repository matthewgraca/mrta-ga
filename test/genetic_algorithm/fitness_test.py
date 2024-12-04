import unittest
from src.genetic_algorithm import GeneticAlgorithm

class FitnessTest(unittest.TestCase):
    def test_x(self):
        ga = GeneticAlgorithm()

        task_list = ga.access_tasks('maps/task_list.json')
        grid = ga.access_map('maps/warehouse_small.txt')
        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]
        tasks, robots = 12, 3
        robot_loc = [(3, 8), (5, 3), (20, 19)]
        actual = ga._GeneticAlgorithm__fitness(task_list, grid, chromosome, tasks, robots, robot_loc)
        self.assertEqual(actual, 773)
