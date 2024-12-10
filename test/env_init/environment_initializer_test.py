import unittest
from src.environment_initializer import EnvironmentInitializer

class EnvironmentInitializerTest(unittest.TestCase):
    def test_robot_loc_read_properly(self):
        robots = 3
        tasks = 10
        env = EnvironmentInitializer(robots, tasks)
        actual = env.get_robot_loc()
        expected = [(1,5), (1,9), (1,12)]
        self.assertEqual(actual, expected)

    def test_accessing_task_list(self):
        robots = 3
        tasks = 10
        env = EnvironmentInitializer(robots, tasks)
        actual = env.get_task_list()
        expected = [
            [(22, 17), (15, 31)],
            [(9, 1), (31, 47)],
            [(22, 45), (12, 1)],
            [(1, 44), (7, 22)],
            [(7, 45), (13, 16)],
            [(1, 47), (31, 44)],
            [(19, 26), (10, 8)],
            [(19, 8), (31, 23)],
            [(1, 26), (31, 26)],
            [(1, 12), (25, 17)]
        ]

        self.assertEqual(actual, expected)
