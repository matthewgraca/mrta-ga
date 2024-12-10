import unittest
from src.environment_initializer import EnvironmentInitializer

class EnvironmentInitializerTest(unittest.TestCase):
    def test_robot_loc_read_properly(self):
        robots = 3
        tasks = 10
        env = EnvironmentInitializer(robots, tasks)
        actual = env._EnvironmentInitializer__access_robot_loc()
        expected = [(1,5), (1,9), (1,12)]
        self.assertEqual(actual, expected)
