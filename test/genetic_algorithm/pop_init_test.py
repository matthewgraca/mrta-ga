import numpy as np
from src.environment_initializer import EnvironmentInitializer
import unittest
from src.genetic_algorithm import GeneticAlgorithm

class PopInitTest(unittest.TestCase):
    def test_two_part_chromosome_10_3(self):
        np.random.seed(0)

        # run 100 times for robust measure on stochastic method
        m, n = 10, 3
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='random', env=env)
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_random_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)

    def test_two_part_chromosome_100_13(self):
        np.random.seed(0)

        # run 100 times for robust measure on stochastic method
        m, n = 100, 13
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='random', env=env)
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_random_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)

    def test_two_part_chromosome_1000_50(self):
        np.random.seed(0)

        # run 100 times for robust measure on stochastic method
        m, n = 1000, 50
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='random', env=env)
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_random_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)

    def test_pop_init(self):
        np.random.seed(0)

        size = 500
        tasks, robots = 10, 3
        env = EnvironmentInitializer(robots=robots, tasks=tasks)
        ga = GeneticAlgorithm(pop_init='random', env=env)
        actual = ga._GeneticAlgorithm__pop_init(size)
        self.assertTrue(len(actual) == size)

    def test_greedy_init(self):
        np.random.seed(0)

        size = 100
        tasks, robots = 10, 3
        env = EnvironmentInitializer(robots=robots, tasks=tasks)
        ga = GeneticAlgorithm(pop_init='greedy', env=env)
        actual = ga._GeneticAlgorithm__pop_init(size)
        self.assertTrue(True)

    def test_greedy_two_part_chromosome_10_3(self):
        np.random.seed(0)

        # run 100 times for robust measure on stochastic method
        m, n = 10, 3
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='greedy', env=env)
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_greedy_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)

    def test_greedy_two_part_chromosome_50_13(self):
        np.random.seed(0)

        # run 10 times for robust measure on stochastic method (would run more, but is highly expensive)
        m, n = 50, 13
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='greedy', env=env)
        for _ in range(10):
            actual = ga._GeneticAlgorithm__create_greedy_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)

    def test_greedy_two_part_chromosome_70_30(self):
        np.random.seed(0)

        # run a few times for robust measure on stochastic method (would run more, but is highly expensive)
        m, n = 70, 30
        env = EnvironmentInitializer(robots=n, tasks=m)
        ga = GeneticAlgorithm(pop_init='greedy', env=env)
        for _ in range(5):
            actual = ga._GeneticAlgorithm__create_greedy_two_part_chromosome()

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track, actual)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual[j]

            self.assertTrue(total_sum == m)
