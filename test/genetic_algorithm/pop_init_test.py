import numpy as np
import unittest
from src.genetic_algorithm import GeneticAlgorithm

class PopInitTest(unittest.TestCase):
    def test_two_part_chromosome_10_3(self):
        np.random.seed(0)
        ga = GeneticAlgorithm()

        # run 100 times for robust measure on stochastic method
        m, n = 10, 3
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            actual.tolist()

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
        ga = GeneticAlgorithm()

        # run 100 times for robust measure on stochastic method
        m, n = 100, 13
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            actual.tolist()

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
        ga = GeneticAlgorithm()

        # run 100 times for robust measure on stochastic method
        m, n = 1000, 50
        for _ in range(100):
            actual = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            actual.tolist()

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
