import unittest
from src.genetic_algorithm import GeneticAlgorithm
import numpy as np

class CrossoverTest(unittest.TestCase):
    def test_xover1(self):
        np.random.seed(0)
        ga = GeneticAlgorithm()

        # run 100 times for robust measure on stochastic method
        m, n = 10, 3
        for _ in range(100):
            p1 = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            p2 = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            actual1, actual2 = ga._GeneticAlgorithm__two_part_crossover(
                p1, p2, m, n 
            )

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual1[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual1[j]

            self.assertTrue(total_sum == m)

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual2[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual2[j]

            self.assertTrue(total_sum == m)
