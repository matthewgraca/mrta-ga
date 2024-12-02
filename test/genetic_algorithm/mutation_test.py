import unittest
from src.genetic_algorithm import GeneticAlgorithm
import numpy as np

class MutationTest(unittest.TestCase):
    def test_inverse_mutation_1(self):
        np.random.seed(0)
        ga = GeneticAlgorithm(mutation='inverse')

        # run 100 times for robust measure on stochastic method
        m, n = 10, 3
        for _ in range(100):
            chromo = ga._GeneticAlgorithm__create_two_part_chromosome(m, n)
            actual = ga._GeneticAlgorithm__mutation(chromo, m, n)

            # check first part is a permutation of tasks
            track = set()
            for i in range(m):
                val = actual[i]
                self.assertTrue(val < m+1 and val not in track)
                track.add(val)

            # check second part sums to tasks
            total_sum = 0
            for j in range(n+m-1, m-1, -1):
                total_sum += actual1[j]

            self.assertTrue(total_sum == m)
