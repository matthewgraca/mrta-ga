import unittest
from src.environment_initializer import EnvironmentInitializer
from src.genetic_algorithm import GeneticAlgorithm
import numpy as np

class MutationTest(unittest.TestCase):
    def test_inverse_mutation_1(self):
        m, n = 10, 3
        env = EnvironmentInitializer(robots=n, tasks=m)
        np.random.seed(0)
        ga = GeneticAlgorithm(mutation='inverse', env=env)

        # run 100 times for robust measure on stochastic method
        for _ in range(100):
            chromo = ga._GeneticAlgorithm__create_random_two_part_chromosome()
            actual = ga._GeneticAlgorithm__mutation(chromo)

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

            # check if a subtour was actually inverted
            inverted = False 
            segments = ga._GeneticAlgorithm__get_subtour_start_indices_of(chromo)
            for i in range(n):
                # get subtour slice
                subtour_reversed = chromo[segments[i]:chromo[m+i]][::-1]
                subtour_actual = actual[segments[i]:chromo[m+i]]
                if np.array_equal(subtour_reversed, subtour_actual):
                    inverted = True
            self.assertTrue(inverted)

    def test_swap_mutation_1(self):
        m, n = 10, 3
        env = EnvironmentInitializer(robots=n, tasks=m)
        np.random.seed(0)
        ga = GeneticAlgorithm(mutation='swap', env=env)

        # run 100 times for robust measure on stochastic method
        for _ in range(100):
            chromo = ga._GeneticAlgorithm__create_random_two_part_chromosome()
            actual = ga._GeneticAlgorithm__mutation(chromo)

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

            # check if genes have actually been swapped; i.e. each portion
            #   differs in exactly 2 places
            # check part 1
            actual_part_1 = actual[:m]
            chromo_part_1 = chromo[:m]
            diff = 0
            for i in range(m):
                if actual_part_1[i] != chromo_part_1[i]:
                    diff += 1
            self.assertEqual(diff, 2)

            # check part 2
            diff = 0
            same = set()
            chromo_part_2 = chromo[m:m+n]
            actual_part_2 = actual[m:m+n]
            for i in range(n):
                # exception: same values (eg. [8, 1, 1]) can't be detected
                if chromo_part_2[i] not in same:
                    same.add(chromo_part_2[i])
                if actual_part_2[i] != chromo_part_2[i]:
                    diff += 1

            if len(same) != n:
                self.assertTrue(True)
            else:
                self.assertEqual(diff, 2)

