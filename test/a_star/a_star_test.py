import unittest
import numpy as np
from src.a_star import * 

class AStarTest(unittest.TestCase):
    '''
    Test initialization
    '''
    def test_node_init(self):
        node = Node()
        self.assertTrue(True)

    def test_a_star_init(self):
        grid = np.zeros((40, 60)).tolist()
        a_star = AStar(grid)
        self.assertTrue(True)

    '''
    Test helper functions
    '''
    def test_is_valid(self):
        grid = np.zeros((40, 60)).tolist()
        a_star = AStar(grid)

        # open
        self.assertTrue(a_star.is_valid(3, 4))

        # open
        self.assertTrue(a_star.is_valid(0, 0))

        # row out of bound
        self.assertFalse(a_star.is_valid(-1, 4))

        # col out of bound
        self.assertFalse(a_star.is_valid(1, -1))

        # row and col out of bound
        self.assertFalse(a_star.is_valid(-1, -1))
        self.assertFalse(a_star.is_valid(40, 60))

    def test_search_1(self):
        grid = [
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '@', '.', '@', '.'],
            ['@', '@', '.', '@', '.', '@', '@', '@', '@', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '@'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '@', '@'],
            ['.', '@', '@', '@', '@', '.', '@', '@', '@', '.'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '@', '@', '.', '@', '@', '.']
        ]

        src = [8, 0]
        dest = [0, 0]

        a_star = AStar(grid)
        actual = a_star.a_star_search(grid, src, dest)
        expected = [
            (8, 0), (7, 0), (6, 0), (5, 0), (4, 0), (4, 1), 
            (4, 2), (3, 2), (2, 2), (1, 2), (1, 1), (1, 0), (0, 0)
        ]

        self.assertEqual(actual, expected)


    def test_invalid_src_blocked(self):
        grid = [
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '@', '.', '@', '.'],
            ['@', '@', '.', '@', '.', '@', '@', '@', '@', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '@'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '@', '@'],
            ['.', '@', '@', '@', '@', '.', '@', '@', '@', '.'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '@', '@', '.', '@', '@', '.']
        ]

        a = AStar(grid)
        src = [0, 1]
        dest = [0, 0]

        actual = a.a_star_search(grid, src, dest)
        expected = []

        self.assertEqual(actual, expected)

    def test_invalid_dest_blocked(self):
        grid = [
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '@', '.', '@', '.'],
            ['@', '@', '.', '@', '.', '@', '@', '@', '@', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '@'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '@', '@'],
            ['.', '@', '@', '@', '@', '.', '@', '@', '@', '.'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '@', '@', '.', '@', '@', '.']
        ]

        a = AStar(grid)
        src = [0, 0]
        dest = [0, 1]

        actual = a.a_star_search(grid, src, dest)
        expected = []

        self.assertEqual(actual, expected)       

    def test_invalid_path_src_is_dest(self):
        grid = [
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '.'],
            ['.', '.', '.', '@', '.', '.', '@', '.', '@', '.'],
            ['@', '@', '.', '@', '.', '@', '@', '@', '@', '.'],
            ['.', '.', '.', '@', '.', '.', '.', '@', '.', '@'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '@', '@'],
            ['.', '@', '@', '@', '@', '.', '@', '@', '@', '.'],
            ['.', '@', '.', '.', '.', '.', '@', '.', '.', '.'],
            ['.', '.', '.', '@', '@', '@', '.', '@', '@', '.']
        ]

        a = AStar(grid)
        src = [0, 0]
        dest = [0, 0]

        actual = a.a_star_search(grid, src, dest)
        expected = []

        self.assertEqual(actual, expected)
