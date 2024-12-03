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

        self.assertTrue(a_star.is_valid(3, 4))
        self.assertTrue(a_star.is_valid(0, 0))
        self.assertFalse(a_star.is_valid(-1, 4))
        self.assertFalse(a_star.is_valid(1, -1))
        self.assertFalse(a_star.is_valid(-1, -1))
        self.assertFalse(a_star.is_valid(40, 60))

    def test_search_1(self):
        # Define the grid ('.' for unblocked, '@' for blocked)
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

        # Define the source and destination
        src = [8, 0]
        dest = [0, 0]

        # Run the A* search algorithm
        a_star = AStar(grid)
        a_star.a_star_search(grid, src, dest)
