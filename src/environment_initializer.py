import numpy as np
import json

class EnvironmentInitializer():
    def __init__(self, 
        robots, tasks, 
        grid_path='maps/warehouse_small.txt', 
        task_path='maps/task_list.json'
    ):
        self.robots = robots        # number of agents
        self.tasks = tasks          # number of tasks

        self.grid = self.__access_grid(grid_path)
        self.task_list = self.__access_tasks(task_path)
        self.robot_loc = self.__access_robot_loc() # initial locations of agents

    '''
    Accesses the list of tasks from a json file, and stores them into a list 
        of size n, where n is the number of tasks given.
    Format -- tasks: [id, release_time, [x1, y1, x2, y2]]
        - id : the id number of the task
        - release_time: the time when the task will be made available
        - x1, y1: the coordinate of the source of the errand
        - x2, y2: the coordinate of the destination of the errand

    Params:
        None

    Returns:
        The tasks using the formatting: [(x1, y1), (x2, y2)] for each element.
    '''
    def __access_tasks(self, task_path):
        # read tasks json file
        with open(task_path) as file:
            t = json.load(file)

        # remove release time, reformat as:
        # [(x1, y1), x2, y2)]
        errands = []
        for task_list in t.values():
            for details in task_list:
                id_num, release_time, errand_coords = details
                errand_src = errand_coords[0], errand_coords[1]
                errand_dest = errand_coords[2], errand_coords[3]
                errands.append([errand_src, errand_dest])

        return errands[:self.tasks]

    '''
    Reads the map into a 2D grid
    Format -- '.' for a valid path, '@' for an obstacle.

    Params:
        None

    Returns:
        A 2D list containing the map
    '''
    def __access_grid(self, grid_path):
        # each character is a cell
        return np.genfromtxt(grid_path, dtype='str', delimiter=1)

    '''
    Reads the 2D grid, acquiring the first n robot locations, where n is the 
        defined number of robots.

    Params:
        None

    Returns:
        A list of the initial robot locations.
    '''
    def __access_robot_loc(self):
        init_locations = []
        rows, cols = len(self.grid), len(self.grid[0])
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] == 'E':
                    init_locations.append((r, c))

        return init_locations[:self.robots]

    '''
    Getters
    '''
    def num_of_tasks(self):
        return self.tasks

    def num_of_robots(self):
        return self.robots

    def get_robot_loc(self):
        return self.robot_loc

    def get_grid(self):
        return self.grid

    def get_task_list(self):
        return self.task_list
