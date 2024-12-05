import numpy as np
import json

class EnvironmentInitializer():
    def __init__(self, 
        robots, tasks, robot_loc=None, 
        grid_path='maps/warehouse_small.txt', 
        task_path='maps/task_list.json'
    ):
        self.robots = robots        # number of agents
        self.tasks = tasks          # number of tasks
        self.robot_loc = robot_loc  # initial locations of the agents

        self.grid = self.__access_grid(grid_path)
        self.task_list = self.__access_tasks(task_path)

    '''
    Accesses the list of tasks from a json file
    Format -- tasks: [id, release_time, [x1, y1, x2, y2]]
        - id : the id number of the task
        - release_time: the time when the task will be made available
        - x1, y1: the coordinate of the source of the errand
        - x2, y2: the coordinate of the destination of the errand

    Params:
        None

    Returns:
        The tasks using the formatting:
            id : [(x1, y1), (x2, y2)], where id counts from 0
    '''
    def __access_tasks(self, task_path):
        # read tasks json file
        with open(task_path) as file:
            t = json.load(file)

        # remove release time, reformat as:
        # id : [(x1, y1), (x2, y2)]
        errands = {}
        for task_list in t.values():
            for details in task_list:
                id_num, release_time, errand_coords = details
                errand_src = errand_coords[0], errand_coords[1]
                errand_dest = errand_coords[2], errand_coords[3]
                errands[id_num] = [errand_src, errand_dest]

        return errands

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
