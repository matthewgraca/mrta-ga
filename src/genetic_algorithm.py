import json
import numpy as np
from src.a_star import AStar
from itertools import islice

class GeneticAlgorithm:
    '''
    Parameters:
        pop_size: Population size, default 100
        pop_init: Population initialization method, default random
        selection: Selection method, default stochastic universal sampling
        crossover: Crossover method, default two part crossover (tcx)
        mutation: Mutation method, default inverse mutation
        pc: Probability of crossover, default 0.4
        pm: Probability of mutation, default 0.6
        replacement: Replacement method, default none
    '''
    def __init__(
            self, 
            pop_size=100, 
            pop_init='random',
            selection='sus',
            crossover='tcx', 
            mutation='inverse', 
            pc=0.4, 
            pm=0.6,
            replacement='none'
    ):
        # check parameters
        self.__validate_params(
            pop_size, pop_init, 
            selection, crossover, mutation, pc, pm, 
            replacement
        )

        # initialize
        self.pop_size = pop_size
        self.pop_init = pop_init
        self.selection = selection
        self.crossover = crossover 
        self.mutation = mutation
        self.pc = pc
        self.pm = pm
        self.replacement = replacement

    '''
    Validates the parameters. Throws errors if they are not valid
    '''
    def __validate_params(
        self, 
        pop_size, pop_init, 
        selection, crossover, mutation, pc, pm, 
        replacement
    ):
        # dictionary of valid parameters
        valid_params = {
            'pop_init'      : {'random'},
            'selection'     : {'sus'},
            'crossover'     : {'tcx'},
            'mutation'      : {'inverse'},
            'replacement'   : {'none'}
        }

        # limit pop size to [1, 100000]
        if not 1 <= pop_size <= 100000:
            raise ValueError(f"pop_size should be in the range [1, 100000]")

        # ensure valid methods
        if pop_init not in valid_params['pop_init']:
            method = "Population initialization"
            raise ValueError(f"{method} method \'{pop_init}\' is not defined")
        if selection not in valid_params['selection']:
            method = "Selection"
            raise ValueError(f"{method} method \'{selection}\' is not defined")
        if crossover not in valid_params['crossover']: 
            method = "Crossover"
            raise ValueError(f"{method} method \'{crossover}\' is not defined")
        if mutation not in valid_params['mutation']:
            method = "Mutation"
            raise ValueError(f"{method} method \'{mutation}\' is not defined")
        if replacement not in valid_params['replacement']:
            method = "Replacement"
            raise ValueError(f"{method} method \'{replacement}\' is not defined")

        # limit pc to [0.0, 1.0]
        if not 0.0 <= pc <= 1.0:
            raise ValueError("pc should be in the range [0.0, 1.0]")
        if not 0.0 <= pm <= 1.0:
            raise ValueError("pm should be in the range [0.0, 1.0]")

    '''
    Returns: The current parameter setup of the genetic algorithm as a 
        dictionary.
    '''
    def get_parameters(self):
        return {
            'pop_size'      : self.pop_size,
            'pop_init'      : self.pop_init,
            'selection'     : self.selection,
            'crossover'     : self.crossover,
            'mutation'      : self.mutation,
            'pc'            : self.pc,
            'pm'            : self.pm,
            'replacement'   : self.replacement
        }

    '''
    Accesses the list of tasks from a json file
    Format -- tasks: [id, release_time, [x1, y1, x2, y2]]
        - id : the id number of the task
        - release_time: the time when the task will be made available
        - x1, y1: the coordinate of the source of the errand
        - x2, y2: the coordinate of the destination of the errand

    Params:
        path: Path of the file containing the tasks

    Returns:
        The tasks using the formatting:
            id : [(x1, y1), (x2, y2)]
    '''
    def access_tasks(self, path='maps/task_list.json'):
        # read tasks json file
        with open(path) as file:
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
        path: Path of the file containing the map 

    Returns:
        A 2D list containing the map
    '''
    def access_map(self, path):
        # each character is a cell
        return np.genfromtxt(path, dtype='str', delimiter=1)

    '''
    Runs the genetic algorithm with the defined parameters
    '''
    def run(self):
        tasks, robots = 9, 3
        # initialization
        pop = self.__pop_init(self.pop_size, tasks, robots)
        '''
        # parent selection
        p1, p2 = self.parent_selection(method=self.selection, population=pop)
        '''
        # crossover
        c1, c2 = self.__crossover(p1, p2, tasks, robots)
        # mutation
        c1 = self.mutation(c1, tasks, robots)
        c2 = self.mutation(c2, tasks, robots)
        '''
        # replacement
        pop = self.replacement(method=self.replacement, pop=pop, c1=c1, c2=c2)
        # termination
        '''

    '''
    **Population initialization functions**
    '''

    '''
    Wrapper function for initializing population

    Params:
        pop_size: The size of the population
        tasks: The number of tasks to finish
        robots: The number of robots

    Returns:
        The population of candidate solutions, as a result of the given method
    '''
    def __pop_init(self, pop_size, tasks, robots):
        method = self.pop_init
        # list of current population initialization methods
        pop_init_methods = {
            'random' : self.__random_pop_init
        }
        return pop_init_methods[method](pop_size, tasks, robots)

    '''
    Implements randomized population initialization.

    Params:
        pop_size: The size of the population
        tasks: The number of tasks
        robots: The number of robots

    Returns:
        The population, randomly generated based on the given tasks and robots.
    '''
    def __random_pop_init(self, pop_size, tasks, robots):
        pop = []
        for i in range(pop_size):
            # uses two-part chromosome representation
            chromosome = self.__create_two_part_chromosome(tasks, robots)
            pop.append(chromosome)

        return pop 

    '''
    Implements the candidate solutions. Design is a two-part chromosome, where 
        the first part contains the tours of each robot, while the second 
        part contains the size of the tour taken by each robot.

    Params:
        tasks: The number of tasks to complete
        robots: The number of robots in the fleet

    Returns:
        The chromosome encoding of the candidate solution.
    '''
    def __create_two_part_chromosome(self, tasks, robots):
        # part 1: random permutation of tasks
        chromo1 = np.random.permutation(tasks) + 1
        chromo2 = [0] * robots
        
        # part 2: random amount of tasks
        tasks_left, robots_left = tasks, robots
        for i in range(robots-1):
            tasks_assigned = np.random.randint(1, tasks_left-robots_left+2)
            tasks_left = tasks_left - tasks_assigned 
            robots_left -= 1
            chromo2[i] = tasks_assigned 
        chromo2[robots-1] = tasks_left 

        chromosome = np.concatenate((chromo1, chromo2)) 

        return chromosome

    '''
    **Crossover functions**
    '''

    '''Wrapper function for crossover

    Params:
        p1: The first parent
        p2: The second parent
        tasks: The number of tasks
        robots: The number of robots

    Returns:
        A pair of offspring, as a result of whatever crossover method was given
    '''
    def __crossover(self, p1, p2, tasks, robots):
        method = self.crossover
        # list of current crossover methods
        xover_methods= {
            'tcx' : self.__two_part_crossover
        }
        return xover_methods[method](p1, p2, tasks, robots)


    '''
    Helper function that serves as the implementation for TCX. Produces one 
        child at a time.

    Params:
        p1: The first parent
        p2: The second parent
        tasks: The number of tasks
        robots: The number of robots 

    Returns:
        A child, based on p1, using TCX
    '''
    def __tcx_create_child(self, p1, p2, tasks, robots):
        # save start index of each tour
        segment_idx = self.__get_subtour_start_indices_of(p1, tasks, robots)

        # select gene segment for each agent
        saved_genes, saved_tour_sizes = self.__tcx_select_gene_segments(
            p1, tasks, robots, segment_idx
        )

        # find and sort gene positions of genes not in the segment wrt p2
        unsaved_genes, unsaved_tour_sizes = self.__tcx_find_and_sort_unsaved_genes(
            p1, p2, saved_genes, tasks, robots
        )

        # combine genes to create child
        c1 = self.__tcx_combine_genes(
            saved_genes, saved_tour_sizes, unsaved_genes, unsaved_tour_sizes
        )

        return c1

    '''
    Performs TCX to create two children.

    Params:
        p1: The first parent
        p2: The second parent
        tasks: The number of tasks
        robots: The number of robots

    Returns:
        A pair of offspring, as a result of TCX
    '''
    def __two_part_crossover(self, p1, p2, tasks, robots):
        c1 = self.__tcx_create_child(p1, p2, tasks, robots)
        c2 = self.__tcx_create_child(p2, p1, tasks, robots)
        return np.array(c1), np.array(c2)

    '''
    Helper function that finds the start index of every robot's subtour

    Params:
        chromosome: The chromosome containing the candidate solution
        tasks: The number of tasks
        robots: The number of robots

    Returns:
        A list of the start index of each robot's subtour
    '''
    def __get_subtour_start_indices_of(self, chromosome, tasks, robots):
        segment_idx = [0]
        for i in range(robots - 1):
            assigned_tasks = chromosome[tasks + i]
            segment_idx.append(segment_idx[i] + assigned_tasks)
        return segment_idx

    '''
    Helper function that selects the segments of each subtour for each robot

    Params:
        chromosome: The chromosome containing the candidate solution
        tasks: The number of tasks
        robots: The number of robots
        segment_idx: The start indices of each robot's subtour

    Returns:
        A list containing the saved genes and a list containing their tour size
    '''
    def __tcx_select_gene_segments(self, chromosome, tasks, robots, segment_idx):
        saved_genes = []
        saved_tour_sizes = []
        for m in range(robots):
            # get each segment range
            assigned_tasks = chromosome[tasks + m]
            start = segment_idx[m]
            end = start + assigned_tasks

            # pick subtour w/in the segment range
            lo = np.random.randint(start, end)
            hi = np.random.randint(lo, end)
            saved_tour_sizes.append(hi + 1 - lo)

            # save
            for i in range(lo, hi+1):
                saved_genes.append(chromosome[i])

        return saved_genes, saved_tour_sizes

    '''
    Helper function that finds the unsaved genes of p1 in p2, and sorts them 
        w.r.t. their positions in p2.

    Params:
        p1: The first parent
        p2: The second parent
        saved_genes: The gene segments that have been saved from p1
        tasks: The number of tasks
        robots: The number of robots

    Returns:
        A list containing the sorted, unsaved genes wrt p2, and a list 
            containing their tour size
    '''
    def __tcx_find_and_sort_unsaved_genes(self, p1, p2, saved_genes, tasks, robots):
        unsaved_genes = []
        # collect all unsaved genes
        for i in range(tasks):
            if p1[i] not in saved_genes:
                unsaved_genes.append(p1[i])

        sorted_unsaved_genes = []
        temp = []
        # find where the unsaved genes are in p2
        for gene in unsaved_genes:
            temp.append((np.where(p2 == gene)[0][0], gene))
        # sort unsaved genes base on their location in p2
        for i, gene in sorted(temp):
            sorted_unsaved_genes.append(gene)
        unsaved_genes = sorted_unsaved_genes

        # determine the unsaved genes that will attach to the saved genes 
        unsaved_tour_sizes = []
        genes_remaining = len(sorted_unsaved_genes)
        for m in range(robots):
            # if the end is reached, add the rest
            num_genes = 0
            if m == robots - 1 or genes_remaining == 0:
                num_genes = genes_remaining
            else:
                num_genes = np.random.randint(1, genes_remaining + 1)
            genes_remaining -= num_genes
            unsaved_tour_sizes.append(num_genes)

        return unsaved_genes, unsaved_tour_sizes

    '''
    Helper function that combines the selected genes from p1 with the 
        unselected genes of p1 w.r.t. p2 to create the child. 

    Params:
        saved_genes: The saved genes of the first parent
        saved_tour_sizes: The size of each tour of each robot
        unsaved_genes: The unsaved genes of the first parent w.r.t. the second
        unsaved_tour_sizes: The size of each tour of each robot

    Returns:
        The child of the first and second parent.
    '''
    def __tcx_combine_genes(self, saved_genes, saved_tour_sizes, unsaved_genes, unsaved_tour_sizes): 
        c1_tasks, c1_robots = [], []
        p1_start, p1_end = 0, 0
        p2_start, p2_end = 0, 0
        for p1_tour, p2_tour in zip(saved_tour_sizes, unsaved_tour_sizes):
            temp = []
            # parent 1 segment
            p1_start = p1_end
            p1_end = p1_start + p1_tour 
            segment = saved_genes[p1_start:p1_end]
            temp.extend(segment)

            # parent 2 segment
            p2_start = p2_end
            p2_end = p2_start + p2_tour
            segment = unsaved_genes[p2_start:p2_end]
            temp.extend(segment)

            # combine segments
            c1_tasks.extend(temp)
            c1_robots.append(p1_tour + p2_tour)

        return c1_tasks + c1_robots

    '''
    **Mutation functions**
    '''

    '''
    Wrapper function for mutation 

    Params:
        chromosome: The chromosome that will be mutated
        tasks: The number of tasks to finish
        robots: The number of robots

    Returns:
        The mutated chromosome
    '''
    def __mutation(self, chromosome, tasks, robots):
        method = self.mutation
        # list of current crossover methods
        mut_methods= {
            'inverse': self.__inverse_mutation
        }
        return mut_methods[method](chromosome, tasks, robots)

    '''
    Performs inverse mutation on the given chromosome. A random subtour is 
        picked, then inverted.

    Params:
        chromosome: The chromosome that will be mutated
        tasks: The number of tasks to finish
        robots: The number of robots

    Returns:
        The mutated chromosome
    '''
    def __inverse_mutation(self, chromosome, tasks, robots):
        # pick a random agent's subtour
        agent = np.random.randint(0, robots)
        startIdxes = self.__get_subtour_start_indices_of(chromosome, tasks, robots)

        # get their indices
        startIdx = startIdxes[agent]
        endIdx = startIdx + chromosome[tasks + agent]

        # invert the subtour of that agent
        inverted_subtour = chromosome[startIdx : endIdx][::-1]
        chromosome[startIdx : endIdx] = inverted_subtour
        return chromosome 

    '''
    **Fitness functions**
    '''

    '''
    Determines the fitness of the individual. Currently, our metric is 
        sum of distances between robot and task, with the goal of minimization.

    Params
        grid: The map that will be used for the robots to traverse
        chromosome: The candidate solution
        tasks: The number of tasks
        robots: The number of robots
        robot_loc: The initial robot locations, as pairs

    Returns
        The fitness, which is simply the sum of the distances between the 
            robots and their tasks.
    '''
    def __fitness(self, grid, chromosome, tasks, robots, robot_loc):
        '''
        # TODO wrapper, move the below code down to fitness_a_star
        fitness_methods = {
            'makespan'      : self.__makespan,
            'longest_path'  : self.__longest_path
        }
        '''

        # TODO remove hardcoded grid, tasks, robots, and chromosome
        grid = self.access_map('maps/warehouse_small.txt')
        task_list = self.access_tasks('maps/task_list.json')
        tasks, robots = 12, 3
        chromosome = [12, 3, 4, 10, 11, 8, 5, 9, 2, 6, 1, 7, 5, 3, 4]

        # TODO remove hardcoded task list?
        cut_task_list = [task_list[t] for t in list(islice(task_list, tasks))]

        # TODO remove hardcode
        robot_loc = [(3, 8), (5, 3), (20, 20)]

        # get the length of each robot's subtour
        astar = AStar(grid)
        subtours = []
        path_len = 0
        segment_idx = self.__get_subtour_start_indices_of(chromosome, tasks, robots)
        # go through each robot's subtour
        for m in range(robots):
            # subtour start-end indices
            start = segment_idx[m]
            end = start + chromosome[tasks + m]

            # calculate subtour path taken by this robot
            path = []
            for i in range(start, end):
                errand_idx = chromosome[i] - 1

                # path from robot to errand source
                src = robot_loc[m]
                dest = cut_task_list[errand_idx][0]
                path.extend(astar.a_star_search(src, dest))

                # path from errand source to errand destination
                src = dest
                dest = cut_task_list[errand_idx][1]
                path.extend(astar.a_star_search(src, dest))

                # update robot's new location
                robot_loc[m] = dest

            # append subtour to subtours list, and total path length of subtours
            subtours.append(path)
            path_len += len(path)

        return path_len 

    '''
    Helper function for fitness. Calculates the distance of the subtour,
        and the robot using A*

    Params
        grid: The map that will be used for the robots to traverse
        subtour: The subtour whose distances will be evaluated
        robot_loc: The initial robot locations

    Returns
        The distance between all of the tasks in the subtour.
    '''
    def __fitness_A_star(self, subtour, robot_loc):
        return 0
