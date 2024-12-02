import json
import numpy as np
from collections import deque

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

    Params:
        path: path of the file
    '''
    def access_tasks(path='maps/task_list.json'):
        with open(path, 'r') as file:
            tasks = json.load(file)

    '''
    Runs the genetic algorithm with the defined parameters
    '''
    def run(self):
        tasks, robots = 9, 3
        # initialization
        pop = self.__pop_init(self.pop_init, self.pop_size, tasks, robots)
        '''
        # parent selection
        p1, p2 = self.parent_selection(method=self.selection, population=pop)
        '''
        # crossover
        c1, c2 = self.__crossover(p1, p2, tasks, robots)
        '''
        # mutation
        c1, c2 = self.mutation(method=self.mutation, c1=c1, c2=c2)
        # replacement
        pop = self.replacement(method=self.replacement, pop=pop, c1=c1, c2=c2)
        # termination
        '''

    '''
    Wrapper function for initializing population

    Params:
        method: The method being used to initialize
        pop_size: The size of the population
        tasks: The number of tasks to finish
        robots: The number of robots

    Returns:
        The population of candidate solutions, as a result of the given method
    '''
    def __pop_init(self, method, pop_size, tasks, robots):
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
        segment_idx = [0]
        for i in range(robots - 1):
            assigned_tasks = p1[tasks + i]
            segment_idx.append(segment_idx[i] + assigned_tasks)

        # select gene segment for each agent
        saved_genes = []
        saved_tour_sizes = []
        for m in range(robots):
            # get each segment range
            assigned_tasks = p1[tasks + m]
            start = segment_idx[m]
            end = start + assigned_tasks

            # pick subtour w/in the segment range
            lo = np.random.randint(start, end)
            hi = np.random.randint(lo, end)
            saved_tour_sizes.append(hi + 1 - lo)

            # save
            for i in range(lo, hi+1):
                saved_genes.append(p1[i])

        # find and sort gene positions of genes not in the segment wrt p2
        unsaved_genes = []
        for i in range(tasks):
            if p1[i] not in saved_genes:
                unsaved_genes.append(p1[i])
        sorted_unsaved_genes = []
        temp = []
        for gene in unsaved_genes:
            temp.append((np.where(p2 == gene)[0][0], gene))
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

        # combine genes to create child
        c1_tasks, c2_tasks = [], []
        c1_robots, c2_robots = [], []
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

        c1 = c1_tasks + c1_robots
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
