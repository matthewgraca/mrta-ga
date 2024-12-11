import numpy as np
from src.a_star import AStar
from src.environment_initializer import EnvironmentInitializer
import pprint

class GeneticAlgorithm:
    '''
    Parameters:
        objective_func: Objective function that will be used for calculating
            fitness, default flow_time 
        pop_size: Population size, default 100
        pop_init: Population initialization method, default random
        selection: Selection method, default roulette wheel selection (rws) 
        crossover: Crossover method, default two part crossover (tcx)
        mutation: Mutation method, default inverse mutation
        pc: Probability of crossover, default 0.4
        pm: Probability of mutation, default 0.6
        replacement: Replacement method, default replace_worst 
    '''
    def __init__(
            self, 
            objective_func='flow_time',
            pop_size=100, 
            pop_init='random',
            selection='rws',
            crossover='tcx', 
            mutation='inverse', 
            pc=0.4, 
            pm=0.6,
            replacement='replace_worst',
            env=None
    ):
        # dictionary of valid parameters
        self.valid_params = {
            'objective_func': {'makespan', 'flow_time'},
            'pop_init'      : {'random', 'greedy'},
            'selection'     : {'rws'},
            'crossover'     : {'tcx'},
            'mutation'      : {'inverse', 'swap'},
            'replacement'   : {'replace_worst', 'elitism'}
        }   

        # check parameters
        self.__validate_params(
            objective_func,
            pop_size, pop_init, 
            selection, crossover, mutation, pc, pm, 
            replacement
        )

        # initialize
        self.objective_func = objective_func
        self.pop_size = pop_size
        self.pop_init = pop_init
        self.selection = selection
        self.crossover = crossover 
        self.mutation = mutation
        self.pc = pc
        self.pm = pm
        self.replacement = replacement

        # initialize environment
        self.env = env

    '''
    Validates the parameters. Throws errors if they are not valid
    '''
    def __validate_params(
        self, 
        objective_func,
        pop_size, pop_init, 
        selection, crossover, mutation, pc, pm, 
        replacement
    ):
        # limit pop size to [1, 100000]
        if not 10 <= pop_size <= 100000:
            raise ValueError(f"pop_size should be in the range [10, 100000]")

        # ensure valid methods
        if objective_func not in self.valid_params['objective_func']:
            method = "Objective function"
            err_msg = f"{method} method \'{objective_func}\' is not defined"
            raise ValueError(err_msg)
        if pop_init not in self.valid_params['pop_init']:
            method = "Population initialization"
            raise ValueError(f"{method} method \'{pop_init}\' is not defined")
        if selection not in self.valid_params['selection']:
            method = "Selection"
            raise ValueError(f"{method} method \'{selection}\' is not defined")
        if crossover not in self.valid_params['crossover']: 
            method = "Crossover"
            raise ValueError(f"{method} method \'{crossover}\' is not defined")
        if mutation not in self.valid_params['mutation']:
            method = "Mutation"
            raise ValueError(f"{method} method \'{mutation}\' is not defined")
        if replacement not in self.valid_params['replacement']:
            method = "Replacement"
            err_msg = f"{method} method \'{replacement}\' is not defined"
            raise ValueError(err_msg)

        # limit pc to [0.0, 1.0]
        if not 0.0 <= pc <= 1.0:
            raise ValueError("pc should be in the range [0.0, 1.0]")
        if not 0.0 <= pm <= 1.0:
            raise ValueError("pm should be in the range [0.0, 1.0]")

    '''
    Returns: The current parameter setup of the genetic algorithm as a 
        dictionary.
    '''
    def get_params(self):
        return {
            'objective_func': self.objective_func,
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
    Returns: All of the possible valid parameters, as a dictionary
    '''
    def get_valid_params(self):
        return self.valid_params

    '''
    Runs the genetic algorithm with the defined parameters

    Params:
        generations: The number of generations to run the experiment
        update_step: The number of generations before each update
    '''
    def run(self, generations=100, update_step=1):
        # initialization
        pop = self.__pop_init(self.pop_size)
        pop_fits = self.__fitness_of_pop(pop, constraint=True)

        # termination condition
        for gen in range(generations):
            # select a mating pool from the population
            mating_pool = self.__selection(pop, pop_fits)
            while mating_pool:
                # parent selection
                p1, p2 = mating_pool.pop(), mating_pool.pop()

                # crossover
                if np.random.rand() < self.pc:
                    c1, c2 = self.__crossover(p1, p2)
                else:
                    c1, c2 = p1, p2

                # mutation
                c1 = self.__mutation(c1) if np.random.rand() < self.pm else c1
                c2 = self.__mutation(c2) if np.random.rand() < self.pm else c2

                # fitness calcs
                c1_fit = self.__fitness(c1, constraint=True)
                c2_fit = self.__fitness(c2, constraint=True)

                # add children to the population
                pop = np.append(pop, [c1, c2], axis=0)
                pop_fits = np.append(pop_fits, [c1_fit, c2_fit])

            # replacement
            pop_fits, pop = self.__replacement(pop, pop_fits)

            # results
            best, worst = pop[-1], pop[0]
            best_fit, worst_fit = self.__fitness(best), self.__fitness(worst)
            if (gen + 1) % update_step == 0:
                print(f'Generation: {gen}, Best solution:  {best}, Best fitness:  {best_fit}')
                print(f'Generation: {gen}, Worst solution: {worst}, Worst fitness: {worst_fit}')
                print(f'Average fitness: {np.average(pop_fits)}')

        print("Best path of the robots --")
        print(best)
        best_path = self.__fitness_get_all_subtours(best)
        for i in range(len(best_path)):
            pprint.pprint(f'Path of robot {i+1}: {best_path[i]}')

        return

    '''
    **Population initialization functions**
    '''

    '''
    Wrapper function for initializing population

    Params:
        pop_size: The size of the population

    Returns:
        The population of candidate solutions, as a result of the given method
    '''
    def __pop_init(self, pop_size):
        method = self.pop_init
        # list of current population initialization methods
        pop_init_methods = {
            'random' : self.__random_pop_init,
            'greedy' : self.__greedy_pop_init
        }
        return pop_init_methods[method](pop_size)

    '''
    Implements randomized population initialization.

    Params:
        pop_size: The size of the population

    Returns:
        The population, randomly generated based on the given tasks and robots.
    '''
    def __random_pop_init(self, pop_size):
        return [self.__create_random_two_part_chromosome() for _ in range(pop_size)]

    '''
    Implements greedy population intialization, where all tasks are assigned
        to the closest robot.

    Params:
        pop_size: The size of the population

    Returns:
        The population, greedily seeded 
    '''
    def __greedy_pop_init(self, pop_size):
        return [self.__create_greedy_two_part_chromosome() for _ in range(pop_size)]

    '''
    Implements the candidate solutions. Design is a two-part chromosome, where 
        the first part contains the tours of each robot, while the second 
        part contains the size of the tour taken by each robot.
        
        The tasks assigned to each robot is greedily determined. Each robot 
            is guaranteed at least one task, and the rest of the tasks are 
            assigned to the closest robot. Then, the subtour of each robot 
            is shuffled.

    Returns:
        The chromosome encoding of the candidate solution.
    '''
    def __create_greedy_two_part_chromosome(self):
        robots = self.env.num_of_robots()
        robot_assignments = [[] for _ in range(robots)]
        task_list = self.env.get_task_list()
        robot_loc = self.env.get_robot_loc()
        search = AStar(self.env.get_grid())

        # guaranteed every robot has at least 1 task
        for i in range(robots):
            robot_assignments[i].append(i + 1)

        # assign each task to the closest robot
        for i in range(robots, len(task_list)):
            src, _ = task_list[i] 
            min_dist = float('inf')
            robot = -1
            # for each task, find the closest robot
            for j in range(len(robot_loc)):
                dest = robot_loc[j]
                path = search.a_star_search(src, dest)
                if len(path) < min_dist:
                    min_dist = len(path) 
                    robot = j # closest robot to task
            # assign task to closest robot
            robot_assignments[robot].append(i + 1)
        
        # shuffle tasks of each robot
        assgn_len = [0] * robots
        for i in range(robots):
            np.random.shuffle(robot_assignments[i])
            assgn_len[i] = len(robot_assignments[i])

        # create the chromosome
        tasks = np.concatenate(robot_assignments)
        chromosome = np.concatenate([tasks, assgn_len], axis=0)

        return chromosome

    '''
    Implements the candidate solutions. Design is a two-part chromosome, where 
        the first part contains the tours of each robot, while the second 
        part contains the size of the tour taken by each robot.

        The perumutation of tasks and the tasks assigned to the robots are 
            randomly determined.

    Returns:
        The chromosome encoding of the candidate solution.
    '''
    def __create_random_two_part_chromosome(self):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
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
    **Selection functions**
    '''

    '''Wrapper function for selection 
    For the experiment, the number of children is predetermined.

    Params:
        pop: The population that is being selected from
        pop_fit: The population fitnesses

    Returns:
        A list containing the mating pool.
    '''
    def __selection(self, pop, pop_fit):
        method = self.selection
        # list of current selection methods
        selection_methods= {
            'rws' : self.__roulette_wheel_selection
        }
        mating_pool_size = self.__get_lambda(len(pop))
        return selection_methods[method](pop, pop_fit, mating_pool_size)

    '''
    Helper function for population selection and replacement that determines 
        how many individuals need to be selected or replaced from the 
        population. Currently, the number is guaranteed to be at least 2.

    Params:
        pop_size: The size of the population
        mating_pool_prop: The proportion of the population that will be 
            selected for the mating pool or selected for replacement. Default 
            set to 0.2 (20%).

    Returns:
        The number of individuals that will be selected for selection or 
            replacement.
    '''
    def __get_lambda(self, pop_size, mating_pool_prop=0.2):
        a = int(pop_size * mating_pool_prop)
        b = a if a % 2 == 0 else a - 1  # ensure even to pair all parents
        c = b if b >= 2 else 2          # ensure there are at least 2 parents
        return c

    '''
    Performs roulette wheel selection, or fitness proportionate selection.

    Params:
        pop: The population being picked from
        pop_fit: The fitnesses of the population
        mating_pool_size: Lambda, the size of the mating pool

    Returns:
        The parents from the mating pool.
    '''
    def __roulette_wheel_selection(self, pop, pop_fit, mating_pool_size):
        # sort by fitness
        pop_fitness, pop = self.__sort_pop_by_fitness(pop_fit, pop)

        # calculate cumulative probability distribution
        cpd = np.cumsum(pop_fitness)
        cpd = cpd / cpd[-1]

        # spin wheel lambda times to get that many members for the mating pool
        lam = mating_pool_size
        curr_member = 0
        mating_pool = [0] * lam
        while curr_member < lam:
            r = np.random.rand()
            i = 0
            while cpd[i] < r:
                i += 1
            mating_pool[curr_member] = pop[i]
            curr_member += 1
        return mating_pool 

    '''
    Helper function that sorts a list by the other. Used when the fitnesses 
        have already been calculated

    Params:
        fits: The first list being sorted, according to itself 
        pop: The second list, sorted according to fits 
    Returns:
        Two lists; fits, sorted by itself, and pop, sorted by fits 
    '''
    def __sort_pop_by_fitness(self, fits, pop):
        fits, pop = (np.array(t) for t in 
            zip(*sorted(zip(fits, pop), key=lambda x: x[0]))
        )
        
        return fits, pop

    '''
    **Replacement functions**
    '''

    '''Wrapper function for replacement 
    For the experiment, the number individuals replaced is predetermined.

    Params:
        pop: The population that will be culled. Should already contain 
            children from mating; (population = mu + lambda)
        pop_fit: The fitnesses of the population

    Returns:
        The population, with lambda individuals removed
    '''
    def __replacement(self, pop, pop_fit):
        method = self.replacement
        # list of current replacement methods
        replacement_methods= {
            'replace_worst' : self.__replace_worst,
            'elitism'       : self.__elitism
        }
        lmbda = self.__get_lambda(len(pop))
        return replacement_methods[method](pop, pop_fit, lmbda)

    '''
    Implements replacement that removes the lambda worst individuals in the 
        population.

    Params:
        pop: The population that will have individuals removed 
        pop_fit: The fitnesses of the population
        lmbda: The number of individuals that will be removed

    Returns:
        The population, with lambda individuals removed.
    '''
    def __replace_worst(self, pop, pop_fit, lmbda):
        # sort population by fitness
        pop_fitness, pop = self.__sort_pop_by_fitness(pop_fit, pop)

        # fitness sorted from less fit -> most fit, so drop front end 
        next_gen_fits, next_gen = pop_fitness[lmbda:], pop[lmbda:]

        return next_gen_fits, next_gen

    '''
    Implements elitist replacement, a combination of age-based replacement
        while keeping the best individual.
            - The lambda oldest individuals are put up for replacement
            - If the individual being replaced is the best individual, they 
                compete with the children for a spot in the next generation
            - If any child has better fitness, the old individual is replaced
            - If none of the children have better fitness, a child is randomly 
                replaced. The old individual returns to the front of the queue.

    Params:
        pop: The population that will have individuals removed.
        pop_fit: The fitness of the population
        lmbda: The number of individuals that will be removed

    Returns:
        The population, with lambda individuals removed
    '''
    def __elitism(self, pop, pop_fit, lmbda):
        # TODO pass in the best fitness? otherwise we need to recalculate it many times over.
        # would just need to calc best fitness in pop initiatlization, then keep updating it.
        next_gen_fits, next_gen = [], []
        return next_gen_fits, next_gen

    '''
    **Crossover functions**
    '''

    '''Wrapper function for crossover

    Params:
        p1: The first parent
        p2: The second parent

    Returns:
        A pair of offspring, as a result of whatever crossover method was given
    '''
    def __crossover(self, p1, p2):
        method = self.crossover
        # list of current crossover methods
        xover_methods= {
            'tcx' : self.__two_part_crossover
        }
        return xover_methods[method](p1, p2)

    '''
    Helper function that serves as the implementation for TCX. Produces one 
        child at a time.

    Params:
        p1: The first parent
        p2: The second parent

    Returns:
        A child, based on p1, using TCX
    '''
    def __tcx_create_child(self, p1, p2):
        tasks, robots = self.env.num_of_robots(), self.env.num_of_tasks()
        # save start index of each tour
        segment_idx = self.__get_subtour_start_indices_of(p1)

        # select gene segment for each agent
        saved_genes, saved_tour_sizes = self.__tcx_select_gene_segments(
            p1, segment_idx
        )

        # find and sort gene positions of genes not in the segment wrt p2
        unsaved_genes, unsaved_tour_sizes = self.__tcx_find_and_sort_unsaved_genes(
            p1, p2, saved_genes
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

    Returns:
        A pair of offspring, as a result of TCX
    '''
    def __two_part_crossover(self, p1, p2):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
        c1 = self.__tcx_create_child(p1, p2)
        c2 = self.__tcx_create_child(p2, p1)
        return np.array(c1), np.array(c2)

    '''
    Helper function that finds the start index of every robot's subtour

    Params:
        chromosome: The chromosome containing the candidate solution

    Returns:
        A list of the start index of each robot's subtour
    '''
    def __get_subtour_start_indices_of(self, chromosome):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
        segment_idx = [0]
        for i in range(robots - 1):
            assigned_tasks = chromosome[tasks + i]
            segment_idx.append(segment_idx[i] + assigned_tasks)
        return segment_idx

    '''
    Helper function that selects the segments of each subtour for each robot

    Params:
        chromosome: The chromosome containing the candidate solution
        segment_idx: The start indices of each robot's subtour

    Returns:
        A list containing the saved genes and a list containing their tour size
    '''
    def __tcx_select_gene_segments(self, chromosome, segment_idx):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
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

    Returns:
        A list containing the sorted, unsaved genes wrt p2, and a list 
            containing their tour size
    '''
    def __tcx_find_and_sort_unsaved_genes(self, p1, p2, saved_genes):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
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

    Returns:
        The mutated chromosome
    '''
    def __mutation(self, chromosome):
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()
        method = self.mutation
        # list of current crossover methods
        mut_methods= {
            'inverse'   : self.__inverse_mutation,
            'swap'      : self.__swap_mutation
        }
        return mut_methods[method](chromosome)

    '''
    Performs inverse mutation on the given chromosome. A random subtour is 
        picked, then inverted.

    Params:
        chromosome: The chromosome that will be mutated

    Returns:
        The mutated chromosome
    '''
    def __inverse_mutation(self, chromosome):
        chromo = chromosome.copy()
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()

        # pick a random agent's subtour
        agent = np.random.randint(0, robots)
        startIdxes = self.__get_subtour_start_indices_of(chromo)

        # get their indices
        startIdx = startIdxes[agent]
        endIdx = startIdx + chromo[tasks + agent]

        # invert the subtour of that agent
        inverted_subtour = chromo[startIdx : endIdx][::-1]
        chromo[startIdx : endIdx] = inverted_subtour

        return chromosome 

    '''
    Performs swap mutation on the given chromosome. Two genes will be 
        swapped in the first part and the second part, resulting in 
        four swapped genes in total.

    Params:
        chromosome: The chromosome that will be mutated

    Returns:
        The mutated chromosome
    '''
    def __swap_mutation(self, chromosome):
        chromo = chromosome.copy()
        tasks, robots = self.env.num_of_tasks(), self.env.num_of_robots()

        # pick two indices in the first part of the chromosome, swap contents
        i, j = np.random.choice(tasks, 2, replace=False)
        chromo[i], chromo[j] = chromo[j], chromo[i]

        # pick two indices in the second part of the chromosome, swap contents
        i, j = np.random.choice(robots, 2, replace=False)
        i, j = i + tasks, j + tasks
        chromo[i], chromo[j] = chromo[j], chromo[i]

        return chromo

    '''
    **Fitness functions**
    '''

    '''
    Determines the fitness of the individual. 
    Contains the constraint of collisions, which will reduce the fitness 
        of the individual to 0.

    Params:
        chromosome: The candidate solution
        constraint: The option for the fitness function to check the
            collision constraint

    Returns:
        The fitness. Larger is better.
    '''
    def __fitness(self, chromosome, constraint=False):
        method = self.objective_func
        fitness_methods = {
            'makespan'      : self.__makespan,
            'flow_time'     : self.__flow_time
        }
        return 1 / fitness_methods[method](chromosome, constraint) 

    '''
    Determines the fitness of a given population. 

    Params:
        pop: The list of candidate solutions
        constraint: The option for the fitness function to check the
            collision constraint
    Returns:
        A list of the population's fitnesses. Larger is better.
    '''
    def __fitness_of_pop(self, pop, constraint=False):
        pop_fit = [0] * len(pop)
        for i in range(len(pop)):
            pop_fit[i] = self.__fitness(pop[i], constraint)

        return pop_fit

    '''
    Helper function for fitness. An objective function that measures the 
        average length of all the agents' paths.

    Params:
        chromosome: The candidate solution
        check_collisons: The option that checks for collisions in the tours

    Returns
        The average length of all the agents' paths.
    '''
    def __flow_time(self, chromosome, check_collisions=False):
        # get the length of each robot's subtour
        path_len = 0
        subtours = self.__fitness_get_all_subtours(chromosome)
        for subtour in subtours:
            path_len += len(subtour)

        # get collisions
        c = self.__constraint_collision(subtours) if check_collisions else 0
        return float('inf') if c > 0 else path_len / len(subtours)

    '''
    Helper fitness function that calculates the path of the subtour.

    Params:
        segment: The segment of the candidate solution containing the subtour 
        robot_loc: The initial location of the robot

    Returns:
        The path taken by the robot to complete the subtour, as a list of pairs.
    '''
    def __fitness_get_subtour(self, segment, robot_loc):
        grid = self.env.get_grid()
        task_list = self.env.get_task_list()

        # calculate subtour path taken by this robot
        astar = AStar(grid)
        path = []
        for vertex_id in segment:
            errand_idx = vertex_id - 1

            # path from robot to errand source
            src = robot_loc
            dest = task_list[errand_idx][0]
            curr_path = astar.a_star_search(src, dest)
            path.extend(curr_path)

            # path from errand source to errand destination
            src = dest
            dest = task_list[errand_idx][1]
            curr_path = astar.a_star_search(src, dest)
            path.extend(curr_path)

            # update robot's new location
            robot_loc = dest

        # append subtour to subtours list, and total path length of subtours
        return path 

    '''
    Helper fitness function that calculates the paths of all the subtours.
    
    Params:
        chromosome: The candidate solution

    Returns:
        The paths taken by every robot to complete their subtours.
    '''
    def __fitness_get_all_subtours(self, chromosome):
        robots = self.env.num_of_robots()
        tasks = self.env.num_of_tasks()
        robot_loc = self.env.get_robot_loc()
        segment_idx = self.__get_subtour_start_indices_of(chromosome)
        all_paths = []

        for m in range(robots):
            # subtour start-end indices
            start = segment_idx[m]
            end = start + chromosome[tasks + m]

            # calculate subtour path taken by this robot
            path = self.__fitness_get_subtour(
                chromosome[start:end], robot_loc[m]
            )
            all_paths.append(path.copy())

        return all_paths

    '''
    Helper function for fitness. An objective function that measures the 
        longest path length out of all the agents' paths.

    Params:
        chromosome: The candidate solution
        check_collisons: The option that checks for collisions in the tours

    Returns:
        The longest path length out of all the agents' paths.
    '''
    def __makespan(self, chromosome, check_collisions=False):
        # get the max length of each robot's subtour
        max_path_len = 0
        subtours = self.__fitness_get_all_subtours(chromosome)
        for subtour in subtours:
            max_path_len = max(max_path_len, len(subtour))

        # get collisions
        c = self.__constraint_collision(subtours) if check_collisions else 0
        return float('inf') if c > 0 else max_path_len

    '''
    Constraint to the objective function that determines if the robots collide 
        at any point in their paths

    Params:
        tours: The paths that every robot takes.
    Returns:
        The number of collisions found.
    '''
    def __constraint_collision(self, tours):
        collisions_detected = 0

        # pad uneven tours with a default value for easier comparing
        max_path_len = max(len(row) for row in tours)
        padding = (-1, -1)
        padded_tours = np.array(
            [path + [padding] * (max_path_len - len(path)) for path in tours]
        )

        # check the location of each robot at each point in time
        for col in range(max_path_len):
            locations = set()
            column = [padded_tours[row, col] for row in range(len(padded_tours))]
            for x, y in column:
                cell = (x, y)
                if cell == padding:
                    continue
                if cell in locations:
                    collisions_detected += 1
                else:
                    locations.add(cell)

        return collisions_detected
