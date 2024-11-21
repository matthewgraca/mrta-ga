import json
class GeneticAlgorithm:
    '''
    Parameters:
        pop_size: Population size, default 100
        pop_init: Population initialization method, default random
        selection: Selection method, default stochastic universal sampling
        crossover: Crossover method, default partially mapped crossover (pmx)
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
            crossover='pmx', 
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
            'crossover'     : {'pmx'},
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
        # initialization
        pop = self.__pop_init(method=self.pop_init, pop_size=self.pop_size)
        return pop
        '''
        # parent selection
        p1, p2 = self.parent_selection(method=self.selection, population=pop)
        # crossover
        c1, c2 = self.crossover(method=self.crossover, p1=p1, p2=p2)
        # mutation
        c1, c2 = self.mutation(method=self.mutation, c1=c1, c2=c2)
        # replacement
        pop = self.replacement(method=self.replacement, pop=pop, c1=c1, c2=c2)
        # termination
        '''

    '''
    Initializes the population of candidate solutions

    Params:
        method: The method being used to initialize
        pop_size: The size of the population

    Returns:
        The population of candidate solutions
    '''
    def __pop_init(self, method, pop_size):
        pop_init_methods = {
            'random' : self.__random_pop_init
        }
        return pop_init_methods[method](pop_size)

    def __random_pop_init(self, pop_size):
        return 0

