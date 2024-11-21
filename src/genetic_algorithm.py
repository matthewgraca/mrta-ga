class GeneticAlgorithm:
    '''
    Parameters:
        pop_size: Population size, default 100
        crossover: Crossover method, default partially mapped crossover (pmx)
        mutation: Mutation method, default inverse mutation
        pc: Probability of crossover, default 0.4
        pm: Probability of mutation, default 0.6
    '''
    def __init__(
            self, 
            pop_size=100, 
            crossover='pmx', 
            mutation='inverse', 
            pc=0.4, 
            pm=0.6
    ):
        # check parameters
        self.__validate_params(pop_size, crossover, mutation, pc, pm)

        self.pop_size = pop_size
        self.crossover = crossover 
        self.mutation = mutation
        self.pc = pc
        self.pm = pm

    '''
    Validates the parameters. Throws errors if they are not valid
    '''
    def __validate_params(self, pop_size, crossover, mutation, pc, pm):
        # dictionary of valid parameters
        valid_params = {
            'crossover' : {'pmx'},
            'mutation' : {'inverse'}
        }

        # limit pop size to [1, 100000]
        if not 1 <= pop_size <= 100000:
            raise ValueError(f"pop_size should be in the range [1, 100000]")
        # ensure crossover and mutation methods are defined
        if crossover not in valid_params['crossover']: 
            raise ValueError(f"Crossover method \'{crossover}\' is not defined")
        if mutation not in valid_params['mutation']:
            raise ValueError(f"Mutation method \'{mutation}\' is not defined")
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
            'pop_size' : self.pop_size,
            'crossover' : self.crossover,
            'mutation' : self.mutation,
            'pc' : self.pc,
            'pm' : self.pm
        }
