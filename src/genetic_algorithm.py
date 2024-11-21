class GeneticAlgorithm:
    '''
        Parameters:
            pop_size: Population size, default 100
            crossover: Crossover method, default partially mapped crossover 
                (pmx)
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
        self.pop_size = pop_size
        self.crossover = crossover
        self.mutation = mutation
        self.pc = pc
        self.pm = pm

    '''
        Returns: The current parameter setup of the genetic algorithm as 
        a dictionary.
    '''
    def get_parameters(self):
        return {
            'pop_size' : self.pop_size,
            'crossover' : self.crossover,
            'mutation' : self.mutation,
            'pc' : self.pc,
            'pm' : self.pm
        }

    def test(self):
        return 0
