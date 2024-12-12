from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np
from itertools import product

def main():
    np.random.seed(0)
    env = EnvironmentInitializer(robots=3, tasks=10)
    generations = 500
    update_step = generations // 10

    # experiment -- random init
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='random', 
        mutation='swap', pc=0.85, pm= 0.01, 
        env=env
    )
    print(ga.get_params())
    ga.run(generations, update_step)

    # experiment -- with elitism 
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='random', 
        replacement='elitism',
        mutation='swap', pc=0.85, pm= 0.01, 
        env=env
    )
    print(ga.get_params())
    ga.run(generations, update_step)


if __name__=='__main__':
    main()
