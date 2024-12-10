from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

def main():
    # TODO potentially have map seeded with initial robot locations
    np.random.seed(0)
    env = EnvironmentInitializer(robots=3, tasks=10)
    ga = GeneticAlgorithm(pop_size=100, pc=0.85, pm= 0.01, env=env)
    print(ga.get_params())
    generations = 50
    update_step = generations // 10
    ga.run(generations, update_step)

if __name__=='__main__':
    main()
