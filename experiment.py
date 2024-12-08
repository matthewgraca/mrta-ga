from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np

def main():
    # TODO potentially have map seeded with initial robot locations
    np.random.seed(0)
    env = EnvironmentInitializer(robots=3, tasks=10, robot_loc=[(3, 8), (5, 3), (20, 19)])
    ga = GeneticAlgorithm(pop_size=100, env=env)
    ga.run()

if __name__=='__main__':
    main()
