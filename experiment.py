from src.genetic_algorithm import GeneticAlgorithm
from src.environment_initializer import EnvironmentInitializer
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(0)
    env = EnvironmentInitializer(robots=3, tasks=10)
    generations = 500
    update_step = generations // 10

    Y = []

    # simple experiment
    # exp 1: baseline
    ga = GeneticAlgorithm(
        pop_size=50, pop_init='greedy', 
        replacement='replace_worst',
        mutation='swap', pc=0.85, pm= 0.01, 
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(100, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)

    # print best path
    print("Best path of the robots --")
    for i in range(len(best_path)):
        print(f'Path of robot {i+1}: {best_path[i]}')

    # various experiments that were run
    '''
    # exp 2
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='greedy', 
        replacement='elitism',
        mutation='swap', pc=0.85, pm= 0.01, 
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(generations, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)
    '''

    '''
    # exp 3: baseline, lower lambda 
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='greedy', 
        replacement='replace_worst',
        mutation='swap', pc=0.85, pm= 0.1, 
        replace_percent=0.1,
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(generations, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)

    # exp 4, higher mutation
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='greedy', 
        replacement='elitism',
        mutation='swap', pc=0.85, pm= 0.01, 
        replace_percent=0.1,
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(generations, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)
    '''

    '''
    # exp 1: baseline
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='greedy', 
        replacement='replace_worst',
        mutation='swap', pc=0.85, pm= 0.01, 
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(generations, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)

    # exp 1: baseline with higher mutation
    ga = GeneticAlgorithm(
        pop_size=100, pop_init='greedy', 
        replacement='replace_worst',
        mutation='swap', pc=0.85, pm= 0.5, 
        env=env
    )
    print(ga.get_params())
    avg_fits, best_fits, best_path = ga.run(generations, update_step)
    y = (avg_fits, best_fits)
    Y.append(y)
    '''

    plot(Y, "Fitness of experiments over generations")

# Y = [(y1, y2), (y3, y4) ...]
def plot(Y, title):
    fig, ax = plt.subplots()
    x = np.arange(0, len(Y[0][0]))
    i = 1
    for y1, y2 in Y:
        plt.plot(x, y1, label=f'Avg fit for experiment {i}')
        plt.plot(x, y2, label=f'Best fit for experiment {i}')
        i += 1

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    #fig.set_size_inches(6.875, 6.875) # for report; comment out otherwise. 
    plt.show()

if __name__=='__main__':
    main()
