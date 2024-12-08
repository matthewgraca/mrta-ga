# mrta-ga
Multi-Robot Task Allocation with Genetic Algorithms
# Tests
To run the tests from here, run the command: `python3 -m unittest discover -v test/ "*_test.py"` 
# Files
## working directory 
- `experiment.py` contains the driver code that runs the experiments.
## maps
- Contains the maps and the tasks lists.
## src
- `a_star.py` contains the implementation for the path planning algorithm, A\*.
- `genetic_algorithm.py` contains the implementation for the genetic algorithm responsible for creating a close-to optimal task schedule.
- `environment_initializer.py` contains the details of the environment -- the tasks, the map, the number of robots, their locations, etc.

