import heapq

class Node:
    def __init__(self):
        # parent cell row/col index
        self.i = 0
        self.j = 0
        
        # cost of cell f = g + h
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.max_row = len(grid)
        self.max_col = len(grid[0])
        self.node = Node()

    '''
    Determines if a given pair of indices are within range of the grid.

    Params:
        row: The row index
        col: The column index

    Returns:
        True if the row and column are in range, false if not
    '''
    def is_valid(self, row, col):
        return (
            row >= 0 and row < self.max_row and
            col >= 0 and col < self.max_col
        )

    '''
    Determines if the given cell is blocked (invalid path)
        We currently use '@' to denote a block, '.' for unblocked

    Params:
        grid: The map being traversed
        row: The row index
        col: The column index

    Returns:
        True if the cell is a valid path, false if not
    '''
    def is_unblocked(self, grid, row, col):
        return grid[row][col] == '.'

    '''
    Determines if the given cell is the destination

    Params:
        dest: The coordinates of the destination
        row: The row index
        col: The column index

    Returns:
        True if the cell is the destination, false if not
    '''
    def is_destination(self, row, col, dest):
        x, y = dest
        return row == x and col == y

    '''
    Calculates the heuristic -- currently euclidean, straight line distance.

    Params:
        dest: The coordinates of the destination
        row: The row index
        col: The column index

    Returns:
        The euclidean distance between the given cell and the destination
    '''
    def heuristic_func(self, dest, row, col):
        x, y = dest[0], dest[1]
        return ((row - x) ** 2 + (col - y) ** 2) ** 0.5
    
    '''
    Traces the path taken from the source to the destination

    Params:
        cell_details: 2D list of nodes
        dest: The destination

    Returns:
        The path taken to reach the destination
    '''
    def trace_path(self, cell_details, dest):
        print("The Path is ")
        path = []
        row, col = dest[0], dest[1]

        # Trace the path from destination to source using parent cells
        while not (cell_details[row][col].i == row and cell_details[row][col].j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].i
            temp_col = cell_details[row][col].j
            row = temp_row
            col = temp_col

        # Add the source cell to the path
        path.append((row, col))

        # Print the path in reverse to get source -> destination
        for i in reversed(path):
            print("->", i, end=" ")
        print()

    '''
    Performs A* search

    Params:
        grid: The space that is being traversed
        src: The location of the source cell
        dest: The locatin of the destination cell

    Returns:
        None
    '''
    def a_star_search(self, grid, src, dest):
        # run initial validity check for src and dest
        print(dest)
        self.__initial_validity_check(grid, src, dest)

        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.max_col)] for _ in range(self.max_row)]

        # embed each cell with a node containing details
        cell_details = [[Node() for _ in range(self.max_col)] for _ in range(self.max_row)]

        # Initialize the start cell details
        curr_x, curr_y = src
        cell = cell_details[curr_x][curr_y]
        cell.f = cell.g = cell.h = 0
        cell.i, cell.j = curr_x, curr_y

        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, curr_x, curr_y))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while open_list:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            _, curr_x, curr_y = p
            closed_list[curr_x][curr_y] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for direction in directions:
                next_x = curr_x + direction[0]
                next_y = curr_y + direction[1]

                # If the successor is valid, unblocked, and not visited
                if (
                    self.is_valid(next_x, next_y) and 
                    self.is_unblocked(grid, next_x, next_y) and 
                    not closed_list[next_x][next_y]
                ):
                    # If the successor is the destination
                    if self.is_destination(next_x, next_y, dest):
                        # Set the parent of the destination cell
                        cell_details[next_x][next_y].i = curr_x 
                        cell_details[next_x][next_y].j = curr_y
                        print("The destination cell is found")

                        # Trace and print the path from source to destination
                        self.trace_path(cell_details, dest)
                        found_dest = True

                        return
                    else:
                        # Calculate the new f, g, and h values
                        g_new = cell_details[curr_x][curr_y].g + 1.0
                        h_new = self.heuristic_func(dest, next_x, next_y)
                        f_new = g_new + h_new

                        # If the cell is not in the open list or the new f value is smaller
                        new_cell = cell_details[next_x][next_y]
                        if new_cell.f == float('inf') or new_cell.f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, next_x, next_y))
                            # Update the cell details
                            new_cell.f = f_new
                            new_cell.g = g_new
                            new_cell.h = h_new
                            new_cell.i = curr_x
                            new_cell.j = curr_y

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")

    '''
    Performs initial validity check:
        - If the source and destination cells are valid
        - If the source and destination are not blocked cells
        - If the source is not already at the destination

    Params:
        grid: The space being traversed
        src: The source cell
        dest: The destination cell

    Returns:
        None
    '''
    def __initial_validity_check(self, grid, src, dest):
        src_x, src_y = src
        dest_x, dest_y = dest

        # Check if the source and destination are valid
        if (
            not self.is_valid(src_x, src_y) or 
            not self.is_valid(dest_x, dest_y)
        ):
            raise ValueError("Source or destination is invalid")

        # Check if the source and destination are unblocked
        if (
            not self.is_unblocked(grid, src_x, src_y) or 
            not self.is_unblocked(grid, dest_x, dest_y)
        ):
            raise ValueError("Source or the destination is blocked")

        # Check if we are already at the destination
        if self.is_destination(src_x, src_y, dest):
            raise ValueError("We are already at the destination")

        return 

