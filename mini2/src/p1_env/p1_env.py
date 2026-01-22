import numpy as np

class WindyGridChasm:
    def __init__(self):
        self.grid_size = [20,7]
        self.reward_range = [-3, -1]
        self.curr_grid = self.generate_grid

    def generate_grid(self):
        random_array = np.random.uniform(
            self.reward_range[0], 
            self.reward_range[1], 
            [self.grid_size[0], self.grid_size[1] - 1]
        )
        exit_space = np.full([self.grid_size[0], 1], )