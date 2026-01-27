import numpy as np
from grid_world import windy_gridworld
from value_convergence import value_iteration

class WindyGridChasm:
    def __init__(self):
        self.grid_size = [20, 7]      # rows, columns
        self.reward_range = [-3, -1]  # rewards range
        self.curr_grid = self.generate_grid()  # call the method here

        self.gamma = 0.9

    def generate_grid(self):
        # Random rewards for the 19 rows before the exit
        random_array = np.random.randint(
            self.reward_range[0],
            self.reward_range[1] + 1,
            [self.grid_size[0] - 1, self.grid_size[1]]
        ).astype(int)

        # Exit row with terminal state reward
        exit_space = np.full([1, self.grid_size[1]], 1)

        self.curr_grid = np.vstack([random_array, exit_space])
        return self.curr_grid
    
    def compute_optimal_policy(self):
        S, A, P, r, terminal_states = windy_gridworld(self.curr_grid)
        V, policy = value_iteration(
            S, 
            A, 
            P, 
            r, 
            terminal_states, 
            gamma=self.gamma
        )
        
        self.V = V
        self.policy = policy
        return V, policy

chasm = WindyGridChasm()
V, policy = chasm.compute_optimal_policy()

# full optimal policy
print(f"Optimal policy:\n{policy}")
print(len(policy))


