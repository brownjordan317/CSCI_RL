import numpy as np
from grid_world import windy_gridworld
from value_convergence import value_iteration


class WindyGridChasm:
    def __init__(self):
        self.grid_size = [20, 7] # row, col
        self.reward_range = [-10, -1] # rang of rewards
        self.terminal_state_value = 1
        self.curr_grid = self.generate_grid()
        self.gamma = 0.3

        # step penalty
        self.step_penalty = -1.0

        # build MDP
        self.S, self.A, self.P, self.r, self.terminal_states = \
            windy_gridworld(self.curr_grid, self.terminal_state_value)

    def generate_grid(self):
        random_array = np.random.randint(
            self.reward_range[0],
            self.reward_range[1] + 1,
            [self.grid_size[0] - 1, self.grid_size[1]]
        )

        exit_space = np.full([1, self.grid_size[1]], self.terminal_state_value)
        return np.vstack([random_array, exit_space])

    def step(self, s, a):
        """
        Environment transition model.
        Returns:
            dict: s_next: (probability, reward)
        """
        transitions = {}

        for s_next, prob in self.P[s][a].items():

            env_reward = self.r.get((s, a, s_next), 0.0) - 1
            reward = env_reward + self.step_penalty

            transitions[s_next] = (prob, reward)

        return transitions


chasm = WindyGridChasm()

V, policy = value_iteration(
    chasm.S,
    chasm.A,
    chasm,
    chasm.terminal_states,
    gamma=chasm.gamma
)

print("Optimal policy:")
print(policy)
print("Number of states:", len(policy))
