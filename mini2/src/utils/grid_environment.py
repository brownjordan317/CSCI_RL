import numpy as np
from src.utils.grid_world import gridworld
from src.utils.value_convergence import policy_iteration
import matplotlib.pyplot as plt
import numpy as np

class GridEnvironment:
    def __init__(self, 
                 curr_grid, 
                 reward_range, 
                 terminal_map,
                 action_map,
                 gamma, 
                 step_penalty,
                 random_transition_type,
                 random_probability,
                 start_state
                 ):
        
        self.curr_grid = curr_grid
        self.grid_size = self.curr_grid.shape
        self.terminal_map = terminal_map
        self.terminal_state_values = list(terminal_map.values())
        self.action_map = action_map
        self.reward_range = reward_range
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.random_transition_type = random_transition_type
        self.random_probability = random_probability
        
        # Initial Location
        self.start_state = start_state
        self.current_state = start_state

        # build MDP with spec-compliant wind model
        self.S, self.A, self.P, self.r, self.terminal_states = \
            gridworld(
                self.curr_grid, 
                terminal_map=self.terminal_map,
                action_map=self.action_map,
                transitions_type=self.random_transition_type,
                random_probability=self.random_probability
        )
        
        print(f"S size: {len(self.S)}")
        print(f"A: {self.A}")
        print(f"terminal_states: {self.terminal_states}")
        
        # Debug: check some transitions
        if (0, 3) in self.P:
            print(f"\nTransitions from (0,3) with 'forward': {self.P[(0,3)]['forward']}")
            print(f"Transitions from (0,3) with 'left': {self.P[(0,3)]['left']}")
            print(f"Transitions from (0,3) with 'right': {self.P[(0,3)]['right']}")

    def reset(self):
        """
        Resets the agent to the initial location.

        Returns
        -------
        state : tuple
            The initial state.
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, s, a):
        """
        Returns transitions with (probability, reward) tuples.

        Parameters
        ----------
        s : tuple
            The current state.
        a : str
            The action taken.

        Returns
        -------
        transitions : dict
            A dictionary with the next states as keys and tuples of (probability, reward) as values.
        """
        transitions = {}
        for s_next, prob in self.P[s][a].items():
            # Get the reward from the MDP's reward function
            reward = self.r.get((s, a, s_next), 0.0)
            transitions[s_next] = (prob, reward)
        return transitions
