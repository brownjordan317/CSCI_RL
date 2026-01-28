import numpy as np
from src.utils.grid_world import gridworld
from src.utils.value_convergence import policy_iteration
import matplotlib.pyplot as plt
import numpy as np

class GridEnvironment:
    def __init__(self, 
                 curr_grid, 
                 reward_range, 
                 terminal_state_values,
                 gamma, 
                 step_penalty,
                 random_transition_type,
                 random_probability,
                 start_state
                 ):
        
        self.curr_grid = curr_grid
        self.grid_size = self.curr_grid.shape
        self.terminal_state_values = terminal_state_values
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
                self.terminal_state_values, 
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
        """Resets the agent to the initial location."""
        self.current_state = self.start_state
        return self.current_state

    def step(self, s, a):
        """
        Returns transitions with (probability, reward) tuples.
        """
        transitions = {}
        for s_next, prob in self.P[s][a].items():
            # Get the reward from the MDP's reward function
            reward = self.r.get((s, a, s_next), 0.0)
            transitions[s_next] = (prob, reward)
        return transitions


if __name__ == "__main__":
    chasm = GridEnvironment()

    # Change this line
    V, policy = policy_iteration(
        chasm.S,
        chasm.A,
        chasm,
        chasm.terminal_states,
        chasm.terminal_state_values,
        gamma=chasm.gamma
    )

    print("\nOptimal policy sample:")
    for i in range(5):
        for j in range(7):
            if (i, j) in policy:
                print(f"({i},{j}): {policy[(i,j)]}", end="  ")
        print()
    
    print(f"\nTotal states in policy: {len(policy)}")
    
    # Count actions
    action_counts = {}
    for s, a in policy.items():
        if isinstance(s, tuple):
            action_counts[a] = action_counts.get(a, 0) + 1
    print(f"Action distribution: {action_counts}")
    
    chasm.visualize_policy(policy)