import numpy as np
from mini2.src.utils.grid_world import windy_gridworld
from mini2.src.utils.value_convergence import policy_iteration
import matplotlib.pyplot as plt
import numpy as np

class WindyGridChasm:
    def __init__(self, start_state=(0, 3)): # Added start_state parameter
        self.grid_size = [20, 7] 
        self.reward_range = [-1, -1] 


        self.terminal_state_values = [-100, 100] # failure, success
        self.curr_grid = self.generate_grid()
        self.gamma = 0.99
        self.step_penalty = -1
        
        # Initial Location
        self.start_state = start_state
        self.current_state = start_state

        # build MDP with spec-compliant wind model
        self.S, self.A, self.P, self.r, self.terminal_states = \
            windy_gridworld(self.curr_grid, self.terminal_state_values, 0.4)
        
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

    def generate_grid(self):
        random_array = np.random.randint(
            self.reward_range[0],
            self.reward_range[1] + 1,
            [self.grid_size[0], self.grid_size[1]]
        )
        # Don't set terminal values in the mask anymore - handled by MDP
        return random_array


    def visualize_policy(self, policy_dict):
        ROWS, COLS = self.grid_size
        
        # Mapping directions to arrows (Forward is flipped to Down)
        arrow_map = {
            'left': '←',
            'right': '→',
            'forward': '↓',  # Flipped direction
            None: '●'        # Goal/Terminal
        }

        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(ROWS - 0.5, -0.5) # Row 0 at the top
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.grid(True, linestyle='--', alpha=0.5)

        for key, move in policy_dict.items():
            if isinstance(key, tuple):
                r, c = key
                symbol = arrow_map.get(move, '·')
                # Highlighting based on action
                if move == 'forward':
                    color = 'blue'
                elif move == 'left':
                    color = 'red'
                elif move == 'right':
                    color = 'green'
                else:
                    color = 'black'
                ax.text(c, r, symbol, ha='center', va='center', 
                        fontsize=16, fontweight='bold', color=color)

        plt.title("Grid Policy Visualization (Forward=↓, Left=←, Right=→)", fontsize=14, pad=20)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    chasm = WindyGridChasm()

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