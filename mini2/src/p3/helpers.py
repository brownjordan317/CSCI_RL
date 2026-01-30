import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class Component(Enum):
    EMPTY = 0
    WIRE = 1
    NAND = 2
    SOURCE_A = 3
    SOURCE_B = 4
    SINK_C = 5

class CircuitEvaluator:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.SOURCE_A_POS = (0, 0)
        self.SOURCE_B_POS = (2, 0)
        self.SINK_C_POS = (1, 2)
    
    def _check_physical_connectivity(self, components_dict):
        """BFS to see if any path exists from sources to sink"""
        start_nodes = [self.SOURCE_A_POS, self.SOURCE_B_POS]
        queue = list(start_nodes)
        visited = set(start_nodes)
        while queue:
            curr = queue.pop(0)
            if curr == self.SINK_C_POS: return True
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (curr[0] + di, curr[1] + dj)
                if neighbor in components_dict and neighbor not in visited:
                    if components_dict[neighbor] in [Component.WIRE, Component.NAND, Component.SINK_C]:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return False

    def evaluate_circuit(self, components, correct_reward=5000, incorrect_penalty=-200):
        # 1. Truth table check for XOR: (0,0,0), (0,1,1), (1,0,1), (1,1,0)
        test_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        correct_count = sum(1 for A, B, exp in test_cases 
                            if self.propagate_signals(components, A, B) == exp)
        
        # 2. Physical Connectivity Check (The BFS you wrote)
        is_connected = self._check_physical_connectivity(components)
        
        # 3. Calculate Shaped Reward
        if correct_count == 4:
            return correct_reward # Jackpot
        
        # Give the agent "breadcrumbs" so it doesn't stall at Iteration 3
        connectivity_bonus = 150 if is_connected else 0
        logic_bonus = correct_count * 100 
        
        return incorrect_penalty + connectivity_bonus + logic_bonus
    
    def propagate_signals(self, components, A, B):
        # Initialize signals with input values [cite: 89]
        signals = np.zeros((self.grid_size, self.grid_size), dtype=int)
        signals[self.SOURCE_A_POS] = A
        signals[self.SOURCE_B_POS] = B
        
        # Multi-pass settling to allow signals to flow through the $3\times3$ grid [cite: 84]
        for _ in range(self.grid_size * 2):
            old_signals = signals.copy()
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    # Sources are fixed [cite: 89]
                    if (row, col) in [self.SOURCE_A_POS, self.SOURCE_B_POS]: 
                        continue
                    
                    comp = components.get((row, col), Component.EMPTY)
                    inputs = []
                    
                    # Check 4-neighbor connectivity
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = row + di, col + dj
                        
                        # BLOCK signals between (1,1) and (2,1) 
                        if ((row, col) == (1, 1) and (ni, nj) == (2, 1)) or \
                        ((row, col) == (2, 1) and (ni, nj) == (1, 1)):
                            continue
                            
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            inputs.append(signals[ni, nj])
                    
                    # Apply Component Logic [cite: 92]
                    if comp == Component.WIRE:
                        signals[row, col] = int(any(inputs)) if inputs else 0
                    elif comp == Component.NAND:
                        # NAND: 0 if all inputs are 1, else 1 (requires at least one input) [cite: 92]
                        active_inputs = [i for i in inputs if i is not None]
                        if active_inputs:
                            signals[row, col] = int(not all(active_inputs))
                        else:
                            signals[row, col] = 1 # NAND with no inputs defaults to High
            
            if np.array_equal(old_signals, signals): 
                break
                
        return signals[self.SINK_C_POS] # Returns the signal at C (1, 2) [cite: 90]

class CircuitConfigManager:
    """Manages circuit configurations and encoding/decoding"""
    
    def __init__(self, grid_size=3, fixed_positions=None):
        self.grid_size = grid_size
        self.fixed_positions = fixed_positions or {
            (0, 0): Component.SOURCE_A,
            (2, 0): Component.SOURCE_B,
            (1, 2): Component.SINK_C
        }
        self.free_cells = [
            (i, j) for i in range(grid_size) for j in range(grid_size)
            if (i, j) not in self.fixed_positions
        ]
    
    def encode(self, components_dict):
        """Encode circuit configuration as hashable tuple"""
        return tuple(
            components_dict.get((i, j), Component.EMPTY)
            for i in range(self.grid_size) 
            for j in range(self.grid_size)
        )
    
    def decode(self, config_tuple):
        """Decode configuration tuple back to dictionary"""
        components = {}
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                components[(i, j)] = config_tuple[idx]
                idx += 1
        components.update(self.fixed_positions)
        return components
    
    def generate_all_configs(self):
        """Generate all possible circuit configurations"""
        configs = []
        component_types = [Component.EMPTY, Component.WIRE, Component.NAND]
        
        def backtrack(idx, current):
            if idx == len(self.free_cells):
                configs.append(self.encode(current))
                return
            
            cell = self.free_cells[idx]
            for comp in component_types:
                current[cell] = comp
                backtrack(idx + 1, current)
                del current[cell]
        
        backtrack(0, {})
        return configs
    
def circuit_gridworld(action_map=None, grid_size=3, step_penalty=-0.1, 
                      correct_reward=100, incorrect_penalty=-50):
    """
    Constructs a circuit design MDP. 
    Accepts 5 arguments to match the call in grid_world.py.
    """
    # Initialize helpers
    config_manager = CircuitConfigManager(grid_size)
    evaluator = CircuitEvaluator(grid_size)
    
    # Define the XOR test cases locally so handle_done can see them
    test_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    if action_map is None:
        action_map = {
            "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
            "place_wire": (0, 0), "place_nand": (0, 0), "remove": (0, 0), "done": (0, 0)
        }
    
    movement_actions = {k: v for k, v in action_map.items() if v != (0, 0)}
    all_configs = config_manager.generate_all_configs()
    
    # S includes standard states and terminal markers
    S = {((i, j), config) for i in range(grid_size) for j in range(grid_size) for config in all_configs}
    A = set(action_map.keys())
    P, r = {}, {}
    terminal_states = set()

    for state in S:
        P[state] = {}
        agent_pos, config = state
        
        for action in A:
            if action == "done":
                components = config_manager.decode(config)
                correct_count = sum(1 for A_in, B_in, exp in test_cases 
                                if evaluator.propagate_signals(components, A_in, B_in) == exp)
                
                if correct_count == 4:
                    reward = 1000  # The goal [cite: 33]
                    next_state = ("TERMINAL", config) # ONLY terminate on success
                    terminal_states.add(next_state)
                else:
                    # If wrong, penalize the attempt but stay in the grid.
                    # This forces Value Iteration to find the actual solution.
                    reward = -200 
                    next_state = state
                
            elif action in movement_actions:
                di, dj = movement_actions[action]
                ni, nj = agent_pos[0] + di, agent_pos[1] + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    next_state, reward = ((ni, nj), config), step_penalty
                else:
                    next_state, reward = state, step_penalty
                    
            else: # Component placement
                comp_type = Component.WIRE if "wire" in action else (Component.NAND if "nand" in action else Component.EMPTY)
                if agent_pos in config_manager.fixed_positions:
                    next_state, reward = state, step_penalty
                else:
                    comps = config_manager.decode(config)
                    comps[agent_pos] = comp_type
                    next_state, reward = (agent_pos, config_manager.encode(comps)), step_penalty
            
            P[state][action] = {next_state: 1.0}
            r[(state, action, next_state)] = reward

    # Add terminal absorbing states
    for ts in terminal_states:
        P[ts] = {a: {ts: 1.0} for a in A}
        for a in A: r[(ts, a, ts)] = 0.0

    return S, A, P, r, terminal_states

def visualize_optimal_circuit(state, title="Circuit Design"):
    # 1. Safely unpack the cursor and the components
    # State is ((row, col), (c0, c1, c2, c3, c4, c5, c6, c7, c8))
    cursor_pos, components = state
    
    # 2. Convert Enums to raw integers for the grid
    grid_data = [c.value if hasattr(c, 'value') else c for c in components]
    grid = np.array(grid_data).reshape(3, 3)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Updated Map based on your pinout:
    # (0,0) = Input A, (0,2) = Input B, (1,2) = Output C
    for i in range(3):
        for j in range(3):
            val = int(grid[i, j])
            
            # Determine color/label based on position AND component type
            if (i, j) == (0, 0):
                color, label = "lightblue", "Source A"
            elif (i, j) == (2, 0):
                color, label = "lightblue", "Source B"
            elif (i, j) == (1, 2):
                color, label = "lightgreen", "Sink C"
            else:
                # Standard component mapping
                mapping = {0: ("white", "Empty"), 1: ("lightgrey", "Wire"), 2: ("salmon", "NAND")}
                color, label = mapping.get(val, ("grey", "Unknown"))
            
            # Draw the block (i=row, j=col)
            # We use 2-i to flip the y-axis so row 0 is at the top
            rect = plt.Rectangle((j-0.5, 2-i-0.5), 1, 1, facecolor=color, edgecolor='black', lw=2)
            ax.add_patch(rect)
            ax.text(j, 2-i, label, ha='center', va='center', weight='bold', fontsize=9)
            
            # Draw cursor
            if (i, j) == cursor_pos:
                cursor = plt.Circle((j, 2-i), 0.15, color='red', zorder=10)
                ax.add_patch(cursor)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title(title, pad=20)
    plt.axis('off')
    plt.show()