import numpy as np
import matplotlib.pyplot as plt

def generate_grid(
        grid_size=[20, 7], 
        reward_range=[-1, -1],
        terminal_state_values=[-20, 100]
        ):
    """
    Generates a grid of the given size with random rewards between the given reward range.
    The first and last columns of the grid are set to the first element of terminal_state_values.
    The last row of the grid is set to the second element of terminal_state_values.

    Parameters
    ----------
    grid_size (list): The size of the grid to generate.
    reward_range (list): The range of rewards to use for the grid.
    terminal_state_values (list): The values to use for the terminal states.

    Returns
    -------
    numpy array: The generated grid.
    """
    random_array = np.random.randint(
        reward_range[0],
        reward_range[1] + 1,
        [grid_size[0], grid_size[1]]
    )
    # Set the first and last columns to the first terminal state value
    random_array[:, 0] = terminal_state_values[0]
    random_array[:, -1] = terminal_state_values[0]
    # Set the last row to the second terminal state value
    random_array[-1, :] = terminal_state_values[1]
    return random_array


def visualize_policy(grid_size, policy_dict, mask=None):
    """
    Visualize the policy from a given policy dictionary on a grid of size grid_size.
    The policy is represented as arrows (←, →, ↓) on the grid.
    If a mask is provided, it is used to fill in the "Empty" Terminal Cells.

    Parameters
    ----------
    grid_size : tuple of two integers
        The size of the grid to visualize the policy on.
    policy_dict : dict
        A dictionary where the keys are tuples representing the position on the grid
        and the values are the actions to take at that position.
    mask : numpy array, optional
        A numpy array where the values are used to fill in the "Empty" Terminal Cells.
        The values should be either -20 (for crash_val) or 100 (for exit_val).

    Returns
    -------
    None
    """
    ROWS, COLS = grid_size
    
    arrow_map = {
        'left': '←',
        'right': '→',
        'forward': '↓',
        None: '·'
    }

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, linestyle='--', alpha=0.5)

    # 1. NEW: Logic to fill in the "Empty" Terminal Cells
    if mask is not None:
        for r in range(ROWS):
            for c in range(COLS):
                val = mask[r, c]
                if val == -20: # Match your crash_val
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='red', alpha=0.1))
                    ax.text(c, r, 'X', ha='center', va='center', color='red', alpha=0.6)
                elif val == 100: # Match your exit_val
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='green', alpha=0.1))
                    ax.text(c, r, '★', ha='center', va='center', color='green')

    # 2. Existing Policy Drawing
    for key, move in policy_dict.items():
        if isinstance(key, tuple):
            r, c = key
            symbol = arrow_map.get(move, '·')
            
            color = 'blue' if move == 'forward' else \
                    'red' if move == 'left' else \
                    'green' if move == 'right' else 'black'
            
            ax.text(c, r, symbol, ha='center', va='center', 
                    fontsize=16, fontweight='bold', color=color)

    plt.title("Dynamic Grid Policy (Arrows=Decisions, X=Crash, ★=Exit)")
    plt.show()

def wind_transitions(i, j, a, moves, p_center, rows, cols, center_col):
    """
    Wind model that pushes AWAY from center (dangerous!)
    
    At center (nj=3): Wind pushes to sides (columns 2 and 4)
    Off-center: Wind pushes FURTHER away from center toward the edges
    """
    
    # Step 1: Deterministic move
    di, dj = moves[a]
    ni, nj = i + di, j + dj

    transitions = {}
    
    if nj == center_col:
        # At center: wind pushes you to the SIDES (away from center)
        # "with probability p, pushes to (i,4) or (i,2) (50/50)"
        transitions[(ni, 2)] = p_center / 2  # Pushed left
        transitions[(ni, 4)] = p_center / 2  # Pushed right
        
        # "otherwise, with probability (1-p)p², pushes to (i,5) or (i,1)"
        transitions[(ni, 1)] = (1 - p_center) * (p_center**2) / 2
        transitions[(ni, 5)] = (1 - p_center) * (p_center**2) / 2
        
        # "otherwise, with probability (1-p)(1-p²), stays at (i,3)"
        transitions[(ni, 3)] = (1 - p_center) * (1 - p_center**2)
        
    else:
        # Off-center: wind probability scales with distance, pushes AWAY from center
        E_j = 1 / (1 + (nj - center_col)**2)
        p_wind = p_center ** E_j
        
        # Wind pushes AWAY from center
        if nj < center_col:
            # Left of center: wind pushes FURTHER left (away from center)
            transitions[(ni, nj - 1)] = p_wind
            transitions[(ni, nj)] = 1 - p_wind
        else:
            # Right of center: wind pushes FURTHER right (away from center)
            transitions[(ni, nj + 1)] = p_wind
            transitions[(ni, nj)] = 1 - p_wind

    # Handle crashes and exits
    final_transitions = {}
    for (i_t, j_t), prob in transitions.items():
        if j_t < 0 or j_t >= cols:
            final_transitions['crash'] = final_transitions.get('crash', 0) + prob
        elif i_t >= rows - 1:
            final_transitions['exit'] = final_transitions.get('exit', 0) + prob
        else:
            final_transitions[(i_t, j_t)] = prob

    return final_transitions