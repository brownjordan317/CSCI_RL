import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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
    # Set the last row to the second terminal state value
    random_array[-1, :] = terminal_state_values[1]
    # Set the first and last columns to the first terminal state value
    random_array[:, 0] = terminal_state_values[0]
    random_array[:, -1] = terminal_state_values[0]
    
    return random_array


import matplotlib.pyplot as plt
import numpy as np


def visualize_policy(grid_size, policy_dict, mask=None):
    """
    Visualize the policy from a given policy dictionary on a grid of size grid_size.
    The policy is represented as arrows with RGB colors.
    """

    ROWS, COLS = grid_size

    # -----------------------------
    # Dynamically build arrow map
    # -----------------------------
    arrow_map = {}
    color_map = {}

    max_number = max(int(val.split("_")[1]) for val in policy_dict.values() if val)

    for val in policy_dict.values():
        if not val:
            continue

        number = int(val.split("_")[1])
        color_ratio = number / max_number

        if "left" in val:
            arrow_map[val] = "←"
            color_map[val] = (color_ratio, 0.0, 0.0)   # red
        elif "right" in val:
            arrow_map[val] = "→"
            color_map[val] = (0.0, color_ratio, 0.0)   # green
        elif "forward" in val:
            arrow_map[val] = "↓"
            color_map[val] = (0.0, 0.0, color_ratio)   # blue
        else:
            arrow_map[val] = "·"
            color_map[val] = (0.0, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, linestyle='--', alpha=0.5)

    # -----------------------------
    # Draw terminal cells
    # -----------------------------
    if mask is not None:
        for r in range(ROWS):
            for c in range(COLS):
                val = mask[r, c]

                if val == -20:
                    ax.add_patch(
                        plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      color=(1.0, 0.0, 0.0), alpha=0.15)
                    )
                    ax.text(c, r, 'X',
                            ha='center', va='center',
                            fontsize=16,
                            color=(0.8, 0.0, 0.0),
                            fontweight='bold')

                elif val == 100:
                    ax.add_patch(
                        plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      color=(0.0, 0.8, 0.0), alpha=0.15)
                    )
                    ax.text(c, r, '★',
                            ha='center', va='center',
                            fontsize=18,
                            color=(0.0, 0.6, 0.0),
                            fontweight='bold')

    # -----------------------------
    # Draw policy arrows
    # -----------------------------
    for key, move in policy_dict.items():
        if not isinstance(key, tuple) or not move:
            continue

        r, c = key

        symbol = arrow_map.get(move, '·')
        color = color_map.get(move, (0.0, 0.0, 0.0))

        ax.text(
            c, r,
            symbol,
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            color=color,
            bbox=dict(facecolor=(1, 1, 1), edgecolor='none', alpha=0.6)
        )

    ax.set_title(
        "Dynamic Grid Policy\n"
        "RGB-colored arrows | X = Crash | ★ = Exit",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


def wind_transitions(i, j, a, moves, p_center, rows, cols, center_col):
    """
    Wind model that pushes AWAY from center (dangerous!)
    
    At center (nj=3): Wind pushes to sides (columns 2 and 4)
    Off-center: Wind pushes FURTHER away from center toward the edges
    """
    
    rng = np.random.default_rng()

    # Step 1: Deterministic move
    di, dj = moves[a]
    ni, nj = i + di, j + dj

    transitions = {}

    first_fifth = (cols - 1) // 5
    
    if nj == center_col:
        # At center: wind pushes you to the SIDES (away from center)
        # "with probability p, pushes to (i,4) or (i,2) (50/50)"
        transitions[(ni, 
                     rng.integers(low=first_fifth + 1,
                                  high=(2 * first_fifth) + 1)
                    )] = p_center / 2  # Pushed left
        transitions[(ni, 
                     rng.integers(low=(3 * first_fifth) + 1,
                                  high=(4 * first_fifth) + 1)
                     )] = p_center / 2  # Pushed right
        
        # "otherwise, with probability (1-p)p², pushes to (i,5) or (i,1)"
        transitions[(ni, 
                     rng.integers(low=1, 
                                  high=first_fifth + 1)
                     )] = (1 - p_center) * (p_center**2) / 2
        transitions[(ni, 
                     rng.integers(low=(4 * first_fifth) + 1, 
                                  high=(5 * first_fifth) + 1)
                     )] = (1 - p_center) * (p_center**2) / 2
        
        # "otherwise, with probability (1-p)(1-p²), stays at (i,3)"
        transitions[(ni, 
                     rng.integers(low=(2 * first_fifth) + 1, 
                                  high=(3 * first_fifth) + 1)
                     )] = (1 - p_center) * (1 - p_center**2)
        
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