import numpy as np
from scipy.ndimage import distance_transform_edt

def generate_grid(grid_size=(13, 7), terminal_state_values=(100.0,), slice_angle=80, row_penalty_weight=2.0):
    rows, cols = grid_size
    r0, c0 = rows // 2, cols - 1
    grid = np.full(grid_size, -np.inf, dtype=float)

    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    
    # Angle calculation centered at the right-middle (c0, r0)
    dr, dc = rr - r0, cc - c0
    angles = np.degrees(np.arctan2(dr, dc))
    ang_diff = np.abs((angles - 180.0 + 180) % 360 - 180)
    
    # Safe zone mask
    mask = (cc <= c0) & (ang_diff <= slice_angle / 2)

    # 1. Cubic penalty for proximity to boundaries
    dist_to_wall = distance_transform_edt(mask)
    max_d = np.max(dist_to_wall)
    wall_penalty = -((max_d - dist_to_wall) ** 4)

    # 2. Distance from center row (r0) penalty (SQUARED)
    dist_from_center_row = np.abs(rr - r0)
    row_penalty = -2 * dist_from_center_row

    # Combine rewards
    all_rewards = wall_penalty + row_penalty

    # all_rewards= np.ones_like(all_rewards)

    # Apply to mask
    grid[mask] = all_rewards[mask]

    # Bias the center row slightly higher (additive boost for the path)
    grid[r0, ~np.isinf(grid[r0, :])] = 200

    # The goal
    grid[r0, 0] = terminal_state_values[0]
    return np.round(grid, 2)

def movement_drift_transitions(i, j, a, moves, p_center, rows, cols, center):
    di, dj = moves[a]
    ni, nj = i + di, j + dj

    def is_inside_wedge(ri, ci):
        if not (0 <= ri < rows and 0 <= ci < cols):
            return False
        r0, c0 = rows // 2, cols - 1
        dr, dc = ri - r0, ci - c0
        # Normalizing angle to check if point is within the 80-degree wedge
        ang_diff = np.abs((np.degrees(np.arctan2(dr, dc)) - 180 + 180) % 360 - 180)
        return (ci <= c0) and (ang_diff <= 40)

    transitions = {}

    if a in ["left", "right"]:
        # Horizontal action: Drift happens in the j (column) direction
        # Note: If your 'moves' already changed j, drift adds/subtracts from that
        transitions[(ni, nj - 1)] = p_center / 2  # Drift Left
        transitions[(ni, nj + 1)] = p_center / 2  # Drift Right
        transitions[(ni, nj)] = 1.0 - p_center    # Stay on intended track
        
    elif a == "forward":
        # Forward action: Drift happens in the i (row) direction
        transitions[(ni - 1, nj)] = p_center / 2  # Drift Up
        transitions[(ni + 1, nj)] = p_center / 2  # Drift Down
        transitions[(ni, nj)] = 1.0 - p_center    # Stay on intended track
    
    else:
        # Default fallback for other actions
        transitions[(ni, nj)] = 1.0

    # Apply wedge / boundary crash logic
    final_transitions = {}
    for (next_i, next_j), prob in transitions.items():
        if not is_inside_wedge(next_i, next_j):
            # Any move outside the wedge results in a 'crash' state
            final_transitions['crash'] = final_transitions.get('crash', 0) + prob
        else:
            final_transitions[(next_i, next_j)] = final_transitions.get((next_i, next_j), 0) + prob

    return final_transitions



import matplotlib.pyplot as plt
import numpy as np

def visualize_lidar_policy(grid, policy_dict, grid_size):
    """
    Visualizes the LiDAR wedge policy.
    - Arrows point in the physical direction of movement.
    - Background colors indicate the navigable wedge vs. walls.
    - ★ = Goal, X = Wall/Crash area.
    """
    rows, cols = grid_size
    
    # Define arrow symbols based on the PHYSICAL direction of movement:
    # forward: (0, -1) -> Points towards col 0 (Left Arrow)
    # left: (1, 0)    -> Points towards higher row index (Down Arrow)
    # right: (-1, 0)   -> Points towards lower row index (Up Arrow)
    arrow_symbols = {
        "forward": "←",
        "left": "↓",
        "right": "↑"
    }
    
    # Color mapping for actions
    colors = {
        "forward": "blue",
        "left": "green",
        "right": "red"
    }

    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Set up the grid appearance
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5) # Invert Y to match grid indexing
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    # 1. Draw Background and Terminal States
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            
            # Wall / Crash Area
            if val == -np.inf:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gray', alpha=0.2))
                ax.text(c, r, 'X', ha='center', va='center', color='gray', alpha=0.5)
            
            # Goal State (100)
            elif val == 100:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gold', alpha=0.3))
                ax.text(c, r, '★', ha='center', va='center', fontsize=20, color='orange')
            
            # Navigable Path (Distance rewards)
            else:
                # Slight blue tint for navigable wedge
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='skyblue', alpha=0.05))

    # 2. Draw Policy Arrows
    for state, action in policy_dict.items():
        if not isinstance(state, tuple) or action is None:
            continue
            
        r, c = state
        symbol = arrow_symbols.get(action, "?")
        color = colors.get(action, "black")
        
        ax.text(c, r, symbol, ha='center', va='center', 
                fontsize=22, fontweight='bold', color=color)

    ax.set_title("LiDAR Wedge Optimal Policy\nBlue: Forward (Goal) | Green: Left (Down) | Red: Right (Up)", 
                 fontsize=14, pad=20)
    plt.xlabel("Columns (Distance)", fontsize=12)
    plt.ylabel("Rows (Lateral Position)", fontsize=12)
    
    plt.tight_layout()
    plt.show()

# To use it with your data:
# visualize_lidar_policy(grid, policy, grid_size)