import os
import numpy as np
import matplotlib.pyplot as plt
import random

os.makedirs("heatmaps", exist_ok=True)

def value_dict_to_grid(V, grid_shape, terminal_states):
    """
    Converts a value function dictionary to a numpy grid.

    Args:
        V (dict): The value function dictionary
        grid_shape (tuple): The shape of the grid
        terminal_states (set): The set of terminal states

    Returns:
        grid (numpy array): The numpy grid representing the value function

    """
    grid = np.full(grid_shape, np.nan)

    # Iterate over the value function dictionary
    for s, v in V.items():
        # If the state is a terminal state, skip it
        if s in terminal_states:
            continue
        # If the state is a tuple, it represents a grid position
        if isinstance(s, tuple):
            # Check if it's a simple (r, c) tuple (P1, P2)
            if len(s) == 2 and isinstance(s[0], int) and isinstance(s[1], int):
                r, c = s
                # Make sure it fits in the grid
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    grid[r, c] = float(v)
            # For P3, states are ((position), (circuit_config)) - skip visualization
            # We can't easily visualize a 6561-dimensional state space on a 2D grid

    return grid


def save_value_heatmap(V, iteration, env, out_dir="heatmaps"):
    """
    Save heatmap visualization of value function.
    Works for P1/P2 (simple grid states) but skips P3 (complex states).
    """
    # Check if environment has curr_grid attribute (P1, P2)
    if not hasattr(env, 'curr_grid') or env.curr_grid is None:
        # Skip heatmap for P3 - state space is too complex to visualize
        return
    
    grid = value_dict_to_grid(
        V,
        env.curr_grid.shape,
        env.terminal_states
    )

    # 1. Get the raw terminal values from the environment
    vmin_env, vmax_env = env.terminal_state_values
    
    # 2. Safety check: Filter out inf/nan from the grid to find actual data range
    # This prevents casting 'inf' to 'int'
    mask = np.isfinite(grid)
    if np.any(mask):
        actual_min = np.min(grid[mask])
        actual_max = np.max(grid[mask])
        # Use the more restrictive of environment limits vs actual data
        # (or just use actual_min/max if you want the scale to auto-adjust)
        plot_vmin = max(vmin_env, actual_min) if np.isfinite(vmin_env) else actual_min
        plot_vmax = min(vmax_env, actual_max) if np.isfinite(vmax_env) else actual_max
    else:
        # Fallback if the whole grid is NaN or Inf
        plot_vmin, plot_vmax = -10, 10

    plt.figure(figsize=(6, 12))
    
    # Removed the int() casts that cause the OverflowError
    im = plt.imshow(
        grid,
        cmap="jet",
        origin="upper",
        vmin=plot_vmin * 1.2, 
        vmax=plot_vmax * 1.2
    )
    
    plt.colorbar(im)
    plt.title(f"Value Function – Iteration {iteration}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/V_iter_{iteration:04d}.png")
    plt.close()


def policy_evaluation(env, V, policy, tol=1e-4):
    """
    Policy evaluation with support for both P1/P2 and P3.
    
    P1/P2: Uses env.step(s, a) which returns {s_next: (prob, reward)}
    P3: env.step(s, a) also returns {s_next: (prob, reward)} via GridEnvironment wrapper
    
    Args:
        env: Environment object
        V: Value function dictionary
        policy: Policy dictionary
        tol: Convergence tolerance
    
    Returns:
        V: Updated value function
    """

    gamma = env.gamma
    while True:
        delta = 0
        V_new = V.copy() # Use a copy to ensure stable updates
        for s in env.S:
            a = policy[s]
            if a is None: continue
            
            V_new[s] = sum(prob * (reward + gamma * V.get(s_next, 0.0)) 
                        for s_next, (prob, reward) in env.step(s, a).items())
            delta = max(delta, abs(V[s] - V_new[s]))
        V = V_new
        if delta < tol:
            break
    return V


def policy_improvement(env, V, policy):
    """
    Policy improvement with support for both P1/P2 and P3.
    
    Args:
        env: Environment object
        V: Value function
        policy: Current policy
    
    Returns:
        new_policy: Improved policy
        policy_stable: Whether policy converged
    """
    gamma = env.gamma
    policy_stable = True
    new_policy = policy.copy()

    for s in env.S:
        old_action = policy[s]
        
        action_values = {}
        for a in env.A:
            # Use .get() here as well to handle unseen circuit states
            action_values[a] = sum(prob * (reward + gamma * V.get(s_next, 0.0)) 
                                   for s_next, (prob, reward) in env.step(s, a).items())
        
        best_action = max(action_values, key=action_values.get)
        new_policy[s] = best_action

        if old_action != best_action:
            # Check for significant improvement to avoid infinite loops from float noise
            if action_values[best_action] > action_values[old_action] + 1e-9:
                policy_stable = False

    return new_policy, policy_stable


def policy_iteration(env):
    """
    Runs the policy iteration algorithm to find the optimal policy and 
    value function.
    
    Works for:
    - P1: Windy Chasm (GridEnvironment with curr_grid)
    - P2: Robot Motion Control (GridEnvironment with curr_grid)
    - P3: Circuit Design (GridEnvironment without curr_grid, complex states)

    Args:
        env: Environment object with attributes:
            - S: set of non-terminal states
            - terminal_states: set of terminal states
            - A: set of actions
            - gamma: discount factor
            - step(s, a): method that returns {s_next: (prob, reward)}

    Returns:
        tuple: (V, policy) where
            V (dict): the optimal value function
            policy (dict): the optimal policy
    """
    all_states = env.S.union(env.terminal_states)

    V = {s: 0.0 for s in all_states}

    policy = {}
    for s in env.S:
        policy[s] = random.choice(list(env.A))
    for s in env.terminal_states:
        policy[s] = None

    iteration = 0

    while True:
        iteration += 1

        V = policy_evaluation(env, V, policy)

        policy, stable = policy_improvement(env, V, policy)

        print(f"Policy Iteration: {iteration}")

        save_value_heatmap(V, iteration, env)

        if stable:
            break

    return V, policy