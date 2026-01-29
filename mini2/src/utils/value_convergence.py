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
            # Unpack the tuple
            r, c = s
            # Assign the value to the grid
            grid[r, c] = float(v)

    # grid = np.round(grid, decimals=2)
    # print(grid)

    return grid


def save_value_heatmap(V, iteration, env, out_dir="heatmaps"):
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
    gamma = env.gamma
    while True:
        delta = 0
        # UPDATE IN-PLACE: Remove V_new = V.copy()
        for s in env.S:
            old_v = V[s]
            a = policy[s]
            
            # Use the summation of the Bellman Equation
            V[s] = sum(prob * (reward + gamma * V[s_next]) 
                       for s_next, (prob, reward) in env.step(s, a).items())
            
            delta = max(delta, abs(old_v - V[s]))

        if delta < tol:
            break
    return V

def policy_improvement(env, V, policy):
    gamma = env.gamma
    policy_stable = True
    new_policy = policy.copy()

    for s in env.S:
        old_action = policy[s]
        
        # Calculate Q(s, a) for all possible actions
        action_values = {}
        for a in env.A:
            action_values[a] = sum(prob * (reward + gamma * V[s_next]) 
                                   for s_next, (prob, reward) in env.step(s, a).items())
        
        # Select the best action
        best_action = max(action_values, key=action_values.get)
        new_policy[s] = best_action

        # In policy_improvement
        if old_action != best_action:
            # Use a small epsilon to ensure the improvement is significant
            if action_values[best_action] > action_values[old_action] + 1e-9:
                policy_stable = False

    return new_policy, policy_stable


def policy_iteration(env):
    """
    Runs the policy iteration algorithm to find the optimal policy and 
    value function.

    Args:
        env (FiniteMDP): a Finite Markov Reward Process

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
