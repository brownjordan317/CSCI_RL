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

    return grid


def save_value_heatmap(V, iteration, env, out_dir="heatmaps"):
    """
    Saves the value function V as a heatmap image to file.

    Args:
        V (dict): The value function
        iteration (int): The iteration number
        env (FiniteMDP): The finite MDP environment
        out_dir (str, optional): The output directory for the heatmap image. 
        Defaults to "heatmaps".

    Returns:
        None
    """
    grid = value_dict_to_grid(
        V,
        env.curr_grid.shape,
        env.terminal_states
    )

    vmin, vmax = env.terminal_state_values

    plt.figure(figsize=(6, 12))
    im = plt.imshow(
        grid,
        cmap="jet",
        origin="upper",
        vmin=int(vmin * 1.2),
        vmax=int(vmax * 1.2)
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
    Evaluates the value function for a given policy.

    Args:
        env (FiniteMDP): The finite MDP environment
        V (dict): The value function
        policy (dict): The policy
        tol (float, optional): The tolerance for the value function. 
        Defaults to 1e-4.

    Returns:
        dict: The updated value function
    """
    all_states = env.S.union(env.terminal_states)
    gamma = env.gamma

    while True:
        delta = 0
        V_new = V.copy()

        for s in all_states:
            if s in env.terminal_states:
                V_new[s] = 0.0
                continue

            a = policy[s]
            v_temp = 0.0

            for s_next, (prob, reward) in env.step(s, a).items():
                v_temp += prob * (reward + gamma * V[s_next])

            V_new[s] = v_temp
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < tol:
            break

    return V


def policy_improvement(env, V, policy):
    """
    Runs the policy improvement algorithm to find the optimal policy given a 
    value function.

    Args:
        env (FiniteMDP): a Finite Markov Reward Process
        V (dict): the value function
        policy (dict): the current policy

    Returns:
        tuple: (new_policy, policy_stable) where
            new_policy (dict): the improved policy
            policy_stable (bool): whether the policy has been improved
    """
    gamma = env.gamma
    policy_stable = True
    new_policy = policy.copy()

    for s in env.S:
        if s in env.terminal_states:
            # Skip terminal states
            continue

        old_action = policy[s]

        # Find the action that maximizes the expected reward
        new_policy[s] = max(
            env.A,
            key=lambda a: sum(
                prob * (reward + gamma * V[s_next])
                for s_next, (prob, reward) in env.step(s, a).items()
            )
        )

        if new_policy[s] != old_action:
            # If the policy has been improved, set policy_stable to False
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
