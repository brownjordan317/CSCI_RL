import os
import numpy as np
import matplotlib.pyplot as plt
import random

os.makedirs("heatmaps", exist_ok=True)

def value_dict_to_grid(V, grid_shape, terminal_states):
    """
    Converts V[(row, col)] -> 2D array for heatmap plotting
    """
    grid = np.full(grid_shape, np.nan)

    for s, v in V.items():
        if s in terminal_states:
            continue
        if isinstance(s, tuple):
            r, c = s
            grid[r, c] = float(v)

    return grid


def save_value_heatmap(
        V, 
        iteration, 
        grid_shape, 
        terminal_states, 
        terminal_values, 
        out_dir="heatmaps"
    ):
    grid = value_dict_to_grid(V, grid_shape, terminal_states)

    plt.figure(figsize=(6, 12))
    im = plt.imshow(
        grid, 
        cmap="jet", 
        origin="upper", 
        vmin=int(terminal_values[0] * 1.2), 
        vmax=int(terminal_values[1] * 1.2)
    )
    plt.colorbar(im)
    plt.title(f"Value Function – Iteration {iteration}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/V_iter_{iteration:04d}.png")
    plt.close()

def policy_evaluation(env, V, policy, all_states, gamma, tol=1e-4):
    while True:
        delta = 0
        V_new = V.copy()
        for s in all_states:
            if s in env.terminal_states:
                V_new[s] = 0.0 # Terminal states have no future value
                continue
            
            a = policy[s]
            v_temp = 0
            for s_next, (prob, reward) in env.step(s, a).items():
                # V[s_next] will be 0 if s_next is terminal, 
                # so only the 'reward' will count.
                v_temp += prob * (reward + gamma * V[s_next])
            
            V_new[s] = v_temp
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < tol:
            break
    return V

def policy_improvement(env, V, S, A, gamma, policy):
    policy_stable = True
    new_policy = policy.copy()

    for s in S:
        if s in env.terminal_states:
            continue

        old_action = policy[s]
        
        # Optimization step: find the action that maximizes expected return
        new_policy[s] = max(
            A,
            key=lambda a: sum(
                prob * (reward + gamma * V[s_next])
                for s_next, (prob, reward) in env.step(s, a).items()
            )
        )

        if old_action != new_policy[s]:
            policy_stable = False
            
    return new_policy, policy_stable

def policy_iteration(S, A, env, terminal_states, terminal_values, gamma=0.99):
    all_states = S.union(terminal_states)
    V = {s: 0.0 for s in all_states}
    
    # Define center-seeking policy
    policy = {}
    for s in S:
        policy[s] = random.choice(list(A)) # A must be a list or tuple of available actions

    for s in terminal_states:
        policy[s] = None

    iteration = 0
    while True:
        iteration += 1

        V = policy_evaluation(env, V, policy, all_states, gamma)
        
        policy, policy_stable = policy_improvement(env, V, S, A, gamma, policy)
        
        print(f"Policy Iteration: {iteration}")
        
        # Visualizing the value function at each policy change
        save_value_heatmap(
            V, 
            iteration, 
            env.curr_grid.shape, 
            terminal_states, 
            terminal_values
        )

        if policy_stable:
            break

    return V, policy