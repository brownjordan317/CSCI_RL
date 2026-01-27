import os
os.makedirs("heatmaps", exist_ok=True)


def bellman_expectation(env, V, s, a, gamma):
    return sum(
        prob * (reward + gamma * V[s_next])
        for s_next, (prob, reward) in env.step(s, a).items()
    )


def bellman_backup(env, V, s, A, gamma):
    values = []
    for a in A:
        val = bellman_expectation(env, V, s, a, gamma)
        values.append(val)
    max_val = max(values)
    return max_val

def bellman_sweep(env, V, S, A, terminal_states, gamma):
    V_new = {}
    delta = 0.0

    for s in S:
        if s in terminal_states:
            V_new[s] = 0.0
            continue

        V_new[s] = bellman_backup(env, V, s, A, gamma)
        delta = max(delta, abs(V_new[s] - V[s]))

    return V_new, delta

def extract_greedy_policy(env, V, S, A, terminal_states, gamma):
    policy = {}

    for s in S:
        if s in terminal_states:
            policy[s] = None
            continue

        policy[s] = max(
            A,
            key=lambda a: bellman_expectation(env, V, s, a, gamma)
        )

    return policy


import numpy as np

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


import matplotlib.pyplot as plt

def save_value_heatmap(V, iteration, grid_shape, terminal_states, out_dir="heatmaps"):
    grid = value_dict_to_grid(V, grid_shape, terminal_states)

    plt.figure(figsize=(6, 12))
    im = plt.imshow(grid, cmap="jet", origin="upper", vmin=-25, vmax=0)
    plt.colorbar(im)
    plt.title(f"Value Function – Iteration {iteration}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/V_iter_{iteration:04d}.png")
    plt.close()



def value_iteration(
    S, A, env, terminal_states,
    gamma=0.99, tol=1e-4, max_iter=1000
):
    all_states = S.union(terminal_states)
    V = {s: 0.0 for s in all_states}

    grid_shape = env.curr_grid.shape
    iteration = 1
    for _ in range(max_iter):
        V_new, delta = bellman_sweep(
            env, V, all_states, A, terminal_states, gamma
        )
        V = V_new.copy()

        if V == V_new:
            print(f"same_{iteration}")

        iteration += 1
        save_value_heatmap(
            V,
            iteration,
            grid_shape,
            terminal_states
        )

        if delta < tol:
            break

    policy = extract_greedy_policy(
        env, V, S, A, terminal_states, gamma
    )

    print(f"num iters = {iteration}")
    return V, policy
