import numpy as np

def gridworld_from_mask(mask, num_actions=4):
    rows, cols = mask.shape

    # State space (exclude walls)
    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j])
    }

    # Reward function
    r = {
        (i, j): mask[i, j]
        for (i, j) in S
    }

    # Transition model
    P = {}

    if num_actions == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif num_actions == 8:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for (i, j) in S:
        neighbors = []

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (ni, nj) in S:
                neighbors.append((ni, nj))

        # Edge case: isolated cell → absorbing
        if not neighbors:
            P[(i, j)] = {(i, j): 1.0}
        else:
            prob = 1.0 / len(neighbors)
            P[(i, j)] = {n: prob for n in neighbors}

    return S, P, r


def gridworld_mdp_from_mask(mask, step_cost=-1.0):
    rows, cols = mask.shape

    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j])
    }

    A = {"U", "D", "L", "R"}

    moves = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1)
    }

    terminal_states = {
        (i, j) for (i, j) in S if mask[i, j] > 0
    }

    P = {}
    r = {}

    for s in S:
        P[s] = {}

        for a in A:
            # Terminal states are absorbing
            if s in terminal_states:
                P[s][a] = {s: 1.0}
                r[(s, a)] = 0.0
                continue

            i, j = s
            di, dj = moves[a]
            ni, nj = i + di, j + dj

            if (ni, nj) in S:
                s_next = (ni, nj)
            else:
                s_next = s

            P[s][a] = {s_next: 1.0}
            r[(s, a)] = step_cost + mask[s_next]

    return S, A, P, r, terminal_states

