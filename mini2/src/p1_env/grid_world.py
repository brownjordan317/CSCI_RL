import numpy as np

def gridworld_mdp_from_mask(mask, step_cost=-1.0):
    rows, cols = mask.shape

    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j])
    }

    A = {"forward", "left", "right"}

    moves = {
        "forward": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
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

