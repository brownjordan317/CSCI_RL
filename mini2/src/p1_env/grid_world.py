import numpy as np

def windy_gridworld(mask, step_cost=-1.0, p_center=0.8):
    """
    Creates a Markov decision process (MDP) from a grid world with windy transitions.

    Parameters
    ----------
    mask : numpy array
        A 2D array where each cell represents a state in the grid world.
        A value of -np.inf represents a wall, and a value greater than 0 represents a reward.
    step_cost : float, optional
        The cost of taking each step in the grid world. Defaults to -1.0.
    p_center : float, optional
        The probability of being pushed towards the center column when on the second column from the left or right. Defaults to 0.8.

    Returns
    -------
    S : set
        The set of all states in the MDP.
    A : set
        The set of all actions in the MDP.
    P : dict
        The transition model of the MDP.
    r : dict
        The reward model of the MDP.
    terminal_states : set
        The set of all terminal states in the MDP.

    """

    
    rows, cols = mask.shape

    # State space
    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j])
    }

    # Action space
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

            # Stochastic transitions with wind
            transitions = wind_transitions(*s, a, moves, p_center, rows, cols)
            P[s][a] = {}
            for s_next, prob in transitions.items():
                P[s][a][s_next] = prob
                # Reward: step cost + mask reward if it's a grid cell
                if isinstance(s_next, tuple):
                    r[(s, a, s_next)] = step_cost + mask[s_next]
                else:
                    # Crash or exit
                    r[(s, a, s_next)] = 0.0

    # Add crash and exit as terminal states
    terminal_states = terminal_states.union({'crash', 'exit'})

    return S, A, P, r, terminal_states


def wind_transitions(i, j, a, moves, p_center, rows, cols):
    """
    Compute wind-based transitions for a given state and action.

    Given a state (i, j) and an action a, compute the probability of
    transitioning to each possible next state, taking into account the
    effects of wind.

    Parameters
    ----------
    i : int
        Row coordinate of current state
    j : int
        Column coordinate of current state
    a : str
        Action taken (one of 'forward', 'left', 'right')

    Returns
    -------
    transitions : dict
        A dictionary mapping each possible next state to its probability
    """

    # Deterministic move
    di, dj = moves[a]
    ni, nj = i + di, j + dj

    # Initialize
    transitions = {}

    # Compute distance-based wind probability
    B = p_center
    E = 1 / (1 + (nj - 3)**2)
    p = B * E
    p2 = 0.5  # second-layer wind probability

    if nj == 3:
        # Center column
        transitions[(ni, 2)] = p / 2
        transitions[(ni, 4)] = p / 2
        transitions[(ni, 1)] = (1 - p) * p2 / 2
        transitions[(ni, 5)] = (1 - p) * p2 / 2
        transitions[(ni, 3)] = (1 - p) * (1 - p2)
    else:
        # Off-center: wind pushes toward center
        left_prob = p if nj > 3 else 0
        right_prob = p if nj < 3 else 0
        transitions[(ni, nj)] = 1 - left_prob - right_prob
        if left_prob > 0:
            transitions[(ni, nj - 1)] = left_prob
        if right_prob > 0:
            transitions[(ni, nj + 1)] = right_prob

    # Handle crashes (outside grid)
    final_transitions = {}
    for (i_t, j_t), prob in transitions.items():
        if j_t < 0 or j_t >= cols:
            final_transitions['crash'] = final_transitions.get('crash', 0) + prob
        elif i_t >= rows:
            final_transitions['exit'] = final_transitions.get('exit', 0) + prob
        else:
            final_transitions[(i_t, j_t)] = prob

    return final_transitions