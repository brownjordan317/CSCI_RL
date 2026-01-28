import numpy as np


def import_random_transitions(transitions_type):
    """
    Imports a random transition model based on the given type.

    Parameters:
    transitions_type (str): Type of transitions to import (e.g. "m2p1" for
        windy grid world transitions).

    Returns:
    random_transitions (function): Imported random transition model.

    Raises:
    ValueError: If the given transitions type is unknown.
    """
    if transitions_type == "m2p1":
        from src.p1.helpers import wind_transitions as random_transitions
    else:
        raise ValueError(f"Unknown transitions type: {transitions_type}")

    return random_transitions


import numpy as np

def gridworld(mask, terminal_map, action_map, step_penalty=-1, 
              transitions_type="m2p1", random_probability=0.8):
    """
    Constructs a GridWorld MDP model from a given mask, terminal map, 
    action map, step penalty, transitions type, and random probability.

    Parameters:
    mask (numpy array): Grid world mask, where -1 represents walls and 
        positive values represent terminal states.
    terminal_map (dict): Dictionary mapping terminal states to their 
        corresponding rewards.
    action_map (dict): Dictionary mapping actions to their corresponding
        movements (e.g. {"U": (-1, 0)}).
    step_penalty (float): Reward penalty for moving from one state to another.
    transitions_type (str): Type of transitions to use (e.g. "m2p1" for windy 
        grid world transitions).
    random_probability (float): Probability of a random transition occurring.

    Returns:
    S (set): Set of all non-terminal states in the grid world.
    A (set): Set of all actions in the grid world.
    P (dict): Transition model, mapping each state-action pair to a 
        probability distribution over next states.
    r (dict): Reward model, mapping each state-action-next state pair to its 
        corresponding reward.
    terminal_states (set): Set of all terminal states in the grid world.
    """
    random_transitions = import_random_transitions(transitions_type)

    rows, cols = mask.shape
    center_col = cols // 2


    # Terminal bookkeeping
    terminal_states = set(terminal_map.keys())
    terminal_reward_values = set(terminal_map.values())

    # State space (grid states only)
    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j]) and mask[i, j] not in terminal_reward_values
    }

    # Action space
    A = set(action_map.keys())

    moves = action_map

    # Transition + reward models
    P = {}
    r = {}

    # Transitions for grid states
    for s in S:
        P[s] = {}

        for a in A:
            raw_transitions = random_transitions(
                *s,
                a,
                moves,
                random_probability,
                rows,
                cols,
                center_col
            )

            P[s][a] = {}

            for s_next, prob in raw_transitions.items():
                actual_next = s_next

                # Check if landing on terminal cell
                if isinstance(s_next, tuple):
                    cell_val = mask[s_next]

                    if cell_val in terminal_reward_values:
                        # Find ALL terminals with this reward
                        matching_terminals = [
                            t for t, v in terminal_map.items()
                            if v == cell_val
                        ]

                        # Deterministic selection (safe)
                        actual_next = matching_terminals[0]

                # Accumulate probability mass
                P[s][a][actual_next] = P[s][a].get(actual_next, 0.0) + prob

                # Reward assignment
                if actual_next in terminal_states:
                    r[(s, a, actual_next)] = terminal_map[actual_next]
                else:
                    r[(s, a, actual_next)] = step_penalty

    # Absorbing terminal states
    for term in terminal_states:
        P[term] = {}
        for a in A:
            P[term][a] = {term: 1.0}
            r[(term, a, term)] = 0.0

    return S, A, P, r, terminal_states
