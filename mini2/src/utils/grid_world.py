import numpy as np


def import_random_transitions(transitions_type):
    """Import random transition model based on type"""
    if transitions_type == "m2p1":
        from src.p1.helpers import wind_transitions as random_transitions
    elif transitions_type == "m2p2":
        from src.p2.helpers import movement_drift_transitions as random_transitions
    elif transitions_type == "circuit":
        return None
    else:
        raise ValueError(f"Unknown transitions type: {transitions_type}")
    return random_transitions


def gridworld(mask=None, terminal_map=None, action_map=None, step_penalty=-1, 
              transitions_type="m2p1", random_probability=0.8,
              grid_size=3, correct_reward=100, incorrect_penalty=-50):
    """
    Constructs a GridWorld MDP model.
    
    Modes:
    - Standard: requires mask, terminal_map, action_map
    - Circuit: set transitions_type="circuit"
    """
    
    # Circuit mode
    if transitions_type == "circuit":
        from src.p3.helpers import circuit_gridworld
        return circuit_gridworld(action_map, grid_size, step_penalty, 
                                 correct_reward, incorrect_penalty)
    
    # Standard mode - validation
    if mask is None or terminal_map is None or action_map is None:
        raise ValueError("Standard mode requires mask, terminal_map, and action_map")
    
    random_transitions = import_random_transitions(transitions_type)
    rows, cols = mask.shape
    center = (cols // 2) if transitions_type == "m2p1" else (rows // 2)

    # Terminal setup
    terminal_states = set(terminal_map.keys())
    terminal_reward_values = set(terminal_map.values())

    # State space
    S = {
        (i, j) for i in range(rows) for j in range(cols)
        if not np.isneginf(mask[i, j]) and mask[i, j] not in terminal_reward_values
    }

    A = set(action_map.keys())
    P = {}
    r = {}

    # Build transitions
    for s in S:
        P[s] = {}
        for a in A:
            raw_transitions = random_transitions(
                *s, a, action_map, random_probability, rows, cols, center
            )

            P[s][a] = {}
            for s_next, prob in raw_transitions.items():
                actual_next = s_next

                # Check terminal landing
                if isinstance(s_next, tuple) and mask[s_next] in terminal_reward_values:
                    matching_terminals = [t for t, v in terminal_map.items() if v == mask[s_next]]
                    actual_next = matching_terminals[0]

                P[s][a][actual_next] = P[s][a].get(actual_next, 0.0) + prob
                r[(s, a, actual_next)] = (terminal_map[actual_next] if actual_next in terminal_states 
                                          else step_penalty)

    # Absorbing terminals
    for term in terminal_states:
        P[term] = {a: {term: 1.0} for a in A}
        for a in A:
            r[(term, a, term)] = 0.0

    return S, A, P, r, terminal_states