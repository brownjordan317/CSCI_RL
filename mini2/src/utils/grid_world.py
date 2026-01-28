import numpy as np


def import_random_transitions(transitions_type):
    if transitions_type == "m2p1":
        from src.p1.helpers import wind_transitions as random_transitions

    return random_transitions

def gridworld(mask, terminal_values, value_map=None, transitions_type="m2p1", random_probability=0.8):
    """
    Creates an MDP where terminal states are defined by values in the mask.
    """
    if value_map is None:
        # Default mapping based on your current generate_grid values
        value_map = {-20: 'crash', 100: 'exit'}

    random_transitions = import_random_transitions(transitions_type)

    rows, cols = mask.shape
    fail_value, terminal_value = terminal_values
    center_col = 3 # Can also be made dynamic (cols // 2)

    # State space: Any cell that is NOT -inf and NOT a terminal value in the map
    # This ensures terminal states don't have their own 'next moves' calculated here
    terminal_vals_set = set(value_map.keys())
    S = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if not np.isneginf(mask[i, j]) and mask[i, j] not in terminal_vals_set
    }

    A = {"forward", "left", "right"}
    moves = {"forward": (1, 0), "left": (0, -1), "right": (0, 1)}

    # Extract the symbolic names from your value_map (e.g., {'crash', 'exit'})
    terminal_states = set(value_map.values())

    P = {}
    r = {}

    for s in S:
        P[s] = {}
        for a in A:
            # 1. Get raw transitions from the helper (e.g., wind_transitions)
            raw_transitions = random_transitions(*s, a, moves, random_probability, rows, cols, center_col)
            P[s][a] = {}
            
            for s_next, prob in raw_transitions.items():
                # 2. Resolve the "Actual" next state based on the mask
                actual_next = s_next
                
                if isinstance(s_next, tuple):
                    # Check if the coordinate landed on a special value in the mask
                    cell_val = mask[s_next]
                    if cell_val in value_map:
                        actual_next = value_map[cell_val]

                # 3. Accumulate probabilities and assign rewards
                P[s][a][actual_next] = P[s][a].get(actual_next, 0) + prob
                
                if actual_next == 'crash':
                    r[(s, a, actual_next)] = fail_value
                elif actual_next == 'exit':
                    r[(s, a, actual_next)] = terminal_value
                else:
                    r[(s, a, actual_next)] = -1 # Standard step cost

    # Add absorbing transitions for all unique terminal symbols found in value_map
    for term in terminal_states:
        P[term] = {a: {term: 1.0} for a in A}
        for a in A:
            r[(term, a, term)] = 0.0

    return S, A, P, r, terminal_states


