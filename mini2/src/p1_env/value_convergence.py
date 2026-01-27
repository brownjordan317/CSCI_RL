

def value_iteration(S, A, P, r, terminal_states, gamma=0.99, tol=1e-4, max_iter=1000):
    """
    Perform Value Iteration on a stochastic MDP.

    Args:
        S: set of states
        A: set of actions
        P: dict of dict of dict, P[s][a][s_next] = probability
        r: dict of rewards, r[(s, a, s_next)]
        terminal_states: set of terminal states
        gamma: discount factor
        tol: convergence tolerance
        max_iter: maximum iterations

    Returns:
        V: dict of state values
        policy: dict of optimal action per state
    """
    V = {s: 0.0 for s in S}  # initialize values
    policy = {s: None for s in S}

    for it in range(max_iter):
        delta = 0
        V_new = V.copy()
        for s in S:
            if s in terminal_states:
                V_new[s] = 0.0
                continue

            q_values = []
            for a in A:
                q = 0
                for s_next, prob in P[s][a].items():
                    reward = r.get((s, a, s_next), 0.0)
                    q += prob * (reward + gamma * V[s_next] if s_next not in terminal_states else reward)
                q_values.append((q, a))
            
            best_q, best_a = max(q_values, key=lambda x: x[0])
            V_new[s] = best_q
            policy[s] = best_a
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < tol:
            break

    return V, policy
