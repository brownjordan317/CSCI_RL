def value_iteration(S, A, P, r, terminal_states,
                            gamma=0.99, tol=1e-4, max_iter=1000):
    """
    Value iteration algorithm to compute the value function and
    policy for a given finite Markov Decision Process (MDP).

    Parameters
    ----------
    S : set
        Set of states
    A : set
        Set of actions
    P : dict
        Transition probability matrix
    r : dict
        Reward function
    terminal_states : set
        Set of terminal states
    gamma : float, optional
        Discount factor in [0,1]. Defaults to 0.99.
    tol : float, optional
        Tolerance for convergence. Defaults to 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1000.

    Returns
    -------
    V : dict
        Value function
    policy : dict
        Greedy policy
    """

    # Include both normal and terminal states
    all_states = S.union(terminal_states)

    V = {s: 0.0 for s in all_states}

    # Main value iteration loop
    for _ in range(max_iter):
        delta = 0.0
        V_new = {}

        # Perform a Bellman optimality update for each state
        for s in all_states:

            # Terminal states are absorbing with zero value
            if s in terminal_states:
                V_new[s] = 0.0
                continue

            # -------- Bellman optimality backup --------
            # For the current state s:
            #   1. Compute the expected return for each action a
            #   2. Select the action that maximizes this expected return
            #
            # This implements:
            #   V_{k+1}(s) = max_a E[ R(s,a,s') + γ V_k(s') ]
            V_new[s] = max(
                sum(
                    prob * (
                        r.get((s, a, s_next), 0.0)
                        + gamma * V[s_next]  # discounted value of next state
                    )
                    for s_next, prob in P[s][a].items()
                )
                for a in A
            )
            # -------------------------------------------

            # Track the maximum change in value for convergence checking
            delta = max(delta, abs(V_new[s] - V[s]))

        # Update the value function
        V = V_new

        # Stop if the Bellman updates have converged
        if delta < tol:
            break

    # -------- Policy extraction (greedy w.r.t. V*) --------
    # After convergence, derive the optimal policy by choosing,
    # for each state, the action that maximizes the Bellman expectation
    policy = {}
    for s in S:
        if s in terminal_states:
            policy[s] = None
            continue

        policy[s] = max(
            A,
            key=lambda a: sum(
                prob * (
                    r.get((s, a, s_next), 0.0)
                    + gamma * V[s_next]
                )
                for s_next, prob in P[s][a].items()
            )
        )
    # ------------------------------------------------------

    return V, policy