import numpy as np



def value_iteration_vectorized(mrp, epsilon=1e-6, max_iters=10000):
    """
    Runs the value iteration algorithm to find the converged value function.

    Args:
        mrp (FiniteMRP): a Finite Markov Reward Process
        epsilon (float): tolerance for convergence
        max_iters (int): maximum number of iterations

    Returns:
        tuple: (V, iters) where
            V (dict): the converged value function
            iters (int): the number of iterations it took to converge

    Raises:
        RuntimeError: if the value iteration algorithm did not converge
    """
    P, r, S = mrp.P_matrix, mrp.r_vector, list(mrp.states())
    gamma = mrp.discount()

    V = np.zeros(len(S))

    for k in range(max_iters):
        V_new = r + gamma * P @ V

        if np.max(np.abs(V_new - V)) < epsilon:
            return dict(zip(S, V_new)), k + 1

        V = V_new

    raise RuntimeError("Value iteration did not converge")


def value_iteration_mdp(mdp, epsilon=1e-6, max_iters=10000):
    V = {s: 0.0 for s in mdp.states()}

    for k in range(max_iters):
        delta = 0.0
        V_new = {}

        for s in mdp.states():
            values = []
            for a in mdp.actions(s):
                q = mdp.reward(s, a)
                q += mdp.discount() * sum(
                    mdp.transition_prob(s, a, s_next) * V[s_next]
                    for s_next in mdp.states()
                )
                values.append(q)

            V_new[s] = max(values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < epsilon:
            return V, k + 1

    raise RuntimeError("Value iteration did not converge")
