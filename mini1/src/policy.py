import numpy as np

def greedy_policy(mdp, V):
    policy = {}

    for s in mdp.states():
        best_a = None
        best_v = -np.inf

        for a in mdp.actions(s):
            q = mdp.reward(s, a)
            q += mdp.discount() * sum(
                mdp.transition_prob(s, a, s_next) * V[s_next]
                for s_next in mdp.states()
            )

            if q > best_v:
                best_v = q
                best_a = a

        policy[s] = best_a

    return policy
