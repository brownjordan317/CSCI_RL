from typing import Dict, Set
import numpy as np

class FiniteMRP:
    def __init__(
        self,
        S: Set,
        P: Dict,
        r: Dict,
        gamma: float
    ):
        """
        S: set of states
        P[s][s']: transition probability 
        r[s]: expected reward 
        gamma: discount factor in [0,1]
        """

        self.S = S
        self.P = P
        self.r = r
        self.gamma = gamma

        self.validate()
        self.build_mrp_matrices(self)

    def validate(self):
        """
        Validate the FiniteMRP instance.

        Raises ValueError if:
        - gamma is not in [0,1]
        - transition model missing for any state
        - transition probabilities do not sum to 1
        - transitions go to unknown states
        - reward missing for any state
        """

        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("Discount factor gamma must be in [0,1]")

        for s in self.S:
            if s not in self.P:
                raise ValueError(f"Missing transition model for state {s}")

            probs = self.P[s]
            total_prob = sum(probs.values())

            if abs(total_prob - 1.0) > 1e-6:
                raise ValueError(
                    f"Sum of transition probabilities for state {s} ≠ 1"
                )

            for s_next in probs:
                if s_next not in self.S:
                    raise ValueError(
                        f"Transition to unknown state {s_next} from {s}"
                    )

            if s not in self.r:
                raise ValueError(f"Missing reward for state {s}")
            
    def build_mrp_matrices(self, mrp):
        """
        Builds the transition probability matrix P, reward vector r, and 
        state vector s corresponding to the given Finite Markov Reward Process 
        (MRP).

        Args:
            mrp (FiniteMRP): a Finite Markov Reward Process

        Returns:
            tuple: (P, r, states) where
                P (numpy.array): transition probability matrix
                r (numpy.array): reward vector
                states (list): list of states
        """
        states = list(self.states())
        idx = {s: i for i, s in enumerate(states)}

        n = len(states)
        P = np.zeros((n, n))
        r = np.zeros(n)

        for s in states:
            i = idx[s]
            r[i] = mrp.reward(s)
            for s_next in states:
                j = idx[s_next]
                P[i, j] = mrp.transition_prob(s, s_next)

        self.P_matrix = P
        self.r_vector = r

    def states(self):
        """
        Returns the set of states for the FiniteMRP.

        Returns:
        Set: a set of states
        """
        return self.S

    def transition_prob(self, state, next_state):
        """
        Returns the transition probability from state to next_state.

        Args:
            state (str): current state
            next_state (str): next state

        Returns:
            float: transition probability
        """
        return self.P.get(state, {}).get(next_state, 0.0)

    def reward(self, state):
        """
        Returns the reward for the given state.

        Args:
            state (str): state for which to retrieve the reward

        Returns:
            float: reward for the given state
        """
        return self.r[state]

    def discount(self):
        """
        Returns the discount factor for the FiniteMRP.

        Returns:
            float: discount factor in [0,1]
        """
        return self.gamma
