from typing import Dict, Set, Tuple
import numpy as np


class FiniteMDP:
    def __init__(
        self,
        S: Set,
        A: Set,
        P: Dict,     # P[s][a][s'] = Pr(s' | s, a)
        r: Dict,     # r[s][a] = reward
        gamma: float
    ):
        self.S = S
        self.A = A
        self.P = P
        self.r = r
        self.gamma = gamma

        self.validate()

    def validate(self):
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0,1]")

        for s in self.S:
            if s not in self.P:
                raise ValueError(f"Missing transitions for state {s}")

            for a in self.A:
                if a not in self.P[s]:
                    raise ValueError(f"Missing action {a} in state {s}")

                probs = self.P[s][a]
                if abs(sum(probs.values()) - 1.0) > 1e-6:
                    raise ValueError(f"Probabilities for ({s},{a}) ≠ 1")

                for s_next in probs:
                    if s_next not in self.S:
                        raise ValueError(f"Unknown next state {s_next}")

                if (s, a) not in self.r:
                    raise ValueError(f"Missing reward for ({s},{a})")

    def states(self):
        return self.S

    def actions(self, state):
        return self.A

    def transition_prob(self, s, a, s_next):
        return self.P[s][a].get(s_next, 0.0)

    def reward(self, s, a):
        return self.r[(s, a)]

    def discount(self):
        return self.gamma
