import numpy as np


class SarsaLambdaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, lam=0.8, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.E = np.zeros_like(self.Q)

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def reset_traces(self):
        self.E.fill(0)

    def update(self, s, a, r, s_next, a_next, done):
        delta = r + self.gamma * self.Q[s_next, a_next] * (not done) - self.Q[s, a]
        self.E[s, a] += 1

        self.Q += self.alpha * delta * self.E
        self.E *= self.gamma * self.lam