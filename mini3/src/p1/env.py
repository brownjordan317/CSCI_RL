import numpy as np


class ReactorEnv:
    """
    Controlled non-stationary bandit with hidden state mu.
    Observations are noisy samples z ~ N(mu, sigma^2).
    """

    def __init__(
        self,
        mu_min=0.0,
        mu_max=10.0,
        mu_hot=7.0,
        mu_lo=3.0,
        mu_hi=6.5,
        alpha=0.5,
        delta=0.3,
        sigma_obs=0.5,
        sigma_process=0.2,
        sigma_reward=0.2,
        rod_cost=0.05,
        meltdown_penalty=50,
        max_steps=200,
        k=2,
        n_bins=30,
    ):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_hot = mu_hot
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.alpha = alpha
        self.delta = delta
        self.sigma_obs = sigma_obs
        self.sigma_process = sigma_process
        self.sigma_reward = sigma_reward
        self.rod_cost = rod_cost
        self.meltdown_penalty = meltdown_penalty
        self.max_steps = max_steps

        self.actions = np.arange(-k, k + 1)
        self.n_actions = len(self.actions)

        self.n_bins = n_bins
        self.bin_edges = np.linspace(mu_min, mu_max, n_bins + 1)

        self.reset()

    # --------------------------
    # Core Environment
    # --------------------------

    def reset(self):
        self.mu = self.mu_min + 0.1
        self.t = 0
        return self._observe()

    def step(self, action_index):
        action = self.actions[action_index]

        drift = self.delta if self.mu >= self.mu_hot else 0.0
        noise = np.random.normal(0, self.sigma_process)

        self.mu = np.clip(
            self.mu - self.alpha * action + drift + noise,
            self.mu_min,
            self.mu_max,
        )

        reward_mean = self._reward_function(self.mu, action)
        reward = np.random.normal(reward_mean, self.sigma_reward)

        self.t += 1
        done = self.mu >= self.mu_max or self.t >= self.max_steps

        return self._observe(), reward, done

    # --------------------------
    # Helpers
    # --------------------------

    def _observe(self):
        z = np.random.normal(self.mu, self.sigma_obs)
        return self._discretize(z)

    def _discretize(self, z):
        idx = np.digitize(z, self.bin_edges) - 1
        return np.clip(idx, 0, self.n_bins - 1)

    def _reward_function(self, mu, action):
        if mu >= self.mu_max:
            return -self.meltdown_penalty

        if self.mu_lo <= mu <= self.mu_hi:
            return (mu - self.mu_lo) - self.rod_cost * abs(action)

        if mu < self.mu_lo:
            return -self.rod_cost * abs(action)

        return -self.meltdown_penalty