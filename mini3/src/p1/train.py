import numpy as np
import matplotlib.pyplot as plt
from env import ReactorEnv
from sarsa import SarsaLambdaAgent
from q_learning import QLearningAgent


# =========================================================
# Utility Functions
# =========================================================

def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")


def convergence_episode(returns, window=20):
    ma = moving_average(returns, window)
    final_avg = np.mean(returns[-50:])
    threshold = 0.05 * abs(final_avg)

    for i in range(len(ma)):
        if abs(ma[i] - final_avg) < threshold:
            return i
    return len(returns)


def critical_region_analysis(Q, env):
    n_states = env.n_bins
    high_bins = range(int(0.8 * n_states), n_states)

    mid = len(env.actions) // 2
    withdraw_idx = list(range(0, mid))
    insert_idx = list(range(mid + 1, len(env.actions)))

    risky_vals = []
    safe_vals = []

    for s in high_bins:
        risky_vals.append(np.mean(Q[s, withdraw_idx]))
        safe_vals.append(np.mean(Q[s, insert_idx]))

    return np.mean(risky_vals), np.mean(safe_vals)


def plot_learning_curves(results, title):
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(moving_average(data["returns"], 20), label=label)

    plt.xlabel("Episode")
    plt.ylabel("Return (20-ep MA)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()


def plot_heatmap(Q, env, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(Q.T, aspect="auto", origin="lower")
    plt.colorbar(label="Q-value")
    plt.yticks(np.arange(len(env.actions)), env.actions)
    plt.xlabel("State (z bin)")
    plt.ylabel("Action")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()


# =========================================================
# Training
# =========================================================

def train(env, agent, episodes=500):
    returns = []
    meltdowns = 0

    for ep in range(episodes):
        s = env.reset()
        total_reward = 0

        if hasattr(agent, "reset_traces"):
            agent.reset_traces()

        a = agent.select_action(s)
        done = False

        while not done:
            s_next, r, done = env.step(a)
            total_reward += r

            if isinstance(agent, SarsaLambdaAgent):
                a_next = agent.select_action(s_next)
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next
            else:
                agent.update(s, a, r, s_next, done)
                s = s_next
                a = agent.select_action(s)

        if env.mu >= env.mu_max:
            meltdowns += 1

        returns.append(total_reward)

    return {
        "returns": returns,
        "meltdowns": meltdowns,
        "Q": agent.Q
    }


# =========================================================
# Experiments
# =========================================================

def run_experiments():
    episodes = 500
    noise_levels = {
        "Low Noise (σ=0.0)": 0.0,
        "High Noise (σ=0.8)": 0.8
    }

    for noise_label, sigma in noise_levels.items():
        print("\n====================================================")
        print(f"Running experiments for {noise_label}")
        print("====================================================")

        env = ReactorEnv(sigma_obs=sigma)

        # SARSA λ=0.8
        sarsa = SarsaLambdaAgent(env.n_bins, env.n_actions, lam=0.8)
        res_sarsa = train(env, sarsa, episodes)

        # Q-learning
        qlearn = QLearningAgent(env.n_bins, env.n_actions)
        res_q = train(env, qlearn, episodes)

        # -------------------------------------------------
        # (a) Learning Curves
        # -------------------------------------------------
        plot_learning_curves(
            {
                "SARSA(λ=0.8)": res_sarsa,
                "Q-learning": res_q
            },
            f"Learning Curves - {noise_label}"
        )

        # -------------------------------------------------
        # (b) Q-Heatmaps + Risk Analysis
        # -------------------------------------------------
        plot_heatmap(res_sarsa["Q"], env, f"SARSA Q Heatmap - {noise_label}")
        plot_heatmap(res_q["Q"], env, f"Q-Learning Q Heatmap - {noise_label}")

        risky_sarsa, safe_sarsa = critical_region_analysis(res_sarsa["Q"], env)
        risky_q, safe_q = critical_region_analysis(res_q["Q"], env)

        print("\n--- Q-Function Critical Region Analysis ---")
        print("SARSA:")
        print(f"Avg Q(high_z, withdraw) = {risky_sarsa:.3f}")
        print(f"Avg Q(high_z, insert)   = {safe_sarsa:.3f}")

        print("Q-learning:")
        print(f"Avg Q(high_z, withdraw) = {risky_q:.3f}")
        print(f"Avg Q(high_z, insert)   = {safe_q:.3f}")

        if risky_sarsa < safe_sarsa:
            print("→ SARSA penalizes risky actions near critical region.")
        if risky_q < safe_q:
            print("→ Q-learning penalizes risky actions near critical region.")

        # -------------------------------------------------
        # (c) Direct Comparison
        # -------------------------------------------------
        conv_sarsa = convergence_episode(res_sarsa["returns"])
        conv_q = convergence_episode(res_q["returns"])

        final_sarsa = np.mean(res_sarsa["returns"][-50:])
        final_q = np.mean(res_q["returns"][-50:])

        print("\n--- Algorithm Comparison ---")
        print(f"SARSA convergence episode: {conv_sarsa}")
        print(f"Q-learning convergence episode: {conv_q}")

        if conv_q < conv_sarsa:
            print("→ Q-learning converges faster.")
        else:
            print("→ SARSA converges faster.")

        print(f"\nSARSA final avg return: {final_sarsa:.2f}")
        print(f"Q-learning final avg return: {final_q:.2f}")

        if final_q > final_sarsa:
            print("→ Q-learning achieves higher return.")
        else:
            print("→ SARSA achieves higher return.")

        print(f"\nSARSA meltdowns: {res_sarsa['meltdowns']}")
        print(f"Q-learning meltdowns: {res_q['meltdowns']}")

        if res_q["meltdowns"] > res_sarsa["meltdowns"]:
            print("→ Q-learning is more prone to meltdown during exploration.")
        else:
            print("→ SARSA is more prone to meltdown during exploration.")


if __name__ == "__main__":
    run_experiments()