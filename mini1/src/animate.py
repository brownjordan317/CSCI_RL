import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def animate_policy(start, mdp, policy, mask, terminal_states, max_steps=100):
    trajectory = [start]
    s = start

    for _ in range(max_steps):
        if s in terminal_states:
            break

        a = policy[s]
        s = list(mdp.P[s][a].keys())[0]
        trajectory.append(s)

    grid = np.copy(mask)
    grid[np.isneginf(grid)] = -10

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="viridis")
    agent = ax.scatter([], [], c="red", s=200)

    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        s = trajectory[frame]
        agent.set_offsets([[s[1], s[0]]])
        ax.set_title(f"Step {frame}")
        return agent,

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=500)
    plt.close(fig)
    return ani
