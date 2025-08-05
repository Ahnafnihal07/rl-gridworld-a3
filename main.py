"""
main.py
Reproducible training and figure generation for SARSA and Q-learning.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from gridworld_env import GridWorld
from sarsa import sarsa
from q_learning import q_learning
from utils import greedy_policy_from_Q, rollout_policy, to_arrows

def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w-1, ma[0])
    return np.concatenate([pad, ma])

def plot_rewards_with_ma(arr, title, path):
    plt.figure()
    plt.plot(arr, label="Episode return")
    plt.plot(moving_average(arr, 50), label="Moving avg (w=50)")
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def visualize_policy_arrows(env, policy, path):
    grid = env.layout()
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='tab20')
    for r in range(n_rows):
        for c in range(n_cols):
            s = (r, c)
            if env.is_terminal(s):
                ax.text(c, r, 'T', ha='center', va='center', fontsize=12, fontweight='bold')
            elif s == env.start:
                ax.text(c, r, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
            elif s in getattr(env, 'red_cells', set()):
                ax.text(c, r, 'R', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                a = policy.get(s, None)
                if a is not None:
                    ax.text(c, r, to_arrows(a), ha='center', va='center', fontsize=12)
    ax.set_title("Greedy policy arrows")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xlim(-0.5, n_cols-0.5)
    ax.set_ylim(n_rows-0.5, -0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def visualize_trajectory(env, traj, path):
    visit = -np.ones((env.n_rows, env.n_cols), dtype=float)
    for i, s in enumerate(traj):
        r, c = s
        visit[r, c] = i
    plt.figure()
    im = plt.imshow(visit, interpolation='nearest')
    plt.colorbar(im, label="Visit step")
    plt.title("Trajectory visit order (lower is earlier)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main(output_dir="../report/figures"):
    os.makedirs(output_dir, exist_ok=True)
    env = GridWorld()

    # Train SARSA
    Q_s, r_s = sarsa(env, episodes=1500, alpha=0.1, gamma=0.99,
                     epsilon=0.3, epsilon_decay=0.999, min_epsilon=0.05, seed=7)
    pol_s = greedy_policy_from_Q(Q_s, env)
    traj_s, _, _ = rollout_policy(GridWorld(), pol_s, max_steps=500)

    # Train Q-learning
    env2 = GridWorld()
    Q_q, r_q = q_learning(env2, episodes=1500, alpha=0.1, gamma=0.99,
                          epsilon=0.3, epsilon_decay=0.999, min_epsilon=0.05, seed=9)
    pol_q = greedy_policy_from_Q(Q_q, env2)
    traj_q, _, _ = rollout_policy(GridWorld(), pol_q, max_steps=500)

    # Save arrays for reproducibility
    np.save(os.path.join(output_dir, "rewards_sarsa.npy"), np.array(r_s))
    np.save(os.path.join(output_dir, "rewards_qlearning.npy"), np.array(r_q))
    np.save(os.path.join(output_dir, "traj_sarsa.npy"), np.array(traj_s, dtype=int))
    np.save(os.path.join(output_dir, "traj_qlearning.npy"), np.array(traj_q, dtype=int))

    # Plots
    plot_rewards_with_ma(r_s, "SARSA: Episode return (raw & moving average)",
                         os.path.join(output_dir, "rewards_sarsa_with_ma.png"))
    plot_rewards_with_ma(r_q, "Q-learning: Episode return (raw & moving average)",
                         os.path.join(output_dir, "rewards_qlearning_with_ma.png"))

    # Policy arrows
    visualize_policy_arrows(env, pol_s, os.path.join(output_dir, "policy_sarsa.png"))
    visualize_policy_arrows(env2, pol_q, os.path.join(output_dir, "policy_qlearning.png"))

    # Trajectories
    visualize_trajectory(GridWorld(), traj_s, os.path.join(output_dir, "trajectory_sarsa.png"))
    visualize_trajectory(GridWorld(), traj_q, os.path.join(output_dir, "trajectory_qlearning.png"))

if __name__ == "__main__":
    main()
