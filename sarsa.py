"""
sarsa.py
On-policy SARSA with epsilon-greedy behavior.
"""
from typing import Dict, Tuple, List
import numpy as np

def epsilon_greedy(Q: Dict[Tuple[Tuple[int,int], int], float],
                   s: Tuple[int,int],
                   actions: List[int],
                   epsilon: float,
                   rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return rng.choice(actions)
    qvals = [Q.get((s, a), 0.0) for a in actions]
    mx = max(qvals)
    best = [a for a, q in zip(actions, qvals) if q == mx]
    return rng.choice(best)

def sarsa(env,
          episodes: int = 1500,
          alpha: float = 0.1,
          gamma: float = 0.99,
          epsilon: float = 0.3,
          epsilon_decay: float = 0.999,
          min_epsilon: float = 0.05,
          seed: int = 7):
    rng = np.random.default_rng(seed)
    actions = env.action_space()
    Q: Dict[Tuple[Tuple[int,int], int], float] = {}
    rewards = []

    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, actions, epsilon, rng)
        ep_ret = 0.0
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            ep_ret += r
            if not done:
                a_next = epsilon_greedy(Q, s_next, actions, epsilon, rng)
            qsa = Q.get((s, a), 0.0)
            target = r + (gamma * Q.get((s_next, a_next), 0.0) if not done else r)
            Q[(s, a)] = qsa + alpha * (target - qsa)
            s, a = s_next, (a_next if not done else a)
        rewards.append(ep_ret)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return Q, rewards
