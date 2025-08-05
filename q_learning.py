"""
q_learning.py
Off-policy Q-learning with epsilon-greedy behavior.
"""
from typing import Dict, Tuple, List
import numpy as np

def q_learning(env,
               episodes: int = 1500,
               alpha: float = 0.1,
               gamma: float = 0.99,
               epsilon: float = 0.3,
               epsilon_decay: float = 0.999,
               min_epsilon: float = 0.05,
               seed: int = 9):
    rng = np.random.default_rng(seed)
    actions = env.action_space()
    Q: Dict[Tuple[Tuple[int,int], int], float] = {}
    rewards = []

    for _ in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            if rng.random() < epsilon:
                a = rng.choice(actions)
            else:
                qvals = [Q.get((s, a), 0.0) for a in actions]
                mx = max(qvals)
                best = [a for a, q in zip(actions, qvals) if q == mx]
                a = rng.choice(best)
            s_next, r, done, _ = env.step(a)
            ep_ret += r
            max_next = 0.0 if done else max([Q.get((s_next, ap), 0.0) for ap in actions])
            qsa = Q.get((s, a), 0.0)
            target = r + gamma * max_next
            Q[(s, a)] = qsa + alpha * (target - qsa)
            s = s_next
        rewards.append(ep_ret)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return Q, rewards
