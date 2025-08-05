"""
utils.py
Greedy policy extraction, trajectory rollout, and arrow utilities.
"""
from typing import Dict, Tuple
import numpy as np

def greedy_policy_from_Q(Q, env):
    actions = env.action_space()
    policy = {}
    for s in env.state_space():
        if env.is_terminal(s):
            continue
        qvals = [Q.get((s, a), 0.0) for a in actions]
        mx = max(qvals)
        best = [a for a, q in zip(actions, qvals) if q == mx]
        policy[s] = best[0]
    return policy

def rollout_policy(env, policy, max_steps=500):
    s = env.reset()
    trajectory = [s]
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        a = policy.get(s, np.random.choice(env.action_space()))
        s_next, r, done, _ = env.step(a)
        trajectory.append(s_next)
        total_reward += r
        s = s_next
        steps += 1
    return trajectory, total_reward, done

def to_arrows(a: int) -> str:
    return {0:'↑', 1:'→', 2:'↓', 3:'←'}.get(a, '·')
