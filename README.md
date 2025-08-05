# RL Assignment 3 — SARSA vs. Q-learning in a Barrier GridWorld

This repository contains reproducible code and a formal report comparing **SARSA** and **Q-learning** with ε-greedy exploration in a deterministic GridWorld with penalty states.

## Environment Summary
- Grid: 6×6
- Start: (5, 0)
- Red wall at column 3 with opening at (3, 3)
- Terminals: (0, 5) and (5, 5)
- Rewards: −20 on red cells (and reset to start), −1 otherwise (including boundary bumps and terminal entry)
- Actions: Up, Right, Down, Left (deterministic)

## Quickstart (Reproduce Results)
```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt

# generate all figures and arrays in report/figures/
python code/main.py
```

Outputs saved to `report/figures/`:
- `rewards_sarsa_with_ma.png`, `rewards_qlearning_with_ma.png`
- `policy_sarsa.png`, `policy_qlearning.png`
- `trajectory_sarsa.png`, `trajectory_qlearning.png`
- `.npy` arrays: `rewards_*`, `traj_*`

## Project Structure
```
.
├── code/
│   ├── gridworld_env.py
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── utils.py
│   └── main.py
├── report/
│   ├── report.tex
│   └── figures/
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Reproducibility Checklist
- [x] Fixed seeds in environment and trainers
- [x] Single-command regeneration (`python code/main.py`)
- [x] Minimal dependencies (`numpy`, `matplotlib`)
- [x] Clear environment layout assumptions in code comments
- [x] Saved intermediate arrays for verification

## Citation
Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
