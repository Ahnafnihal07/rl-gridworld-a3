"""
gridworld_env.py
Deterministic 6x6 GridWorld for RL Assignment 3.
- Start: (5, 0)
- Red wall at column 3 with opening at (3, 3)
- Terminals: (0, 5) and (5, 5)
- Rewards: −20 on red cells (and reset to start), −1 otherwise (including boundary bumps & terminal entry)
- Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
"""

from typing import Tuple, List, Optional, Dict
import numpy as np

class GridWorld:
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

    def __init__(self, n_rows: int = 6, n_cols: int = 6, start: Tuple[int, int] = (5, 0),
                 terminals: Optional[List[Tuple[int, int]]] = None, opening: Tuple[int, int] = (3, 3),
                 seed: int = 42) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.opening = opening
        self.rng = np.random.default_rng(seed)

        if terminals is None:
            terminals = [(0, n_cols - 1), (n_rows - 1, n_cols - 1)]
        self.terminals = set(terminals)

        # Red wall at column 3 with an opening at row 3
        wall_col = 3
        self.red_cells = set((r, wall_col) for r in range(1, n_rows - 1) if (r, wall_col) != opening)

        self.actions = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int):
        r, c = self.state
        drc = {self.UP: (-1,0), self.RIGHT:(0,1), self.DOWN:(1,0), self.LEFT:(0,-1)}
        dr, dc = drc[action]
        nr, nc = r + dr, c + dc

        # Boundary: stay put, reward -1
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            self.state = (r, c)
            return self.state, -1, False, {}

        next_state = (nr, nc)

        # Red cell: -20 and reset to start
        if next_state in self.red_cells:
            self.state = self.start
            return self.state, -20, False, {"hit_red": True}

        # Regular move cost
        reward = -1

        # Terminal?
        if next_state in self.terminals:
            self.state = next_state
            return next_state, reward, True, {}

        self.state = next_state
        return next_state, reward, False, {}

    def layout(self) -> np.ndarray:
        """Return int grid encoding: 0 empty, 1 start, 2 terminal, 3 red, 4 opening."""
        grid = np.zeros((self.n_rows, self.n_cols), dtype=int)
        grid[self.start] = 1
        for t in self.terminals:
            grid[t] = 2
        for rc in self.red_cells:
            grid[rc] = 3
        grid[self.opening] = 4
        return grid

    def action_space(self) -> List[int]:
        return self.actions

    def state_space(self):
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return s in self.terminals
