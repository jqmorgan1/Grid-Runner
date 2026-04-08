"""
Grid World Q-Learning (AI 100 — Adaptive Mechanics demo).
Uses only: standard library, random, time, matplotlib (plotting only).
"""

from __future__ import annotations

import os
import random
import time

import matplotlib.pyplot as plt

# --- Hyperparameters (fixed per assignment) ---
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 500
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
GRID_SIZE = 5
MAX_STEPS_PER_EPISODE = 200

# Rewards
STEP_COST = -1
GOAL_BONUS = 10
TRAP_PENALTY = -10

# Layout: Start (0,0), Goal (4,4); three traps (not on S or G)
START = (0, 0)
GOAL = (4, 4)
TRAPS = {(1, 1), (2, 2), (3, 1)}

ACTIONS = ("up", "down", "left", "right")
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def print_adaptation_summary() -> None:
    """Brief explanation of alpha and epsilon before training."""
    print(
        """
================================================================================
ADAPTATION: alpha (learning rate) and epsilon (exploration rate)
================================================================================
  alpha (learning rate) — How much each new experience updates Q(s,a).
    After each step, the Q-table moves toward the target
    r + gamma * max_a' Q(s',a'). A larger alpha makes updates aggressive
    (fast but noisy); a smaller alpha learns more slowly but smoothly.
    Here alpha = 0.1: the agent blends 10% of the new target with 90%
    of the old estimate, which stabilizes learning on this small grid.

  epsilon (exploration rate) — Probability of taking a random action
    instead of the best-known one (epsilon-greedy). High epsilon early
    encourages trying many paths (exploration); as epsilon decays
    (here × 0.99 after each episode), the agent relies more on the Q-table
    (exploitation), which is how behavior *adapts* from broad search to
    refined policy.
================================================================================
"""
    )


def print_grid_legend() -> None:
    """Terminal visualization of the 5x5 grid layout."""
    cells = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    r0, c0 = START
    rg, cg = GOAL
    cells[r0][c0] = "S"
    cells[rg][cg] = "G"
    for tr, tc in TRAPS:
        cells[tr][tc] = "X"
    print("\nGrid layout (terminal):")
    print("  " + " ".join(str(i) for i in range(GRID_SIZE)))
    for i, row in enumerate(cells):
        print(f"{i} " + " ".join(row))
    print("  S=start, G=goal (+10), X=trap (-10), each move costs -1\n")


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def step_state(row: int, col: int, action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    nr, nc = row + dr, col + dc
    if in_bounds(nr, nc):
        return nr, nc
    return row, col


def reward_for_landing(row: int, col: int) -> float:
    if (row, col) == GOAL:
        return STEP_COST + GOAL_BONUS
    if (row, col) in TRAPS:
        return STEP_COST + TRAP_PENALTY
    return float(STEP_COST)


def is_terminal(row: int, col: int) -> bool:
    return (row, col) == GOAL or (row, col) in TRAPS


def make_q_table() -> dict[tuple[int, int], dict[str, float]]:
    q: dict[tuple[int, int], dict[str, float]] = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if is_terminal(r, c):
                continue
            q[(r, c)] = {a: 0.0 for a in ACTIONS}
    return q


def best_action(q: dict[tuple[int, int], dict[str, float]], state: tuple[int, int]) -> str:
    qa = q[state]
    best = max(qa.values())
    best_actions = [a for a, v in qa.items() if v == best]
    return random.choice(best_actions)


def epsilon_greedy(
    q: dict[tuple[int, int], dict[str, float]],
    state: tuple[int, int],
    epsilon: float,
) -> str:
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return best_action(q, state)


def q_learning_update(
    q: dict[tuple[int, int], dict[str, float]],
    s: tuple[int, int],
    a: str,
    r: float,
    s_next: tuple[int, int],
    done: bool,
) -> None:
    if done:
        target = r
    else:
        max_next = max(q[s_next].values())
        target = r + GAMMA * max_next
    old = q[s][a]
    q[s][a] = old + ALPHA * (target - old)


def run_episode(
    q: dict[tuple[int, int], dict[str, float]],
    epsilon: float,
) -> float:
    row, col = START
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):
        if is_terminal(row, col):
            break
        state = (row, col)
        action = epsilon_greedy(q, state, epsilon)
        nr, nc = step_state(row, col, action)
        r = reward_for_landing(nr, nc)
        total_reward += r
        done = is_terminal(nr, nc)
        q_learning_update(q, state, action, r, (nr, nc), done)
        row, col = nr, nc
        if done:
            break
    return total_reward


def moving_average(values: list[float], window: int) -> list[float]:
    if not values or window < 1:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    print_adaptation_summary()
    print_grid_legend()

    random.seed()
    q = make_q_table()
    epsilon = EPSILON_START
    episode_rewards: list[float] = []

    t0 = time.perf_counter()
    for ep in range(1, EPISODES + 1):
        total = run_episode(q, epsilon)
        episode_rewards.append(total)
        epsilon *= EPSILON_DECAY
    elapsed = time.perf_counter() - t0
    print(f"Training finished: {EPISODES} episodes in {elapsed:.3f}s")
    print(f"Final epsilon (after last decay): {epsilon:.6f}")

    episodes_x = list(range(1, EPISODES + 1))
    window = 25
    trend = moving_average(episode_rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, episode_rewards, alpha=0.35, label="Total reward per episode")
    plt.plot(
        episodes_x,
        trend,
        color="darkorange",
        linewidth=2,
        label=f"Moving average (window={window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Grid World Q-Learning: reward vs episode (adaptive improvement)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("qlearning_adaptation.png", dpi=150)
    print("Saved plot: qlearning_adaptation.png")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
