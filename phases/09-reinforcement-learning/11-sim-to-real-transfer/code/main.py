"""Domain randomization on a cliff-walking GridWorld.

The grid is 4 rows by 6 columns. The bottom row holds the start, a cliff, and
the goal:

      col  0  1  2  3  4  5
    row 0  .  .  .  .  .  .
    row 1  .  .  .  .  .  .
    row 2  .  .  .  .  .  .
    row 3  S  X  X  X  X  G

Every move costs -1. Stepping into a cliff cell costs -20 and teleports the
agent back to the start without ending the episode. `slip` is the probability
that the motors misfire and the agent moves perpendicular to the command.

The cliff is what makes the lesson's claim measurable. The shortest safe path
runs along row 2, directly above the cliff. A policy trained at slip = 0 has no
reason to keep a margin, so it walks that edge -- and falls off it as soon as
the "real" robot slips. A policy trained across a range of slips pays two extra
steps for a route one row further from the edge, and keeps working far outside
the range it trained on.
"""

import random
from collections import defaultdict


ROWS, COLS = 4, 6
START = (ROWS - 1, 0)
GOAL = (ROWS - 1, COLS - 1)
CLIFF = frozenset((ROWS - 1, c) for c in range(1, COLS - 1))
STEP_COST = -1.0
CLIFF_COST = -20.0
MAX_STEPS = 100

ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def step(state, action, slip, rng):
    # The die is rolled unconditionally, even at slip = 0.0: otherwise the
    # number of rng draws depends on slip, and the same seed would produce
    # different rollouts at different slips -- nothing left to compare.
    if rng.random() < slip:
        perp = ("left", "right") if action in ("up", "down") else ("up", "down")
        action = rng.choice(perp)
    dr, dc = DELTAS[action]
    r, c = state
    nr = min(max(r + dr, 0), ROWS - 1)
    nc = min(max(c + dc, 0), COLS - 1)
    if (nr, nc) in CLIFF:
        return START, CLIFF_COST, False
    return (nr, nc), STEP_COST, (nr, nc) == GOAL


def default_q():
    return {a: 0.0 for a in ACTIONS}


def epsilon_greedy(Q, s, rng, eps):
    if rng.random() < eps:
        return rng.choice(ACTIONS)
    q = Q[s]
    return max(ACTIONS, key=lambda a: q[a])


def train_dr(slip_low, slip_high, episodes=3000, alpha=0.1, gamma=0.99, eps=0.15, rng=None):
    """Q-learning with a fresh slip sampled from [slip_low, slip_high] per episode."""
    rng = rng or random.Random(0)
    Q = defaultdict(default_q)
    for _ in range(episodes):
        slip_ep = rng.uniform(slip_low, slip_high)
        s = START
        for _ in range(MAX_STEPS):
            a = epsilon_greedy(Q, s, rng, eps)
            s_next, r, done = step(s, a, slip_ep, rng)
            target = r if done else r + gamma * max(Q[s_next].values())
            Q[s][a] += alpha * (target - Q[s][a])
            s = s_next
            if done:
                break
    return Q


def train_fixed(slip, **kwargs):
    """The same learner with a degenerate range: the only difference is width."""
    return train_dr(slip, slip, **kwargs)


def evaluate(Q, slip, episodes=200, rng=None):
    rng = rng or random.Random(42)
    total = 0.0
    for _ in range(episodes):
        s = START
        ep_total = 0.0
        for _ in range(MAX_STEPS):
            a = max(ACTIONS, key=lambda a: Q[s][a])
            s, r, done = step(s, a, slip, rng)
            ep_total += r
            if done:
                break
        total += ep_total
    return total / episodes


def greedy_path(Q, max_steps=MAX_STEPS):
    """The route the greedy policy takes with the motors behaving perfectly."""
    rng = random.Random(0)
    s = START
    path = [s]
    for _ in range(max_steps):
        a = max(ACTIONS, key=lambda a: Q[s][a])
        s, _, done = step(s, a, 0.0, rng)
        path.append(s)
        if done:
            break
    return path


def render(path):
    cells = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            if (r, c) in CLIFF:
                row.append("X")
            elif (r, c) == START:
                row.append("S")
            elif (r, c) == GOAL:
                row.append("G")
            elif (r, c) in path:
                row.append("*")
            else:
                row.append(".")
        cells.append(" ".join(row))
    return cells


def main():
    print("=== sim-to-real: train in 'sim', evaluate on 'real' ===")
    print(f"env: {ROWS}x{COLS} cliff walk, S={START}, G={GOAL}, cliff cost {CLIFF_COST:.0f}")
    print("slip = probability the motors send the agent sideways")
    print()

    print("three policies, same learner, same compute budget, different DR range:")
    print("  A  fixed     slip = 0.0            (no randomization)")
    print("  B  DR        slip ~ U[0.0, 0.3]    (matched to expected reality)")
    print("  C  over-DR   slip ~ U[0.0, 0.9]    (randomize everything)")
    print()

    Q_fixed = train_fixed(0.0, rng=random.Random(1))
    Q_dr = train_dr(0.0, 0.3, rng=random.Random(1))
    Q_over = train_dr(0.0, 0.9, rng=random.Random(1))

    print("routes chosen (greedy, perfect motors):")
    for name, Q in (("A fixed", Q_fixed), ("B DR", Q_dr), ("C over-DR", Q_over)):
        path = greedy_path(Q)
        if path[-1] == GOAL:
            reached = "reaches G"
        elif set(path) == {START}:
            reached = "never leaves S"
        else:
            reached = "never reaches G"
        print(f"  {name:<12}{len(path) - 1:>3} steps, {reached}")
        for line in render(path):
            print(f"    {line}")
        print()

    print(f"evaluation on 'real' slips ({200} greedy episodes each, cap {MAX_STEPS} steps):")
    print(f"  {'slip':<8}{'A fixed':<12}{'B DR':<12}{'C over-DR':<12}")
    for slip in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        r_fixed = evaluate(Q_fixed, slip)
        r_dr = evaluate(Q_dr, slip)
        r_over = evaluate(Q_over, slip)
        label = "(in-support for B)" if slip <= 0.3 else "(OOD for B)"
        print(f"  {slip:<8.2f}{r_fixed:<12.2f}{r_dr:<12.2f}{r_over:<12.2f}{label}")

    print()
    print("takeaway:")
    print("  A is optimal at slip = 0 and collapses the moment the motors slip:")
    print("    the edge route it learned turns every misfire into a 20-point fall.")
    print("  B pays 2 extra steps at home and still beats A far outside its range.")
    print("  C is so risk-averse it never leaves the start at slip = 0 (-100.00 is")
    print("    the step cap, not a bug): too much randomization has its own price.")


if __name__ == "__main__":
    main()
