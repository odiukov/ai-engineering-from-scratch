import math
import random


GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
N_ACTIONS = len(ACTIONS)
N_FEAT = GRID * GRID


def reset():
    return (0, 0)


def step(state, action_idx):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = DELTAS[ACTIONS[action_idx]]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def features(state):
    x = [0.0] * N_FEAT
    r, c = state
    x[r * GRID + c] = 1.0
    return x


def init_theta(rng):
    return [[rng.gauss(0, 0.1) for _ in range(N_FEAT)] for _ in range(N_ACTIONS)]


def logits(theta, x):
    return [sum(w * xi for w, xi in zip(theta[a], x)) for a in range(N_ACTIONS)]


def softmax(z):
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    Z = sum(exps)
    return [e / Z for e in exps]


def sample(probs, rng):
    x = rng.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return N_ACTIONS - 1


def rollout(theta, rng, max_steps=100):
    traj = []
    s = reset()
    for _ in range(max_steps):
        x = features(s)
        probs = softmax(logits(theta, x))
        a = sample(probs, rng)
        s_next, r, done = step(s, a)
        traj.append((x, a, r, probs))
        if done:
            break
        s = s_next
    return traj


def returns_to_go(traj, gamma):
    G = 0.0
    out = []
    for _, _, r, _ in reversed(traj):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def grad_log_pi(probs, a):
    """grad of log pi(a|s) w.r.t. the logits: onehot(a) - pi(.|s)."""
    return [(1.0 if i == a else 0.0) - probs[i] for i in range(N_ACTIONS)]


def reinforce_step(theta, traj, returns, lr, baseline=0.0):
    """theta += lr * sum_t (G_t - baseline) * grad log pi(a_t | s_t)."""
    for (x, a, _r, probs), G in zip(traj, returns):
        adv = G - baseline
        grads = grad_log_pi(probs, a)
        for i in range(N_ACTIONS):
            for j in range(N_FEAT):
                theta[i][j] += lr * adv * grads[i] * x[j]


def reinforce(episodes, lr=0.05, gamma=0.99, use_baseline=False, rng=None):
    rng = rng or random.Random(0)
    theta = init_theta(rng)
    baseline = 0.0
    returns_log = []
    for _ep in range(episodes):
        traj = rollout(theta, rng)
        returns = returns_to_go(traj, gamma)
        if use_baseline:
            baseline = 0.95 * baseline + 0.05 * returns[0]
        reinforce_step(theta, traj, returns, lr, baseline if use_baseline else 0.0)
        returns_log.append(returns[0] if returns else 0.0)
    return theta, returns_log


def greedy_policy(theta):
    policy = {}
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == TERMINAL:
                continue
            z = logits(theta, features((r, c)))
            policy[(r, c)] = ACTIONS[max(range(N_ACTIONS), key=lambda i: z[i])]
    return policy


def print_policy(policy, title):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    print(f"  {title}")
    for r in range(GRID):
        row = []
        for c in range(GRID):
            if (r, c) == TERMINAL:
                row.append(".")
            elif (r, c) in policy:
                row.append(arrows[policy[(r, c)]])
            else:
                row.append("?")
        print("   " + " ".join(row))


def block_mean(xs, block):
    return [sum(xs[i : i + block]) / block for i in range(0, len(xs) - block + 1, block)]


def main():
    episodes = 2000
    rng = random.Random(42)
    _, r_plain = reinforce(episodes, use_baseline=False, rng=rng)
    rng = random.Random(42)
    theta_b, r_base = reinforce(episodes, use_baseline=True, rng=rng)

    print(f"=== REINFORCE on 4x4 GridWorld, {episodes} episodes, lr=0.05, gamma=0.99 ===")
    print()
    print("learning curves (mean return per 200 episodes):")
    for i, (a, b) in enumerate(zip(block_mean(r_plain, 200), block_mean(r_base, 200))):
        print(f"  block {i+1}: vanilla={a:7.2f}   with-baseline={b:7.2f}")

    print()
    print_policy(greedy_policy(theta_b), "final greedy policy (with-baseline)")
    print()
    print(f"final mean return (last 200 eps): vanilla={sum(r_plain[-200:])/200:.2f}   with-baseline={sum(r_base[-200:])/200:.2f}")
    print("(these are discounted returns; the optimal discounted return on this")
    print(" 4x4 GridWorld is -5.85 = sum of -0.99^i for i in 0..5, i.e. -6.0 undiscounted)")


if __name__ == "__main__":
    main()
