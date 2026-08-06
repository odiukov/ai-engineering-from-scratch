"""Тесты к уроку «RL для игр: AlphaZero, MuZero и GRPO». Правь exercise.py."""

import math

import pytest

from exercise import (
    best_move,
    group_advantages,
    grpo_step,
    kl_penalty_gradient,
    minimax_value,
    puct_score,
    softmax,
    winner,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
LOOSE = lambda x: pytest.approx(x, abs=1e-6)

EMPTY = "." * 9


def variance(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def kl(probs, ref):
    return sum(p * (math.log(p) - math.log(q)) for p, q in zip(probs, ref))


def play_out(board, player):
    """Довести партию до конца, оба игрока играют best_move."""
    while winner(board) is None:
        i = best_move(board, player)
        board = board[:i] + player + board[i + 1:]
        player = "o" if player == "x" else "x"
    return winner(board)


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([0.4, -1.2, 3.0, 0.0])) == pytest.approx(1.0, abs=1e-12)


def test_softmax_survives_huge_logits():
    """Наивный math.exp(1000) падает с OverflowError."""
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0], abs=1e-12)


# -------------------------------------------------------- group_advantages
def test_group_advantages_worked_example():
    assert group_advantages([1.0, 0.0]) == LOOSE([1.0, -1.0])


def test_group_advantages_are_centred_and_scaled():
    advs = group_advantages([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert sum(advs) / len(advs) == LOOSE(0.0)
    assert variance(advs) == pytest.approx(1.0, abs=1e-5)


def test_a_group_that_solved_everything_gives_no_learning_signal():
    """Свойство GRPO, не баг: решённая задача перестаёт двигать политику."""
    assert group_advantages([1.0, 1.0, 1.0, 1.0]) == LOOSE([0.0] * 4)


def test_group_advantages_of_a_single_sample_is_zero_not_a_crash():
    """Ловушка: std == 0, и без +1e-8 это ZeroDivisionError."""
    assert group_advantages([0.5]) == LOOSE([0.0])


# ----------------------------------------------------- kl_penalty_gradient
def test_kl_gradient_is_zero_when_the_policy_equals_the_reference():
    assert kl_penalty_gradient([0.3, 0.7], [0.3, 0.7]) == LOOSE([0.0, 0.0])


def test_kl_gradient_sums_to_zero():
    """Градиент по логитам softmax обязан суммироваться в ноль."""
    got = kl_penalty_gradient([0.6, 0.3, 0.1], [0.2, 0.3, 0.5])
    assert sum(got) == LOOSE(0.0)


def test_kl_gradient_matches_the_numeric_derivative():
    """Аналитический градиент KL по логитам против центральной разности."""
    z = [0.7, -0.4, 1.1]
    ref = [0.2, 0.5, 0.3]
    analytic = kl_penalty_gradient(softmax(z), ref)
    h = 1e-6
    for i in range(len(z)):
        up, down = list(z), list(z)
        up[i] += h
        down[i] -= h
        numeric = (kl(softmax(up), ref) - kl(softmax(down), ref)) / (2 * h)
        assert analytic[i] == pytest.approx(numeric, abs=1e-6)


# -------------------------------------------------------------- grpo_step
def test_grpo_step_worked_example():
    assert grpo_step([0.0, 0.0], [0, 1], [1.0, 0.0], lr=1.0) == LOOSE([0.5, -0.5])


def test_grpo_raises_the_probability_of_the_verified_answer():
    logits = [0.0, 0.0, 0.0, 0.0]
    samples = [2, 0, 1, 3, 2, 0]
    rewards = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    new = grpo_step(logits, samples, rewards, lr=0.5)
    assert softmax(new)[2] > softmax(logits)[2]


def test_grpo_leaves_the_policy_alone_when_the_whole_group_agrees():
    """Нулевой advantage — нулевой шаг, сколько сэмплов ни набери."""
    logits = [0.3, -0.2, 0.8]
    new = grpo_step(logits, [0, 1, 2, 0], [1.0] * 4, lr=1.0)
    assert new == LOOSE(logits)


def test_grpo_step_matches_the_numeric_policy_gradient():
    logits = [0.4, -0.7, 0.2, 1.1]
    samples = [1, 3, 1, 0]
    rewards = [1.0, 0.0, 1.0, 0.0]
    lr = 0.05
    new = grpo_step(logits, samples, rewards, lr=lr)
    advs = group_advantages(rewards)

    def objective(z):
        probs = softmax(z)
        return sum(a * math.log(probs[s]) for s, a in zip(samples, advs)) / len(samples)

    h = 1e-6
    for i in range(len(logits)):
        up, down = list(logits), list(logits)
        up[i] += h
        down[i] -= h
        numeric = (objective(up) - objective(down)) / (2 * h)
        assert (new[i] - logits[i]) / lr == pytest.approx(numeric, abs=1e-5)


def test_grpo_step_matches_the_numeric_gradient_with_the_kl_leash_on():
    logits = [0.4, -0.7, 0.2]
    ref = [0.2, 0.5, 0.3]
    samples = [0, 1, 0, 2]
    rewards = [1.0, 0.0, 1.0, 0.0]
    lr, beta = 0.05, 0.3
    new = grpo_step(logits, samples, rewards, ref_probs=ref, lr=lr, beta=beta)
    advs = group_advantages(rewards)

    def objective(z):
        probs = softmax(z)
        pg = sum(a * math.log(probs[s]) for s, a in zip(samples, advs)) / len(samples)
        return pg - beta * kl(probs, ref)

    h = 1e-6
    for i in range(len(logits)):
        up, down = list(logits), list(logits)
        up[i] += h
        down[i] -= h
        numeric = (objective(up) - objective(down)) / (2 * h)
        assert (new[i] - logits[i]) / lr == pytest.approx(numeric, abs=1e-5)


def test_the_kl_leash_pulls_a_drifted_policy_back_toward_the_reference():
    """Награды одинаковые — двигать политику может только штраф KL."""
    logits = [3.0, 0.0]
    ref = [0.5, 0.5]
    before = kl(softmax(logits), ref)
    new = grpo_step(logits, [0, 1], [1.0, 1.0], ref_probs=ref, lr=1.0, beta=1.0)
    assert kl(softmax(new), ref) < before


def test_grpo_step_does_not_mutate_the_logits_it_was_given():
    logits = [0.0, 0.0]
    grpo_step(logits, [0, 1], [1.0, 0.0], lr=1.0)
    assert logits == APPROX([0.0, 0.0])


# ----------------------------------------------------------------- winner
def test_winner_finds_a_row():
    assert winner("xxx......") == "x"
    assert winner("...ooo...") == "o"


def test_winner_finds_a_column():
    assert winner("x..x..x..") == "x"


def test_winner_finds_both_diagonals():
    assert winner("x...x...x") == "x"
    assert winner("xxoxo.o..") == "o"


def test_an_unfinished_game_is_neither_a_win_nor_a_draw():
    """None и "draw" путать нельзя: для minimax это разные ветки."""
    assert winner("xoxoxoox.") is None
    assert winner(EMPTY) is None


def test_a_full_board_with_no_line_is_a_draw():
    assert winner("xxoooxxox") == "draw"


def test_empty_cells_never_form_a_line():
    """Ловушка: три точки подряд — не победа «пустоты»."""
    assert winner("xo.......") is None


# ----------------------------------------------------------- minimax_value
def test_a_finished_position_is_scored_immediately():
    assert minimax_value("xxx......", "o") == 1
    assert minimax_value("ooo......", "x") == -1
    assert minimax_value("xxoooxxox", "x") == 0


def test_tic_tac_toe_is_a_draw_under_perfect_play():
    """Классический результат: с пустой доски идеальная игра даёт ничью."""
    assert minimax_value(EMPTY, "x") == 0


def test_the_side_to_move_takes_the_win():
    """Одна доска, два вердикта: у обоих готова линия, выигрывает тот, чей ход.

    x закрывает строку 0 ходом в 2, o закрывает строку 1 ходом в 5. Кто
    успел — того и партия. Ровно на этом и стоит minimax.
    """
    board = "xx.oo...."
    assert minimax_value(board, "x") == 1
    assert minimax_value(board, "o") == -1


# -------------------------------------------------------------- best_move
def test_best_move_takes_the_immediate_win():
    assert best_move("xx.oo....", "x") == 2


def test_best_move_blocks_the_opponents_win():
    assert best_move("xx.o.....", "o") == 2


def test_best_move_only_ever_returns_an_empty_cell():
    for board, player in (("xx.o.....", "o"), ("xoxoxo...", "x"), (EMPTY, "x")):
        assert board[best_move(board, player)] == "."


def test_best_move_breaks_ties_toward_the_lower_index():
    assert best_move(EMPTY, "x") == 0


def test_two_perfect_players_always_draw():
    """Самопроверка minimax целиком: идеальная игра с обеих сторон — ничья."""
    assert play_out(EMPTY, "x") == "draw"


def test_a_perfect_player_punishes_a_bad_opening():
    """На угол отвечают только центром. o ответил краем — x обязан выиграть."""
    assert play_out("xo.......", "x") == "x"


# ------------------------------------------------------------- puct_score
def test_puct_of_an_unvisited_child_is_pure_prior():
    assert puct_score(0.0, 0.5, 4, 0, c=1.0) == APPROX(1.0)


def test_visits_shrink_the_exploration_bonus():
    """Разведанная ветка постепенно уступает место неразведанным."""
    scores = [puct_score(0.0, 0.5, 16, n, c=1.0) for n in (0, 1, 3, 7)]
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_a_stronger_prior_gets_explored_first():
    """Ровно за это AlphaZero и держит policy-сеть."""
    weak = puct_score(0.0, 0.05, 9, 0)
    strong = puct_score(0.0, 0.8, 9, 0)
    assert strong > weak


def test_zero_c_reduces_puct_to_greedy_value():
    assert puct_score(0.3, 0.9, 100, 0, c=0.0) == APPROX(0.3)


