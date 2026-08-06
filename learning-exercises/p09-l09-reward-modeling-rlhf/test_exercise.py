"""Тесты к уроку «Reward modeling и RLHF». Правь exercise.py."""

import math

import pytest

from exercise import (
    bt_gradient,
    bt_loss,
    kl_divergence,
    pairwise_accuracy,
    penalized_reward,
    reward_score,
    sigmoid,
    train_reward_model,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

GOOD = ("clear", "specific", "kind", "thorough")
BAD = ("vague", "rude", "wrong", "short")

# синтетические предпочтения: два хороших слова против двух плохих
PAIRS = [
    ([GOOD[i % 4], GOOD[(i + 1) % 4]], [BAD[i % 4], BAD[(i + 2) % 4]])
    for i in range(24)
]
HOLDOUT = [
    ([GOOD[(i + 2) % 4], GOOD[(i + 3) % 4]], [BAD[(i + 1) % 4], BAD[(i + 3) % 4]])
    for i in range(12)
]


def flat_dict(d, keys):
    """pytest.approx не умеет dict-of-list — сравниваем по фиксированному порядку."""
    return [d.get(k, 0.0) for k in keys]


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_is_bounded_and_monotone():
    assert 0.0 < sigmoid(-5.0) < sigmoid(0.0) < sigmoid(5.0) < 1.0


def test_sigmoid_survives_large_negative_input():
    """Наивная 1/(1+exp(-x)) падает с OverflowError на x = -1000."""
    assert sigmoid(-1000.0) == APPROX(0.0)


def test_sigmoid_survives_large_positive_input():
    assert sigmoid(1000.0) == APPROX(1.0)


# ----------------------------------------------------------- reward_score
def test_reward_score_sums_token_weights():
    assert reward_score({"clear": 1.5, "rude": -2.0}, ["clear", "rude"]) == APPROX(-0.5)


def test_reward_score_counts_a_repeated_token_twice():
    """Ловушка: set() вместо Counter — и повтор потеряется вместе с градиентом."""
    assert reward_score({"clear": 1.5}, ["clear", "clear"]) == APPROX(3.0)


def test_reward_score_of_an_unknown_token_is_zero_not_an_error():
    """RM обязана оценивать невиданный ответ, а не падать с KeyError."""
    assert reward_score({"clear": 1.5}, ["banana"]) == APPROX(0.0)


# --------------------------------------------------------------- bt_loss
def test_bt_loss_of_a_tie_is_log_two():
    assert bt_loss({}, ["a"], ["b"]) == pytest.approx(math.log(2), abs=1e-12)


def test_bt_loss_is_near_zero_when_the_preferred_answer_wins_big():
    assert bt_loss({"a": 20.0}, ["a"], ["b"]) < 1e-8


def test_bt_loss_is_large_when_the_model_prefers_the_rejected_answer():
    assert bt_loss({"b": 10.0}, ["a"], ["b"]) > 9.0


def test_bt_loss_is_always_positive():
    for w in ({}, {"a": 5.0}, {"b": 5.0}, {"a": -3.0, "b": 3.0}):
        assert bt_loss(w, ["a"], ["b"]) > 0.0


def test_bt_loss_depends_only_on_the_margin_not_the_absolute_level():
    """Шкала reward model не определена: сдвиг обеих наград ничего не меняет."""
    low = bt_loss({"a": 1.0, "b": 0.0, "c": 0.0}, ["a", "c"], ["b", "c"])
    high = bt_loss({"a": 1.0, "b": 0.0, "c": 100.0}, ["a", "c"], ["b", "c"])
    assert low == pytest.approx(high, abs=1e-9)


# ------------------------------------------------------------ bt_gradient
def test_bt_gradient_worked_example():
    g = bt_gradient({}, ["a"], ["b"])
    assert flat_dict(g, ["a", "b"]) == APPROX([-0.5, 0.5])


def test_bt_gradient_matches_the_numeric_derivative():
    """Аналитический градиент Bradley-Terry против центральной разности."""
    w = {"a": 0.7, "b": -0.3, "c": 1.1}
    y_pos, y_neg = ["a", "c", "a"], ["b", "c"]
    analytic = bt_gradient(w, y_pos, y_neg)
    h = 1e-6
    for t in ("a", "b", "c"):
        up, down = dict(w), dict(w)
        up[t] = up.get(t, 0.0) + h
        down[t] = down.get(t, 0.0) - h
        numeric = (bt_loss(up, y_pos, y_neg) - bt_loss(down, y_pos, y_neg)) / (2 * h)
        assert analytic[t] == pytest.approx(numeric, abs=1e-6)


def test_a_token_present_in_both_answers_gets_zero_gradient():
    """Предпочтение ничего не говорит о слове, которое есть в обоих ответах."""
    g = bt_gradient({}, ["a", "shared"], ["b", "shared"])
    assert g["shared"] == APPROX(0.0)
    assert g["a"] != 0.0


def test_bt_gradient_pushes_the_preferred_tokens_up():
    """Знак минуса у градиента: шаг спуска поднимет вес предпочтённого токена."""
    g = bt_gradient({}, ["a"], ["b"])
    assert g["a"] < 0.0 < g["b"]


def test_a_confident_correct_model_gets_a_vanishing_gradient():
    """(1 - p) в множителе: уверенной и правой модели учиться уже нечему."""
    weak = bt_gradient({}, ["a"], ["b"])["a"]
    strong = bt_gradient({"a": 20.0}, ["a"], ["b"])["a"]
    assert abs(strong) < abs(weak)


# ------------------------------------------------------ train_reward_model
def test_train_reward_model_worked_example():
    w = train_reward_model([(["good"], ["bad"])], lr=1.0)
    assert flat_dict(w, ["good", "bad"]) == APPROX([0.5, -0.5])


def test_trained_model_scores_the_preferred_answer_higher():
    """Приёмка stage 2: RM обязана ставить предпочтённому ответу больший скор."""
    w = train_reward_model(PAIRS, lr=0.1, epochs=20)
    for y_pos, y_neg in PAIRS:
        assert reward_score(w, y_pos) > reward_score(w, y_neg)


def test_training_separates_good_tokens_from_bad_ones():
    w = train_reward_model(PAIRS, lr=0.1, epochs=20)
    assert min(w[t] for t in GOOD) > max(w[t] for t in BAD)


def test_training_lowers_the_bradley_terry_loss():
    before = sum(bt_loss({}, p, n) for p, n in PAIRS)
    w = train_reward_model(PAIRS, lr=0.1, epochs=20)
    after = sum(bt_loss(w, p, n) for p, n in PAIRS)
    assert after < before


def test_train_reward_model_does_not_mutate_the_weights_it_was_given():
    start = {"clear": 0.0}
    train_reward_model(PAIRS, lr=0.1, epochs=2, w=start)
    assert start == {"clear": 0.0}


# ------------------------------------------------------- pairwise_accuracy
def test_untrained_model_scores_zero_because_every_pair_is_a_tie():
    """Ничья — не половина успеха: нулевая модель обязана получить 0.0."""
    assert pairwise_accuracy({}, PAIRS) == APPROX(0.0)


def test_trained_model_generalizes_to_a_holdout_split():
    w = train_reward_model(PAIRS, lr=0.1, epochs=20)
    assert pairwise_accuracy(w, HOLDOUT) > 0.9


def test_pairwise_accuracy_of_an_inverted_model_is_zero():
    w = {t: -1.0 for t in GOOD}
    w.update({t: 1.0 for t in BAD})
    assert pairwise_accuracy(w, PAIRS) == APPROX(0.0)


def test_pairwise_accuracy_of_an_empty_split_is_zero():
    assert pairwise_accuracy({"a": 1.0}, []) == APPROX(0.0)


# ---------------------------------------------------------- kl_divergence
def test_kl_of_identical_distributions_is_zero():
    assert kl_divergence([0.3, 0.7], [0.3, 0.7]) == APPROX(0.0)


def test_kl_worked_example():
    """Ловушка: p = 0 надо пропускать, иначе log(0) уронит расчёт."""
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2), abs=1e-9)


def test_kl_is_never_negative():
    for p, q in (([0.9, 0.1], [0.5, 0.5]), ([0.2, 0.8], [0.7, 0.3]), ([0.5, 0.5], [0.1, 0.9])):
        assert kl_divergence(p, q) >= -1e-12


def test_kl_is_asymmetric():
    p, q = [0.9, 0.1], [0.5, 0.5]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-6)


# ------------------------------------------------------- penalized_reward
def test_no_penalty_when_the_policy_equals_the_reference():
    assert penalized_reward(1.0, [0.5, 0.5], [0.5, 0.5]) == APPROX(1.0)


def test_penalty_worked_example():
    got = penalized_reward(1.0, [1.0, 0.0], [0.5, 0.5], beta=0.1)
    assert got == pytest.approx(1.0 - 0.1 * math.log(2), abs=1e-9)


def test_beta_zero_removes_the_leash_entirely():
    """beta = 0 — прямая дорога к reward hacking: KL перестаёт что-либо стоить."""
    drifted = [0.999, 0.001]
    assert penalized_reward(5.0, drifted, [0.5, 0.5], beta=0.0) == APPROX(5.0)


def test_higher_beta_punishes_the_same_drift_harder():
    drifted = [0.9, 0.1]
    soft = penalized_reward(5.0, drifted, [0.5, 0.5], beta=0.01)
    hard = penalized_reward(5.0, drifted, [0.5, 0.5], beta=1.0)
    assert hard < soft < 5.0


def test_a_high_scoring_but_far_drifted_policy_can_lose_to_a_modest_one():
    """Ровно та ситуация, ради которой KL и стоит в награде."""
    hacked = penalized_reward(2.5, [0.9999, 0.0001], [0.5, 0.5], beta=1.0)
    honest = penalized_reward(2.0, [0.55, 0.45], [0.5, 0.5], beta=1.0)
    assert hacked < honest
