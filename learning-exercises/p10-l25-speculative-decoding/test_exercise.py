"""Тесты к уроку «Спекулятивное декодирование и EAGLE». Правь exercise.py."""

import random

import pytest

from exercise import (
    acceptance_rate,
    expected_tokens,
    longest_accepted_path,
    residual_distribution,
    sample_from,
    speculative_speedup,
    speculative_step,
    tree_attention_mask,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TARGET = [0.50, 0.25, 0.15, 0.10]
DRAFT = [0.20, 0.30, 0.30, 0.20]
TRIALS = 20000


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


def run_many(p, q, trials, seed):
    """Гоняет один раунд спекулятивного декодирования trials раз.

    Возвращает (частоты первого токена, доля раундов с полным принятием).
    """
    rng = random.Random(seed)
    counts = [0] * len(p)
    full = 0
    for _ in range(trials):
        drafted = sample_from(q, rng)
        out = speculative_step([p, p], [q], [drafted], rng)
        counts[out[0]] += 1
        if len(out) == 2:
            full += 1
    return [c / trials for c in counts], full / trials


# --------------------------------------------------------------- sample_from
def test_sample_from_a_point_mass_is_deterministic():
    rng = random.Random(0)
    assert all(sample_from([0.0, 1.0, 0.0], rng) == 1 for _ in range(50))


def test_sample_from_respects_the_probabilities():
    rng = random.Random(1)
    draws = [sample_from([0.9, 0.1], rng) for _ in range(2000)]
    assert 0.85 < draws.count(0) / 2000 < 0.95


def test_sample_from_never_falls_off_the_end():
    """Накопленная сумма 0.9999999 не должна давать None."""
    rng = random.Random(2)
    third = 1.0 / 3.0
    assert all(sample_from([third, third, third], rng) is not None for _ in range(500))


# ----------------------------------------------------------- expected_tokens
def test_expected_tokens_at_the_papers_working_point():
    assert expected_tokens(0.8, 4) == pytest.approx(3.3616)


def test_a_useless_draft_still_yields_one_token():
    assert expected_tokens(0.0, 4) == APPROX(1.0)


def test_a_perfect_draft_does_not_divide_by_zero():
    """alpha = 1 — предел выражения, а не ошибка входа."""
    assert expected_tokens(1.0, 4) == APPROX(5.0)


def test_expected_tokens_grow_with_the_acceptance_rate():
    assert expected_tokens(0.5, 4) < expected_tokens(0.7, 4) < expected_tokens(0.9, 4)


def test_expected_tokens_saturate_as_the_draft_gets_longer():
    """При alpha < 1 длинный черновик почти ничего не добавляет."""
    assert expected_tokens(0.5, 32) - expected_tokens(0.5, 8) < 0.01


def test_expected_tokens_rejects_an_impossible_acceptance_rate():
    with pytest.raises(ValueError):
        expected_tokens(1.5, 4)


# ------------------------------------------------------- speculative_speedup
def test_a_free_draft_gives_the_full_token_count():
    assert speculative_speedup(0.8, 4, 0.0) == APPROX(expected_tokens(0.8, 4))


def test_draft_cost_eats_into_the_speedup():
    assert speculative_speedup(0.8, 4, 0.05) < speculative_speedup(0.8, 4, 0.0)


def test_an_expensive_draft_makes_long_speculation_a_loss():
    """Смысл настройки K: с дорогим черновиком короткий выгоднее длинного."""
    assert speculative_speedup(0.6, 2, 0.4) > speculative_speedup(0.6, 16, 0.4)


def test_high_temperature_output_kills_the_speedup():
    """alpha падает до почти нуля — платим за черновик и не получаем ничего."""
    assert speculative_speedup(0.05, 4, 0.1) < 1.0


# --------------------------------------------------------- acceptance_rate
def test_a_perfect_draft_is_always_accepted():
    assert acceptance_rate(TARGET, TARGET) == APPROX(1.0)


def test_disjoint_distributions_are_never_accepted():
    assert acceptance_rate([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_acceptance_rate_is_the_overlap_area():
    assert acceptance_rate([0.7, 0.3], [0.5, 0.5]) == APPROX(0.8)


def test_predicted_acceptance_matches_the_simulation():
    """Формула sum(min(p, q)) и реальный прогон обязаны сойтись."""
    _, full = run_many(TARGET, DRAFT, TRIALS, seed=3)
    assert full == pytest.approx(acceptance_rate(TARGET, DRAFT), abs=0.02)


# --------------------------------------------------- residual_distribution
def test_residual_is_a_distribution():
    residual = residual_distribution(TARGET, DRAFT)
    assert sum(residual) == pytest.approx(1.0)
    assert all(value >= 0 for value in residual)


def test_residual_is_zero_where_the_draft_already_covers_target():
    """Там, где q >= p, добирать нечего — черновик уже съел всю массу."""
    residual = residual_distribution([0.7, 0.3], [0.3, 0.7])
    assert residual == pytest.approx([1.0, 0.0])


def test_identical_distributions_do_not_divide_by_zero():
    """Положительная часть — сплошные нули; нормировать нечего."""
    assert residual_distribution([0.5, 0.5], [0.5, 0.5]) == pytest.approx([0.5, 0.5])


# ------------------------------------------------------- speculative_step
def test_a_matching_draft_is_accepted_and_earns_a_bonus_token():
    rng = random.Random(0)
    assert speculative_step([[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0]], [0], rng) == [0, 0]


def test_a_wrong_draft_is_replaced_and_the_round_stops():
    rng = random.Random(0)
    assert speculative_step([[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0]], [0], rng) == [1]


def test_a_round_never_returns_nothing_and_never_overruns():
    rng = random.Random(4)
    p_rows = [TARGET] * 5
    q_rows = [DRAFT] * 4
    for _ in range(200):
        drafted = [sample_from(DRAFT, rng) for _ in range(4)]
        out = speculative_step(p_rows, q_rows, drafted, rng)
        assert 1 <= len(out) <= 5


def test_speculative_step_checks_the_row_counts():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        speculative_step([TARGET], [DRAFT], [0], rng)


def test_the_output_distribution_is_exactly_the_target():
    """Главное свойство схемы: быстрее — да, но выборка та же самая."""
    empirical, _ = run_many(TARGET, DRAFT, TRIALS, seed=7)
    assert empirical == pytest.approx(TARGET, abs=0.02)


def test_a_bad_draft_does_not_bend_the_output_distribution():
    """Черновик может врать как угодно — выход всё равно распределён по p."""
    awful = [0.05, 0.05, 0.05, 0.85]
    empirical, full = run_many(TARGET, awful, TRIALS, seed=11)
    assert full < 0.4
    assert empirical == pytest.approx(TARGET, abs=0.02)


# ---------------------------------------------------- tree_attention_mask
def test_a_node_sees_itself_and_its_ancestors():
    assert flat(tree_attention_mask([-1, 0, 0])) == flat([[1, 0, 0], [1, 1, 0], [1, 0, 1]])


def test_sibling_branches_do_not_see_each_other():
    mask = tree_attention_mask([-1, 0, 0, 1, 1])
    assert mask[3][2] == 0 and mask[2][1] == 0
    assert mask[3][1] == 1 and mask[3][0] == 1


def test_a_chain_tree_gives_the_plain_causal_mask():
    """Вырожденное дерево — это обычная причинная маска."""
    mask = tree_attention_mask([-1, 0, 1, 2])
    assert flat(mask) == flat([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])


def test_each_row_counts_the_depth_of_its_node():
    mask = tree_attention_mask([-1, 0, 0, 1, 1])
    assert [sum(row) for row in mask] == [1, 2, 2, 3, 3]


def test_a_child_before_its_parent_is_a_broken_tree():
    with pytest.raises(ValueError):
        tree_attention_mask([1, -1])


# -------------------------------------------------- longest_accepted_path
def test_the_longest_fully_accepted_branch_wins():
    assert longest_accepted_path([-1, 0, 0, 1, 1], [True, True, False, False, True]) == [0, 1, 4]


def test_a_rejected_root_leaves_nothing():
    assert longest_accepted_path([-1, 0], [False, True]) == []


def test_a_rejected_ancestor_cuts_the_whole_branch():
    """Узел 3 принят, но его родитель 1 отклонён — ветка не годится."""
    assert longest_accepted_path([-1, 0, 0, 1], [True, False, True, True]) == [0, 2]


def test_ties_go_to_the_smaller_leaf_index():
    assert longest_accepted_path([-1, 0, 0, 1, 1], [True] * 5) == [0, 1, 3]
