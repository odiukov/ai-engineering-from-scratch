"""Тесты к уроку «Society of Mind и дебаты агентов». Правь exercise.py."""

import pytest

from exercise import (
    agreement_score,
    debate_round,
    opinion_clusters,
    rounds_to_consensus,
    run_debate,
    spread,
    sycophancy_collapse,
    weighted_mean,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in rows for x in row]


# ----------------------------------------------------------- weighted_mean
def test_weighted_mean_with_equal_weights_is_plain_mean():
    assert weighted_mean([10.0, 20.0], [1.0, 1.0]) == APPROX(15.0)


def test_weighted_mean_pulls_toward_the_heavier_value():
    assert weighted_mean([10.0, 20.0], [3.0, 1.0]) == APPROX(12.5)


def test_weighted_mean_ignores_the_overall_scale_of_weights():
    """Уверенности можно умножить на 100 — консенсус не сдвинется."""
    a = weighted_mean([1.0, 4.0], [0.2, 0.6])
    b = weighted_mean([1.0, 4.0], [20.0, 60.0])
    assert a == APPROX(b)


def test_weighted_mean_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_mean([1.0, 2.0], [0.0, 0.0])


def test_weighted_mean_rejects_length_mismatch():
    with pytest.raises(ValueError):
        weighted_mean([1.0, 2.0], [1.0])


# ------------------------------------------------------------------ spread
def test_spread_is_zero_on_full_consensus():
    assert spread([7.0, 7.0, 7.0]) == APPROX(0.0)


def test_spread_ignores_the_middle_values():
    assert spread([1.0, 5.0, 3.0]) == APPROX(4.0)


# --------------------------------------------------------- agreement_score
def test_agreement_score_is_one_when_all_answers_match():
    assert agreement_score([5.0, 5.0, 5.0]) == APPROX(1.0)


def test_agreement_score_counts_only_agents_near_the_mean():
    assert agreement_score([0.0, 5.0, 10.0], 1.0) == APPROX(1 / 3)


def test_agreement_score_of_one_is_not_evidence_of_truth():
    """Три агента дружно ошибаются — согласие 1.0, правда далеко."""
    truth = 42.0
    wrong = [9.0, 9.0, 9.0]
    assert agreement_score(wrong) == APPROX(1.0)
    assert abs(sum(wrong) / len(wrong) - truth) > 30


# ------------------------------------------------------------ debate_round
def test_debate_round_averages_when_everyone_listens_to_everyone():
    assert debate_round([0.0, 10.0, 20.0], [1.0, 1.0, 1.0]) == APPROX([10.0, 10.0, 10.0])


def test_debate_round_is_simultaneous_not_sequential():
    """Все читают позиции ПРОШЛОГО раунда.

    При последовательном обновлении второй агент увидел бы уже сдвинутого
    первого и получил бы 13.33 вместо 10.0.
    """
    assert debate_round([0.0, 10.0, 20.0], [1.0, 1.0, 1.0])[1] == APPROX(10.0)


def test_debate_round_does_not_mutate_the_input():
    answers = [0.0, 10.0]
    debate_round(answers, [1.0, 1.0])
    assert answers == [0.0, 10.0]


def test_debate_round_respects_the_confidence_weights():
    """Уверенный агент тянет консенсус к себе."""
    result = debate_round([0.0, 10.0], [3.0, 1.0])
    assert result == APPROX([2.5, 2.5])


def test_debate_round_ignores_positions_outside_the_radius():
    assert debate_round([0.0, 1.0, 10.0], [1.0, 1.0, 1.0], 2.0) == APPROX([0.5, 0.5, 10.0])


def test_debate_round_with_full_stubbornness_changes_nothing():
    assert debate_round([1.0, 9.0], [1.0, 1.0], stubbornness=1.0) == APPROX([1.0, 9.0])


def test_debate_round_rejects_stubbornness_out_of_range():
    with pytest.raises(ValueError):
        debate_round([1.0, 2.0], [1.0, 1.0], stubbornness=1.5)


# -------------------------------------------------------------- run_debate
def test_run_debate_history_includes_round_zero():
    history = run_debate([0.0, 10.0], [1.0, 1.0], 1)
    assert len(history) == 2
    assert flat(history) == APPROX([0.0, 10.0, 5.0, 5.0])


def test_run_debate_with_zero_rounds_returns_only_the_start():
    assert flat(run_debate([1.0, 2.0], [1.0, 1.0], 0)) == APPROX([1.0, 2.0])


def test_run_debate_does_not_mutate_the_start():
    answers = [0.0, 10.0]
    run_debate(answers, [1.0, 1.0], 3)
    assert answers == [0.0, 10.0]


def test_run_debate_spread_shrinks_geometrically_with_stubbornness():
    """При stubbornness=s и бесконечном радиусе разброс за раунд множится на s."""
    history = run_debate([0.0, 8.0], [1.0, 1.0], 3, stubbornness=0.5)
    widths = [spread(row) for row in history]
    assert widths == APPROX([8.0, 4.0, 2.0, 1.0])


def test_run_debate_gain_per_round_shrinks():
    """Плато Du et al.: первый раунд двигает сильнее любого следующего."""
    history = run_debate([0.0, 8.0], [1.0, 1.0], 4, stubbornness=0.5)
    gains = [spread(a) - spread(b) for a, b in zip(history, history[1:])]
    assert gains[0] > gains[1] > gains[2] > gains[3]


# ------------------------------------------------------- rounds_to_consensus
def test_rounds_to_consensus_is_zero_when_they_already_agree():
    assert rounds_to_consensus([5.0, 5.0], [1.0, 1.0]) == 0


def test_rounds_to_consensus_is_faster_when_positions_overlap_more():
    """Близкие стартовые позиции сходятся строго быстрее далёких."""
    close = rounds_to_consensus([40.0, 41.0, 42.0], [1.0] * 3, 0.5, 1e9, 0.5)
    far = rounds_to_consensus([30.0, 42.0, 54.0], [1.0] * 3, 0.5, 1e9, 0.5)
    assert close is not None and far is not None
    assert close < far


def test_rounds_to_consensus_returns_none_under_polarization():
    """Два лагеря дальше радиуса доверия — консенсуса не будет никогда."""
    answers = [0.0, 1.0, 50.0, 51.0]
    assert rounds_to_consensus(answers, [1.0] * 4, 0.5, radius=5.0) is None


def test_rounds_to_consensus_does_not_hang_on_polarization():
    """Ответ None приходит за max_rounds, а не зависает."""
    assert rounds_to_consensus([0.0, 100.0], [1.0, 1.0], 0.1, radius=1.0, max_rounds=5) is None


# --------------------------------------------------------- opinion_clusters
def test_opinion_clusters_finds_one_cluster_on_consensus():
    assert opinion_clusters([5.0, 5.0, 5.0]) == [[0, 1, 2]]


def test_opinion_clusters_splits_two_camps():
    assert opinion_clusters([1.0, 1.05, 9.0]) == [[0, 1], [2]]


def test_opinion_clusters_chains_close_neighbours_into_one():
    """Порог считается между соседями: цепочка 0-0.05-0.1 — один кластер."""
    assert opinion_clusters([0.0, 0.05, 0.1], 0.06) == [[0, 1, 2]]


def test_opinion_clusters_survives_the_polarized_debate():
    """Дебаты «закончились», но группа осталась разбитой на два лагеря."""
    history = run_debate([0.0, 1.0, 50.0, 51.0], [1.0] * 4, 20, radius=5.0)
    assert len(opinion_clusters(history[-1], 1.0)) == 2


# ------------------------------------------------------ sycophancy_collapse
def test_sycophancy_collapse_copies_the_most_confident_answer():
    assert sycophancy_collapse([1.0, 2.0, 3.0], [0.2, 0.9, 0.5]) == APPROX([2.0, 2.0, 2.0])


def test_sycophancy_collapse_breaks_ties_toward_the_first_speaker():
    assert sycophancy_collapse([1.0, 2.0], [0.5, 0.5]) == APPROX([1.0, 1.0])


def test_sycophancy_collapse_fakes_perfect_agreement_in_one_round():
    """Разброс в ноль и согласие 1.0 — при том, что средний ответ стал ХУЖЕ."""
    truth = 42.0
    answers = [41.0, 10.0, 43.0]
    collapsed = sycophancy_collapse(answers, [0.1, 0.9, 0.2])
    assert spread(collapsed) == APPROX(0.0)
    assert agreement_score(collapsed) == APPROX(1.0)
    before = abs(sum(answers) / len(answers) - truth)
    after = abs(sum(collapsed) / len(collapsed) - truth)
    assert after > before
