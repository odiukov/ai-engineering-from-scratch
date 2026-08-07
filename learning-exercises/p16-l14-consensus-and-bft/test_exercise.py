"""Тесты к уроку «Консенсус и византийская отказоустойчивость». Правь exercise.py."""

import random

import pytest

from exercise import (
    BYZANTINE_LIE,
    TRUE_ANSWER,
    canonicalize,
    cluster_votes,
    geometric_median,
    max_faulty,
    plurality,
    quorum_size,
    simulate_bft,
    weighted_consensus,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------- max_faulty
def test_max_faulty_follows_the_three_f_plus_one_bound():
    assert [max_faulty(n) for n in (4, 7, 10, 100)] == [1, 2, 3, 33]


def test_max_faulty_of_a_tiny_network_is_zero():
    assert (max_faulty(1), max_faulty(2), max_faulty(3)) == (0, 0, 0)


def test_max_faulty_rejects_an_empty_network():
    with pytest.raises(ValueError):
        max_faulty(0)


def test_max_faulty_is_the_largest_f_that_still_fits():
    """n >= 3f + 1 обязано выполняться, а для f + 1 — уже нет."""
    for n in range(1, 60):
        f = max_faulty(n)
        assert n >= 3 * f + 1
        assert n < 3 * (f + 1) + 1


# -------------------------------------------------------------- quorum_size
def test_quorum_size_known_values():
    assert [quorum_size(n) for n in (4, 5, 7, 10)] == [3, 4, 5, 7]


def test_two_quorums_always_overlap_in_an_honest_node():
    """Иначе две группы узлов примут два разных решения — расщепление сети."""
    for n in range(1, 60):
        assert 2 * quorum_size(n) - n > max_faulty(n)


def test_quorum_never_asks_for_more_nodes_than_exist():
    for n in range(1, 60):
        assert quorum_size(n) <= n


def test_quorum_is_always_more_than_half_the_network():
    for n in range(1, 60):
        assert 2 * quorum_size(n) > n


# ------------------------------------------------------------- canonicalize
def test_canonicalize_maps_paraphrases_to_one_key():
    assert canonicalize("the study reports 4.2% improvement") == canonicalize(
        "4.2% gain"
    )


def test_canonicalize_drops_the_trailing_zero():
    """Иначе «42» и «42.0» разъедутся по разным кластерам и развалят счёт."""
    assert canonicalize("42.0") == canonicalize("42")


def test_canonicalize_keeps_genuinely_different_numbers_apart():
    assert canonicalize("4.2%") != canonicalize("42%")


def test_canonicalize_falls_back_to_normalized_text():
    assert canonicalize("  Yes,  DEFINITELY ") == "yes, definitely"


def test_canonicalize_understands_a_negative_number():
    assert canonicalize("loss of -3.5 points") == canonicalize("-3.5")


# ------------------------------------------------------------ cluster_votes
def test_cluster_votes_groups_equivalent_answers():
    clusters = cluster_votes([("4.2%", 0.9), ("4.2 percent", 0.8), ("42", 0.7)])
    assert {k: len(v) for k, v in clusters.items()} == {"4.2": 2, "42": 1}


def test_cluster_votes_keeps_first_appearance_order():
    clusters = cluster_votes([("42", 0.7), ("4.2%", 0.9), ("42.0", 0.6)])
    assert list(clusters) == ["42", "4.2"]


def test_cluster_votes_of_nothing_is_empty():
    assert cluster_votes([]) == {}


# ---------------------------------------------------------------- plurality
def test_plurality_picks_the_biggest_cluster():
    assert plurality([("4.2", 0.9), ("42", 0.1), ("42", 0.1)]) == "42"


def test_plurality_of_no_votes_is_none():
    assert plurality([]) is None


def test_plurality_falls_for_the_monoculture_attack():
    """Три агента на одной базовой модели ошибаются одинаково — и побеждают."""
    votes = [(BYZANTINE_LIE, 0.5)] * 3 + [(TRUE_ANSWER, 0.9)] * 2
    assert plurality(votes) == canonicalize(BYZANTINE_LIE)


def test_plurality_breaks_ties_the_same_way_regardless_of_input_order():
    votes = [(TRUE_ANSWER, 0.5), (BYZANTINE_LIE, 0.5)]
    assert plurality(votes) == plurality(list(reversed(votes)))


# ------------------------------------------------------- weighted_consensus
def test_confidence_weight_beats_the_monoculture_majority():
    """Тот же вход, что провалила plurality: вес уверенности вытягивает истину."""
    votes = [(BYZANTINE_LIE, 0.5)] * 3 + [(TRUE_ANSWER, 0.9)] * 2
    assert weighted_consensus(votes) == canonicalize(TRUE_ANSWER)


def test_sycophantic_conformity_carries_little_weight():
    """Подпевалы повторяют громкого соседа, но своей уверенности у них нет."""
    votes = [
        (BYZANTINE_LIE, 0.95),
        (BYZANTINE_LIE, 0.2),
        (BYZANTINE_LIE, 0.2),
        (TRUE_ANSWER, 0.9),
        (TRUE_ANSWER, 0.9),
    ]
    assert plurality(votes) == canonicalize(BYZANTINE_LIE)
    assert weighted_consensus(votes) == canonicalize(TRUE_ANSWER)


def test_weak_majority_is_escalated_instead_of_accepted():
    assert weighted_consensus([(TRUE_ANSWER, 0.5), (BYZANTINE_LIE, 0.5)]) is None


def test_zero_confidence_everywhere_gives_no_decision():
    assert weighted_consensus([(TRUE_ANSWER, 0.0), (BYZANTINE_LIE, 0.0)]) is None


def test_a_higher_threshold_refuses_what_a_lower_one_accepted():
    votes = [(TRUE_ANSWER, 0.6), (BYZANTINE_LIE, 0.4)]
    assert weighted_consensus(votes, 0.5) == canonicalize(TRUE_ANSWER)
    assert weighted_consensus(votes, 0.67) is None


# --------------------------------------------------------- geometric_median
def test_geometric_median_ignores_a_single_outlier():
    sample = [1.0, 2.0, 3.0, 4.0, 100.0]
    mean = sum(sample) / len(sample)
    assert geometric_median(sample) == pytest.approx(3.0, abs=0.05)
    assert mean == APPROX(22.0)


def test_geometric_median_of_identical_values_is_that_value():
    """Ловушка: оценка совпадает с точкой, d = 0, наивная формула падает."""
    assert geometric_median([5.0, 5.0, 5.0]) == pytest.approx(5.0, abs=1e-6)


def test_geometric_median_of_a_symmetric_sample_is_its_middle():
    assert geometric_median([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0, abs=1e-3)


def test_geometric_median_of_an_empty_sample_raises():
    with pytest.raises(ValueError):
        geometric_median([])


# ------------------------------------------------------------- simulate_bft
def test_bft_survives_the_maximum_number_of_traitors():
    for n in (4, 7, 10):
        result = simulate_bft(n, max_faulty(n), random.Random(1), trials=50)
        assert result["correct"] == APPROX(1.0)


def test_bft_stops_deciding_with_one_traitor_too_many():
    for n in (4, 7, 10):
        result = simulate_bft(n, max_faulty(n) + 1, random.Random(1), trials=50)
        assert result["no_decision"] == APPROX(1.0)


def test_bft_never_commits_the_lie_even_when_it_stalls():
    """Safety держится, ломается liveness: протокол молчит, а не врёт."""
    for n in (4, 7, 10):
        result = simulate_bft(n, max_faulty(n) + 1, random.Random(2), trials=50)
        assert result["wrong"] == APPROX(0.0)


def test_noisy_honest_agents_lower_the_agreement_rate():
    """Честный LLM тоже стохастичен — кворум перестаёт собираться сам собой."""
    clean = simulate_bft(7, 2, random.Random(3), trials=200, honest_noise=0.0)
    noisy = simulate_bft(7, 2, random.Random(3), trials=200, honest_noise=0.5)
    assert noisy["correct"] < clean["correct"]


def test_simulate_bft_rejects_more_traitors_than_nodes():
    with pytest.raises(ValueError):
        simulate_bft(4, 5, random.Random(0), trials=1)
