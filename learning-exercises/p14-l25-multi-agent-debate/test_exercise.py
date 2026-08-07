"""Тесты к уроку «Multi-agent debate». Правь exercise.py."""

import random

import pytest

from exercise import (
    HUB,
    TOPOLOGIES,
    collapsed_early,
    compare_topologies,
    critique_ops,
    debate_round,
    majority_answer,
    run_debate,
    topology_peers,
    weighted_answer,
)


# --------------------------------------------------------- majority_answer
def test_majority_picks_the_most_common_answer():
    assert majority_answer(["A", "B", "A"]) == "A"


def test_majority_breaks_ties_alphabetically():
    assert majority_answer(["B", "A"]) == "A"


def test_majority_is_independent_of_agent_order():
    """Перестановка агентов не должна переворачивать итог дебатов."""
    rng = random.Random(7)
    proposals = ["A", "B", "A", "C", "B", "A"]
    reference = majority_answer(proposals)
    for _ in range(20):
        shuffled = proposals[:]
        rng.shuffle(shuffled)
        assert majority_answer(shuffled) == reference


def test_majority_of_nothing_is_value_error():
    with pytest.raises(ValueError):
        majority_answer([])


# --------------------------------------------------------- weighted_answer
def test_weighted_lets_a_confident_minority_win():
    assert weighted_answer([("A", 0.25), ("A", 0.25), ("B", 0.75)]) == "B"


def test_weighted_matches_majority_when_confidence_is_equal():
    """Одинаковые веса вырождают взвешенное голосование в обычное."""
    proposals = ["A", "B", "A"]
    assert weighted_answer([(p, 1.0) for p in proposals]) == majority_answer(proposals)


def test_weighted_breaks_ties_alphabetically():
    assert weighted_answer([("B", 1.0), ("A", 1.0)]) == "A"


def test_weighted_rejects_negative_confidence():
    with pytest.raises(ValueError):
        weighted_answer([("A", 1.0), ("B", -2.0)])


# ---------------------------------------------------------- topology_peers
def test_full_mesh_connects_everyone():
    assert topology_peers("full_mesh", 3) == {0: [1, 2], 1: [0, 2], 2: [0, 1]}


def test_star_spokes_see_only_the_hub():
    peers = topology_peers("star", 4)
    assert peers[HUB] == [1, 2, 3]
    assert all(peers[i] == [HUB] for i in (1, 2, 3))


def test_ring_gives_each_debater_two_neighbours():
    peers = topology_peers("ring", 5)
    assert all(len(p) == 2 for p in peers.values())
    assert peers[0] == [1, 4]


def test_unknown_topology_is_value_error():
    with pytest.raises(ValueError):
        topology_peers("mesh-ish", 3)


def test_debate_needs_at_least_two_agents():
    with pytest.raises(ValueError):
        topology_peers("full_mesh", 1)


# ------------------------------------------------------------ critique_ops
def test_full_mesh_cost_is_quadratic_in_agents():
    assert critique_ops(topology_peers("full_mesh", 5), 3) == 60


def test_sparse_topology_is_cheaper_than_full_mesh():
    """Ради этого вывода урок и вводит разреженные топологии."""
    star = critique_ops(topology_peers("star", 5), 3)
    mesh = critique_ops(topology_peers("full_mesh", 5), 3)
    assert star < mesh


def test_zero_rounds_cost_nothing():
    assert critique_ops(topology_peers("full_mesh", 9), 0) == 0


# ------------------------------------------------------------ debate_round
def test_round_pulls_the_minority_to_the_majority():
    assert debate_round(["A", "B", "B"], topology_peers("full_mesh", 3)) == ["B", "B", "B"]


def test_round_updates_simultaneously_not_in_place():
    """Все читают СТАРЫЕ мнения.

    На звезде из четырёх агентов обновление на месте дало бы ["B","B","B","B"]:
    спицы увидели бы уже переубеждённый хаб. Одновременное обновление даёт
    перехлёст — хаб уходит к B, а спицы в тот же миг уходят к A.
    """
    assert debate_round(["A", "B", "B", "B"], topology_peers("star", 4)) == [
        "B",
        "A",
        "A",
        "A",
    ]


def test_round_does_not_mutate_the_input():
    opinions = ["A", "B", "B"]
    debate_round(opinions, topology_peers("full_mesh", 3))
    assert opinions == ["A", "B", "B"]


def test_stubborn_debater_keeps_its_answer():
    result = debate_round(["A", "B", "B"], topology_peers("full_mesh", 3), stubborn=(0,))
    assert result[0] == "A"


# --------------------------------------------------------------- run_debate
def test_debate_converges_on_full_mesh():
    result = run_debate(["A", "A", "B"], topology_peers("full_mesh", 3), 3)
    assert result["converged"] is True
    assert result["answer"] == "A"


def test_debate_stops_early_and_stops_paying():
    """Сошлись за раунд — платим за раунд, а не за все три."""
    result = run_debate(["A", "A", "B"], topology_peers("full_mesh", 3), 3)
    assert result["rounds_used"] == 1
    assert result["ops"] == 6


def test_unanimous_start_needs_no_rounds():
    result = run_debate(["A", "A"], topology_peers("full_mesh", 2), 5)
    assert (result["rounds_used"], result["ops"], result["converged"]) == (0, 0, True)


def test_debate_answer_does_not_depend_on_agent_order():
    """Полная сетка симметрична: перестановка агентов итог менять не должна."""
    rng = random.Random(11)
    opinions = ["A", "B", "A", "C", "A"]
    peers = topology_peers("full_mesh", 5)
    reference = run_debate(opinions, peers, 4)["answer"]
    for _ in range(15):
        shuffled = opinions[:]
        rng.shuffle(shuffled)
        assert run_debate(shuffled, peers, 4)["answer"] == reference


def test_history_starts_with_the_initial_opinions():
    opinions = ["A", "B", "B"]
    result = run_debate(opinions, topology_peers("full_mesh", 3), 3)
    assert result["history"][0] == opinions
    assert len(result["history"]) == result["rounds_used"] + 1


# ---------------------------------------------------------- collapsed_early
def test_collapse_flags_instant_unanimity():
    history = run_debate(["A", "B", "B"], topology_peers("full_mesh", 3), 3)["history"]
    assert collapsed_early(history) is True


def test_no_collapse_when_disagreement_survives_a_round():
    """Кольцо из пяти на 3:2 застревает — разногласие переживает раунд."""
    history = run_debate(["A", "A", "A", "B", "B"], topology_peers("ring", 5), 3)["history"]
    assert collapsed_early(history) is False


def test_unanimous_start_is_not_a_collapse():
    history = run_debate(["A", "A", "A"], topology_peers("full_mesh", 3), 3)["history"]
    assert collapsed_early(history) is False


# ------------------------------------------------------- compare_topologies
def test_sparse_matches_full_mesh_at_lower_cost():
    """Главный вывод урока: тот же ответ втрое дешевле."""
    report = compare_topologies(["A", "A", "A", "B", "B"], 3, "A")
    assert report["star"]["answer"] == report["full_mesh"]["answer"]
    assert report["star"]["ops"] < report["full_mesh"]["ops"]


def test_report_covers_every_topology():
    report = compare_topologies(["A", "B", "A", "B"], 2, "A")
    assert sorted(report) == sorted(TOPOLOGIES)


def test_correct_flag_follows_the_truth():
    report = compare_topologies(["A", "A", "A", "B", "B"], 3, "B")
    assert all(entry["correct"] is False for entry in report.values())


def test_ring_can_stall_without_converging():
    """Кольцо не всесильно: локальное большинство бывает устойчивым."""
    report = compare_topologies(["A", "A", "A", "B", "B"], 3, "A")
    assert report["ring"]["converged"] is False
    assert report["full_mesh"]["converged"] is True
