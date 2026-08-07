"""Тесты к уроку «Голосование и топология дебатов». Правь exercise.py."""

import pytest

from exercise import (
    approval_winner,
    borda_scores,
    borda_winner,
    candidates,
    condorcet_cycle,
    condorcet_winner,
    pairwise_margin,
    plurality_winner,
)

# Профиль парадокса Кондорсе: три транзитивных бюллетеня, циклическое общество.
PARADOX = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]

# Профиль, где простое большинство и Кондорсе расходятся.
SPLIT = [
    ["A", "B", "C"],
    ["A", "B", "C"],
    ["C", "B", "A"],
    ["C", "B", "A"],
    ["B", "C", "A"],
]


# -------------------------------------------------------------- candidates
def test_candidates_are_sorted_and_deduplicated():
    assert candidates([["B", "A"], ["A", "B"]]) == ("A", "B")


def test_candidates_reject_an_election_without_ballots():
    with pytest.raises(ValueError):
        candidates([])


def test_candidates_reject_a_repeated_name_in_one_ballot():
    with pytest.raises(ValueError):
        candidates([["A", "A", "B"]])


def test_candidates_reject_ballots_over_different_sets():
    """Забытая строка в одном бюллетене тихо перекашивает весь подсчёт."""
    with pytest.raises(ValueError):
        candidates([["A", "B", "C"], ["A", "B"]])


# --------------------------------------------------------- plurality_winner
def test_plurality_counts_only_first_choices():
    assert plurality_winner([["A", "B"], ["A", "B"], ["B", "A"]]) == "A"


def test_plurality_ignores_everything_below_the_top_line():
    left = [["A", "B", "C"], ["A", "C", "B"], ["B", "C", "A"]]
    right = [["A", "C", "B"], ["A", "B", "C"], ["B", "A", "C"]]
    assert plurality_winner(left) == plurality_winner(right) == "A"


def test_plurality_breaks_ties_deterministically():
    assert plurality_winner([["B", "A"], ["A", "B"]]) == "A"


def test_plurality_can_miss_the_candidate_who_beats_everyone_one_on_one():
    assert plurality_winner(SPLIT) == "A"
    assert condorcet_winner(SPLIT) == "B"


# --------------------------------------------------------------- borda
def test_borda_awards_m_minus_one_points_for_first_place():
    assert borda_scores([["A", "B", "C"], ["B", "C", "A"]]) == {"A": 2, "B": 3, "C": 1}


def test_borda_total_depends_only_on_the_size_of_the_election():
    """Сумма очков фиксирована: голосующие перераспределяют, а не создают вес."""
    scores = borda_scores(SPLIT)
    m = len(candidates(SPLIT))
    assert sum(scores.values()) == len(SPLIT) * m * (m - 1) // 2


def test_borda_winner_of_a_unanimous_election():
    assert borda_winner([["A", "B", "C"]] * 4) == "A"


def test_borda_rewards_the_candidate_everyone_puts_second():
    """У B ни одного первого места у большинства — и всё равно он выигрывает."""
    assert borda_winner(SPLIT) == "B"


def test_borda_breaks_ties_deterministically():
    assert borda_winner([["A", "B"], ["B", "A"]]) == "A"


# ------------------------------------------------------------ approval
def test_approval_counts_every_approved_candidate():
    assert approval_winner([["A", "B"], ["B", "C"], ["B"]]) == "B"


def test_approval_ignores_a_repeated_approval_by_the_same_voter():
    assert approval_winner([["A", "A", "A"], ["B"], ["B"]]) == "B"


def test_approval_without_a_single_approval_has_no_winner():
    assert approval_winner([[], []]) is None


def test_approval_elects_the_broadly_acceptable_candidate():
    """Те же три агента: по первым местам ничья, по приемлемости — ясный B."""
    ranked = [["A", "B", "C"], ["C", "B", "A"], ["B", "A", "C"]]
    assert plurality_winner(ranked) == "A"
    assert approval_winner([["A", "B"], ["C", "B"], ["B", "A"]]) == "B"


# ------------------------------------------------------- pairwise_margin
def test_pairwise_margin_counts_who_is_ranked_higher():
    assert pairwise_margin([["A", "B"], ["A", "B"], ["B", "A"]], "A", "B") == 1


def test_pairwise_margin_is_antisymmetric():
    for a, b in (("A", "B"), ("B", "C"), ("A", "C")):
        assert pairwise_margin(PARADOX, a, b) == -pairwise_margin(PARADOX, b, a)


def test_pairwise_margin_of_a_candidate_against_itself_is_zero():
    assert pairwise_margin(PARADOX, "A", "A") == 0


def test_pairwise_margin_rejects_an_unknown_candidate():
    with pytest.raises(ValueError):
        pairwise_margin(PARADOX, "A", "Z")


# ------------------------------------------------------- condorcet_winner
def test_condorcet_winner_beats_every_rival_one_on_one():
    winner = condorcet_winner(SPLIT)
    others = [c for c in candidates(SPLIT) if c != winner]
    assert all(pairwise_margin(SPLIT, winner, o) > 0 for o in others)


def test_a_single_candidate_wins_by_default():
    assert condorcet_winner([["A"], ["A"]]) == "A"


def test_the_paradox_profile_has_no_condorcet_winner():
    assert condorcet_winner(PARADOX) is None


def test_unanimous_election_has_a_condorcet_winner():
    assert condorcet_winner([["A", "B", "C"]] * 3) == "A"


# -------------------------------------------------------- condorcet_cycle
def test_paradox_profile_has_a_cyclic_social_preference():
    assert condorcet_cycle(PARADOX) == ("A", "B", "C")


def test_individual_ballots_are_transitive_yet_the_group_is_not():
    """Каждый бюллетень — строгий порядок, а общество ходит по кругу."""
    x, y, z = condorcet_cycle(PARADOX)
    assert pairwise_margin(PARADOX, x, y) > 0
    assert pairwise_margin(PARADOX, y, z) > 0
    assert pairwise_margin(PARADOX, z, x) > 0
    assert all(len(set(b)) == len(b) == 3 for b in PARADOX)


def test_no_cycle_when_a_condorcet_winner_exists():
    assert condorcet_cycle(SPLIT) is None


def test_cycle_does_not_depend_on_the_order_of_ballots():
    shuffled = [PARADOX[2], PARADOX[0], PARADOX[1]]
    assert condorcet_cycle(shuffled) == condorcet_cycle(PARADOX)


def test_two_candidates_can_never_cycle():
    assert condorcet_cycle([["A", "B"], ["B", "A"]]) is None
