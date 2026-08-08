"""Тесты к уроку «Tree of Thoughts и LATS: рассуждение как поиск». Правь exercise.py."""

import itertools
import random

import pytest

from exercise import (
    backpropagate,
    beam_search,
    expand,
    make_node,
    mcts,
    select_path,
    uct,
    value,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def _two_branch_root(good_first=True):
    """Корень с двумя ветками: из (24, 1) до цели рукой подать, из (2, 3) — нет."""
    root = make_node((99.0,))
    good = make_node((24.0, 1.0), ("good",))
    bad = make_node((2.0, 3.0), ("bad",))
    root["children"] = [good, bad] if good_first else [bad, good]
    return root, good, bad


# ------------------------------------------------------------------ make_node
def test_make_node_starts_with_empty_statistics():
    node = make_node((6.0, 4.0))
    assert node["visits"] == 0
    assert node["value_sum"] == APPROX(0.0)
    assert node["children"] == []


def test_make_node_gives_every_node_its_own_children_list():
    """Общий список по умолчанию склеил бы всё дерево в один узел."""
    a, b = make_node((1.0,)), make_node((2.0,))
    a["children"].append(b)
    assert b["children"] == []


def test_make_node_keeps_the_trace_that_led_to_the_state():
    node = make_node((24.0,), ("6.0*4.0=24.0",))
    assert tuple(node["trace"]) == ("6.0*4.0=24.0",)


# --------------------------------------------------------------------- expand
def test_expand_includes_both_orders_for_non_commutative_operations():
    children = expand(make_node((6.0, 4.0)))
    assert len(children) == 6
    assert {child["trace"][-1] for child in children} >= {
        "6.0-4.0=2.0",
        "4.0-6.0=-2.0",
        "6.0/4.0=1.5",
        "4.0/6.0=0.6666666666666666",
    }


def test_expand_of_a_finished_state_has_no_children():
    assert expand(make_node((24.0,))) == []


def test_expand_skips_division_by_zero_instead_of_crashing():
    states = [tuple(ch["state"]) for ch in expand(make_node((5.0, 0.0)))]
    assert len(states) == 5


def test_expand_records_the_step_in_the_child_trace():
    child = expand(make_node((6.0, 4.0)))[0]
    assert len(child["trace"]) == 1
    assert "6.0" in child["trace"][0]


def test_expand_normalizes_every_child_state_to_descending_order():
    """Без нормализации (24, 1) и (1, 24) — разные узлы, и дерево удваивается."""
    for child in expand(make_node((6.0, 4.0, 1.0))):
        assert list(child["state"]) == sorted(child["state"], reverse=True)


# ---------------------------------------------------------------------- value
def test_value_of_an_exact_hit_is_one():
    assert value(make_node((24.0,))) == APPROX(1.0)


def test_value_of_a_miss_is_negative_and_grows_with_the_gap():
    assert value(make_node((20.0,))) < 0
    assert value(make_node((20.0,))) > value(make_node((10.0,)))


def test_value_of_a_partial_state_uses_one_step_lookahead():
    assert value(make_node((23.0, 5.0))) == APPROX(-0.04)


def test_value_does_not_reward_an_unfinished_state_that_contains_target():
    """В Game of 24 надо израсходовать все числа: (24, 4) — тупик."""
    assert value(make_node((24.0, 4.0))) < 0.0


def test_value_does_not_depend_on_the_order_inside_the_state():
    assert value(make_node((23.0, 5.0))) == APPROX(value(make_node((5.0, 23.0))))


# ------------------------------------------------------------------------ uct
def test_uct_of_an_unvisited_child_is_infinite():
    assert uct(10, make_node((1.0,))) == float("inf")


def test_uct_without_exploration_is_just_the_mean_value():
    child = make_node((1.0,))
    backpropagate([child], 1.0)
    backpropagate([child], 0.0)
    assert uct(10, child, c=0.0) == APPROX(0.5)


def test_uct_prefers_the_higher_scoring_child_at_equal_visits():
    good, bad = make_node((1.0,)), make_node((2.0,))
    backpropagate([good], 1.0)
    backpropagate([bad], 0.0)
    assert uct(2, good, c=1.4) > uct(2, bad, c=1.4)


def test_uct_exploration_term_favours_the_rarely_visited_child():
    often, rare = make_node((1.0,)), make_node((2.0,))
    for _ in range(20):
        backpropagate([often], 0.5)
    backpropagate([rare], 0.5)
    assert uct(21, rare, c=1.4) > uct(21, often, c=1.4)


# ---------------------------------------------------------------- select_path
def test_select_path_of_a_childless_node_is_just_that_node():
    root = make_node((6.0, 4.0))
    assert select_path(root) == [root]


def test_select_path_visits_an_unexplored_child_first():
    root, good, bad = _two_branch_root()
    backpropagate([root, good], 1.0)
    assert select_path(root)[-1] is bad


def test_select_path_follows_the_better_child_when_exploration_is_off():
    root, good, bad = _two_branch_root()
    backpropagate([root, good], 1.0)
    backpropagate([root, bad], 0.0)
    assert select_path(root, c=0.0)[-1] is good


def test_select_path_returns_the_whole_path_from_the_root_down():
    root, good, _ = _two_branch_root()
    backpropagate([root, good], 1.0)
    path = select_path(root, c=0.0)
    assert path[0] is root
    assert len(path) >= 2


# --------------------------------------------------------------- backpropagate
def test_backpropagate_counts_the_visit_and_adds_the_reward():
    node = make_node((1.0,))
    backpropagate([node], 1.0)
    assert node["visits"] == 1
    assert node["value_sum"] == APPROX(1.0)


def test_backpropagate_keeps_the_mean_not_the_last_reward():
    node = make_node((1.0,))
    backpropagate([node], 1.0)
    backpropagate([node], 0.0)
    assert node["value_sum"] / node["visits"] == APPROX(0.5)


def test_backpropagate_updates_every_ancestor_on_the_path():
    root, good, bad = _two_branch_root()
    backpropagate([root, good], 1.0)
    assert root["visits"] == 1
    assert good["visits"] == 1
    assert bad["visits"] == 0


def test_backpropagate_reports_nothing_and_edits_in_place():
    node = make_node((1.0,))
    assert backpropagate([node], 1.0) is None
    assert node["visits"] == 1


# ----------------------------------------------------------------- beam_search
def test_beam_search_finds_an_exact_solution():
    best, _ = beam_search(make_node((8.0, 3.0, 1.0, 1.0)))
    assert value(best) == APPROX(1.0)


def test_beam_search_result_does_not_depend_on_the_input_order():
    """Ветка с лучшей оценкой выбирается при любом порядке обхода."""
    straight, _ = beam_search(make_node((8.0, 3.0, 1.0, 1.0)))
    shuffled, _ = beam_search(make_node((1.0, 1.0, 3.0, 8.0)))
    assert value(straight) == APPROX(value(shuffled))


def test_beam_search_solves_reverse_division_case_in_every_input_order():
    """6 / (1 - 3 / 4) = 24 требует обеих обратных операций."""
    for numbers in itertools.permutations((1.0, 3.0, 4.0, 6.0)):
        best, _ = beam_search(make_node(numbers), width=50)
        assert value(best) == APPROX(1.0), numbers


def test_beam_search_never_returns_something_worse_than_the_root():
    root = make_node((2.0, 2.0, 2.0, 2.0))
    best, _ = beam_search(root)
    assert value(best) >= value(root)


def test_beam_search_counts_the_nodes_it_opened():
    _, expansions = beam_search(make_node((8.0, 3.0, 1.0, 1.0)))
    assert expansions > 0


def test_beam_search_width_bounds_what_it_pays_for():
    """Ширина луча — это и есть бюджет: узкий луч раскрывает меньше узлов."""
    _, narrow = beam_search(make_node((2.0, 2.0, 2.0, 2.0)), width=1)
    _, wide = beam_search(make_node((2.0, 2.0, 2.0, 2.0)), width=5)
    assert narrow < wide


# ----------------------------------------------------------------------- mcts
def test_mcts_picks_the_branch_that_leads_to_the_target():
    root, _, _ = _two_branch_root()
    best = mcts(root, iterations=40, rng=random.Random(0))
    assert tuple(best["trace"]) == ("good",)


def test_mcts_pick_does_not_depend_on_the_order_of_the_children():
    """Оценки поднимаются наверх backprop'ом, а не порядком обхода."""
    first, _, _ = _two_branch_root(good_first=True)
    second, _, _ = _two_branch_root(good_first=False)
    a = mcts(first, iterations=40, rng=random.Random(0))
    b = mcts(second, iterations=40, rng=random.Random(0))
    assert tuple(a["trace"]) == tuple(b["trace"]) == ("good",)


def test_mcts_spends_exactly_the_iteration_budget_on_the_root():
    root, _, _ = _two_branch_root()
    mcts(root, iterations=25, rng=random.Random(1))
    assert root["visits"] == 25


def test_mcts_is_reproducible_for_the_same_seed():
    root_a, good_a, _ = _two_branch_root()
    root_b, good_b, _ = _two_branch_root()
    mcts(root_a, iterations=30, rng=random.Random(7))
    mcts(root_b, iterations=30, rng=random.Random(7))
    assert good_a["visits"] == good_b["visits"]
    assert good_a["value_sum"] == APPROX(good_b["value_sum"])
