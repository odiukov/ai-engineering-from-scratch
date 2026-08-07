"""Тесты к уроку «Экономики агентов: Шепли, аукционы, репутация». Правь exercise.py."""

import random

import pytest

from exercise import (
    best_bids,
    bidder_utility,
    marginal_contributions,
    reputation_weighted_pick,
    second_price_auction,
    shapley,
    shapley_sampled,
    update_reputation,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TEAM = ["coder", "researcher", "reviewer"]

# Ценность коалиции: пара «кодер + ревьюер» даёт готовый результат,
# исследователь добавляет четверть сверху, в одиночку никто не тянет.
PAIR = lambda s: 1.0 if {"coder", "reviewer"} <= s else 0.0
SIZE = lambda s: float(len(s))
BONUS = lambda s: 0.25 if "researcher" in s else 0.0
COMBINED = lambda s: PAIR(s) + BONUS(s)


# -------------------------------------------------- marginal_contributions
def test_marginal_contribution_depends_on_join_order():
    """Тот же агент в разном порядке приносит разное — отсюда усреднение у Шепли."""
    assert marginal_contributions(PAIR, ["coder", "reviewer"]) == {
        "coder": APPROX(0.0),
        "reviewer": APPROX(1.0),
    }
    assert marginal_contributions(PAIR, ["reviewer", "coder"]) == {
        "reviewer": APPROX(0.0),
        "coder": APPROX(1.0),
    }


def test_marginal_contributions_sum_to_the_full_coalition_value():
    got = marginal_contributions(COMBINED, TEAM)
    assert sum(got.values()) == APPROX(COMBINED(frozenset(TEAM)))


def test_marginal_contributions_cover_every_agent_once():
    assert sorted(marginal_contributions(SIZE, TEAM)) == sorted(TEAM)


def test_marginal_contributions_start_from_the_empty_coalition():
    """Отсчёт от v(пусто), а не от нуля: фиксированная стоимость запуска не делится."""
    fixed = lambda s: 10.0 + len(s)
    assert marginal_contributions(fixed, ["a", "b"]) == {"a": APPROX(1.0), "b": APPROX(1.0)}


# ----------------------------------------------------------------- shapley
def test_shapley_splits_a_symmetric_pair_evenly():
    assert shapley(PAIR, ["coder", "reviewer"]) == {
        "coder": APPROX(0.5),
        "reviewer": APPROX(0.5),
    }


def test_shapley_satisfies_the_efficiency_axiom():
    """Сумма долей равна ценности всей коалиции. Ничего не создано и не потеряно."""
    values = shapley(COMBINED, TEAM)
    assert sum(values.values()) == APPROX(COMBINED(frozenset(TEAM)))


def test_shapley_gives_a_null_player_nothing():
    """Агент, ничего не меняющий ни в одной коалиции, получает ровно ноль."""
    values = shapley(PAIR, TEAM)
    assert values["researcher"] == APPROX(0.0)


def test_shapley_is_symmetric_for_interchangeable_agents():
    values = shapley(SIZE, TEAM)
    assert values["coder"] == APPROX(values["researcher"]) == APPROX(values["reviewer"])


def test_shapley_is_linear_in_the_value_function():
    """v = v1 + v2 даёт shapley(v) = shapley(v1) + shapley(v2)."""
    a, b, c = shapley(PAIR, TEAM), shapley(BONUS, TEAM), shapley(COMBINED, TEAM)
    for agent in TEAM:
        assert c[agent] == APPROX(a[agent] + b[agent])


def test_shapley_rewards_the_agent_that_unlocks_the_bonus():
    values = shapley(COMBINED, TEAM)
    assert values["researcher"] == APPROX(0.25)


# --------------------------------------------------------- shapley_sampled
def test_shapley_sampled_approaches_the_exact_values():
    exact = shapley(COMBINED, TEAM)
    approx = shapley_sampled(COMBINED, TEAM, 4000, random.Random(0))
    for agent in TEAM:
        assert approx[agent] == pytest.approx(exact[agent], abs=0.03)


def test_shapley_sampled_is_reproducible_for_the_same_seed():
    a = shapley_sampled(COMBINED, TEAM, 200, random.Random(7))
    b = shapley_sampled(COMBINED, TEAM, 200, random.Random(7))
    assert a == {k: APPROX(v) for k, v in b.items()}


def test_shapley_sampled_keeps_efficiency_at_any_sample_count():
    """Эффективность держится на каждой перестановке, значит и на любом их числе."""
    approx = shapley_sampled(COMBINED, TEAM, 13, random.Random(1))
    assert sum(approx.values()) == APPROX(COMBINED(frozenset(TEAM)))


def test_shapley_sampled_does_not_shuffle_the_caller_list():
    """rng.shuffle правит список на месте — перемешивать надо копию."""
    agents = list(TEAM)
    shapley_sampled(COMBINED, agents, 50, random.Random(2))
    assert agents == TEAM


# --------------------------------------------------- second_price_auction
def test_second_price_auction_picks_the_top_bid_and_charges_the_runner_up():
    assert second_price_auction([0.82, 0.60, 0.95, 0.45, 0.77]) == (2, APPROX(0.82))


def test_second_price_auction_charges_nothing_without_competition():
    assert second_price_auction([0.5]) == (0, APPROX(0.0))


def test_second_price_auction_breaks_ties_toward_the_lower_index():
    assert second_price_auction([0.5, 0.5]) == (0, APPROX(0.5))


def test_second_price_auction_rejects_an_empty_bid_list():
    with pytest.raises(ValueError):
        second_price_auction([])


def test_second_price_price_never_exceeds_the_winning_bid():
    rng = random.Random(4)
    for _ in range(30):
        bids = [rng.uniform(0, 1) for _ in range(5)]
        winner, price = second_price_auction(bids)
        assert price <= bids[winner] + 1e-12


# ------------------------------------------------------------ bidder_utility
def test_bidder_utility_of_a_winner_is_value_minus_second_price():
    assert bidder_utility(0.9, [0.9, 0.5], 0) == APPROX(0.4)


def test_bidder_utility_goes_negative_when_overbidding_wins():
    """Проклятие победителя: выиграл по цене выше своей ценности."""
    assert bidder_utility(0.4, [0.9, 0.5], 0) == APPROX(-0.1)


def test_bidder_utility_of_a_loser_is_exactly_zero():
    """Проигравший не платит ничего — не «минус ставка»."""
    assert bidder_utility(0.9, [0.3, 0.5], 0) == APPROX(0.0)


def test_bidder_utility_does_not_depend_on_the_winning_bid_itself():
    """Цена определяется чужой ставкой: подняв свою, победитель платит столько же."""
    assert bidder_utility(0.9, [0.6, 0.5], 0) == APPROX(bidder_utility(0.9, [0.89, 0.5], 0))


# --------------------------------------------------------------- best_bids
def test_best_bids_include_the_truthful_bid():
    assert 0.7 in best_bids(0.7, [0.5, 0.3], [0.3, 0.7, 0.9])


def test_best_bids_are_not_unique_when_any_winning_bid_works():
    assert best_bids(0.7, [0.5, 0.3], [0.3, 0.7, 0.9]) == [0.7, 0.9]


def test_truthful_bidding_is_a_dominant_strategy():
    """Ключевое свойство второй цены: честная ставка не хуже НИКАКОЙ другой.

    Перебираем истинные ценности и произвольные чужие ставки. Если честная
    ставка хоть раз выпала из оптимума — механизм не правдивый.
    """
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = random.Random(5)
    for true_value in grid:
        for _ in range(20):
            others = [rng.uniform(0, 1) for _ in range(3)]
            assert true_value in best_bids(true_value, others, grid)


def test_underbidding_can_lose_a_profitable_auction():
    """Занижение не экономит: цена от него не зависит, а выигрыш теряется."""
    assert bidder_utility(0.9, [0.2, 0.5], 0) < bidder_utility(0.9, [0.9, 0.5], 0)


# --------------------------------------------------------- update_reputation
def test_update_reputation_moves_toward_the_new_quality():
    assert update_reputation(0.5, 1.0, 0.9) == APPROX(0.55)
    assert update_reputation(0.5, 0.0, 0.9) == APPROX(0.45)


def test_unverified_contribution_slashes_instead_of_smoothing():
    assert update_reputation(0.5, 1.0, 0.9, verified=False) == APPROX(0.2)


def test_reputation_never_goes_below_zero():
    """Без пола отрицательная репутация ломает взвешенную маршрутизацию."""
    assert update_reputation(0.1, 1.0, 0.9, verified=False) == APPROX(0.0)


def test_reputation_converges_to_a_constant_quality():
    rep = 0.0
    for _ in range(300):
        rep = update_reputation(rep, 0.8, 0.9)
    assert rep == pytest.approx(0.8, abs=1e-3)


def test_higher_alpha_makes_reputation_more_inertial():
    """alpha близко к 1 — старые заслуги держатся, один плохой прогон не топит."""
    slow = update_reputation(0.9, 0.0, 0.99)
    fast = update_reputation(0.9, 0.0, 0.5)
    assert slow > fast


# ------------------------------------------------- reputation_weighted_pick
def test_reputation_weighted_pick_ignores_zero_reputation_agents():
    for seed in range(5):
        assert reputation_weighted_pick([0.0, 1.0, 0.0], random.Random(seed)) == 1


def test_reputation_weighted_pick_is_proportional_to_reputation():
    rng = random.Random(3)
    picks = [reputation_weighted_pick([1.0, 3.0], rng) for _ in range(4000)]
    assert picks.count(1) / len(picks) == pytest.approx(0.75, abs=0.03)


def test_reputation_weighted_pick_is_reproducible_for_the_same_seed():
    a = [reputation_weighted_pick([1.0, 2.0, 3.0], random.Random(9)) for _ in range(1)]
    b = [reputation_weighted_pick([1.0, 2.0, 3.0], random.Random(9)) for _ in range(1)]
    assert a == b


def test_reputation_weighted_pick_rejects_an_all_zero_table():
    """Ловушка холодного старта: без стартового рейтинга выбирать не из кого."""
    with pytest.raises(ValueError):
        reputation_weighted_pick([0.0, 0.0], random.Random(0))


def test_reputation_weighted_pick_rejects_negative_reputation():
    with pytest.raises(ValueError):
        reputation_weighted_pick([1.0, -1.0], random.Random(0))
