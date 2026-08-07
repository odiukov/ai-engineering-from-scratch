"""Тесты к уроку «Переговоры и торг». Правь exercise.py."""

import random

import pytest

from exercise import (
    PERSONAS,
    REASON_LOST,
    REASON_OVER_RESERVE,
    accepts,
    bargain,
    concede,
    contract_net,
    deal_rate,
    naive_bargain,
    narrate,
    zopa,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

BIDS = [
    {"bidder": "A", "price": 90.0, "quality": 0.5},
    {"bidder": "B", "price": 70.0, "quality": 0.9},
    {"bidder": "C", "price": 130.0, "quality": 0.99},
]


# --------------------------------------------------------------------- zopa
def test_zopa_is_the_overlap_of_both_reservations():
    assert zopa(100.0, 60.0) == (60.0, 100.0)


def test_no_zopa_when_the_buyer_cannot_reach_the_seller():
    assert zopa(80.0, 90.0) is None


def test_a_single_point_zopa_still_allows_a_deal():
    assert zopa(90.0, 90.0) == (90.0, 90.0)


def test_zopa_bounds_always_come_out_low_then_high():
    rng = random.Random(11)
    for _ in range(200):
        z = zopa(rng.uniform(50, 150), rng.uniform(50, 150))
        assert z is None or z[0] <= z[1]


# ------------------------------------------------------------------ concede
def test_concede_moves_the_buyer_up_toward_his_ceiling():
    assert concede(60.0, 100.0, 0.3) == APPROX(72.0)


def test_concede_moves_the_seller_down_toward_his_floor():
    assert concede(84.0, 60.0, 0.3) == APPROX(76.8)


def test_a_zero_rate_means_standing_still():
    assert concede(60.0, 100.0, 0.0) == APPROX(60.0)


def test_repeated_concession_never_crosses_the_reservation():
    """Остаток гасится геометрически — резерв достигается только в пределе."""
    position = 60.0
    for _ in range(20):
        position = concede(position, 100.0, 0.3)
        assert position < 100.0


# ------------------------------------------------------------------ accepts
def test_seller_accepts_an_offer_that_beats_his_own_next_move():
    assert accepts("seller", 72.0, 60.0, 71.76) is True


def test_buyer_refuses_an_offer_worse_than_what_he_was_about_to_propose():
    assert accepts("buyer", 76.8, 100.0, 72.0) is False


def test_the_break_point_holds_even_against_a_generous_next_move():
    """Точка разрыва жёстче любой тактики: за резерв не идут никогда."""
    assert accepts("buyer", 120.0, 100.0, 130.0) is False
    assert accepts("seller", 50.0, 60.0, 40.0) is False


def test_accepts_rejects_an_unknown_side():
    with pytest.raises(ValueError):
        accepts("mediator", 72.0, 60.0, 71.0)


# ------------------------------------------------------------------ bargain
def test_bargain_closes_inside_the_zopa():
    price = bargain(100.0, 60.0)
    low, high = zopa(100.0, 60.0)
    assert low <= price <= high


def test_without_a_zopa_there_is_no_deal():
    assert bargain(80.0, 90.0) is None


def test_a_narrow_zopa_needs_more_rounds_than_it_was_given():
    """ZOPA есть, но уступки гасятся геометрически и не успевают сойтись."""
    assert zopa(100.0, 99.9) is not None
    assert bargain(100.0, 99.9) is None
    assert bargain(100.0, 99.9, max_rounds=40) is not None


def test_every_closed_deal_lands_inside_the_zopa():
    rng = random.Random(5)
    for _ in range(200):
        buyer_max, seller_min = rng.uniform(50, 150), rng.uniform(50, 150)
        price = bargain(buyer_max, seller_min)
        if price is not None:
            assert seller_min <= price <= buyer_max


def test_a_deal_implies_the_zopa_was_not_empty():
    rng = random.Random(6)
    for _ in range(200):
        buyer_max, seller_min = rng.uniform(50, 150), rng.uniform(50, 150)
        if bargain(buyer_max, seller_min) is not None:
            assert zopa(buyer_max, seller_min) is not None


# ------------------------------------------------------------ naive_bargain
def test_naive_bargain_also_respects_both_break_points():
    rng = random.Random(7)
    for _ in range(200):
        buyer_max, seller_min = rng.uniform(50, 150), rng.uniform(50, 150)
        price = naive_bargain(buyer_max, seller_min, rng)
        if price is not None:
            assert seller_min <= price <= buyer_max


def test_naive_bargain_is_reproducible_but_seed_dependent():
    same = naive_bargain(100.0, 60.0, random.Random(0))
    assert same == naive_bargain(100.0, 60.0, random.Random(0))
    outcomes = {naive_bargain(100.0, 60.0, random.Random(s)) for s in range(12)}
    assert len(outcomes) > 1


# ---------------------------------------------------------------- deal_rate
def test_structure_beats_improvisation_on_the_same_seed():
    """Протокол один и тот же, резервы одни и те же — разница только в оферте."""
    structured = deal_rate(bargain, 500, random.Random(0))
    improvised = deal_rate(naive_bargain, 500, random.Random(0))
    assert improvised < structured


def test_deal_rate_cannot_beat_the_share_of_trials_that_have_a_zopa():
    """Резервы независимы, значит ZOPA есть примерно в половине попыток."""
    assert deal_rate(bargain, 500, random.Random(1)) < 0.5


def test_deal_rate_is_reproducible_for_a_given_seed():
    assert deal_rate(bargain, 200, random.Random(2)) == deal_rate(
        bargain, 200, random.Random(2)
    )


def test_a_strategy_that_never_closes_has_rate_zero():
    assert deal_rate(lambda bm, sm, rng: None, 50, random.Random(3)) == APPROX(0.0)


# ------------------------------------------------------------------ narrate
def test_every_persona_carries_the_price_unchanged():
    assert all(f"{72.0:.2f}" in narrate(72.0, p) for p in PERSONAS)


def test_persona_changes_the_wording_not_the_number():
    assert narrate(72.0, "desperate") != narrate(72.0, "neutral")


def test_all_personas_produce_distinct_wording():
    assert len({narrate(72.0, p) for p in PERSONAS}) == len(PERSONAS)


def test_unknown_persona_raises():
    with pytest.raises(ValueError):
        narrate(72.0, "sarcastic")


# ------------------------------------------------------------- contract_net
def test_cheapest_rule_awards_the_lowest_affordable_bid():
    result = contract_net(BIDS, 100.0)
    assert (result["winner"], result["price"]) == ("B", 70.0)


def test_quality_rule_can_award_a_more_expensive_bid():
    assert contract_net(BIDS, 100.0, rule="best_quality")["winner"] == "B"
    assert contract_net(BIDS, 150.0, rule="best_quality")["winner"] == "C"


def test_a_bid_over_the_reserve_is_rejected_before_any_comparison():
    reasons = {r["bidder"]: r["reason"] for r in contract_net(BIDS, 100.0)["rejected"]}
    assert reasons == {"A": REASON_LOST, "C": REASON_OVER_RESERVE}


def test_nobody_wins_when_every_bid_is_over_the_reserve():
    result = contract_net(BIDS, 50.0)
    assert result["winner"] is None
    assert len(result["rejected"]) == len(BIDS)


def test_every_loser_is_told_why():
    result = contract_net(BIDS, 150.0)
    losers = {r["bidder"] for r in result["rejected"]}
    assert losers == {b["bidder"] for b in BIDS} - {result["winner"]}


def test_unknown_award_rule_raises():
    with pytest.raises(ValueError):
        contract_net(BIDS, 100.0, rule="random")
