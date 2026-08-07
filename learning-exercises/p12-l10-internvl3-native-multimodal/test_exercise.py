"""Тесты к уроку «InternVL3: нативное мультимодальное предобучение». Правь exercise.py."""

import random

import pytest

from exercise import (
    NATIVE_MIX,
    TIER_TOKENS,
    alignment_debt,
    dvd_speedup,
    normalize_mix,
    route_resolution,
    routed_tokens,
    routing_speedup,
    sample_modalities,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Трафик из урока: 50% запросов с низкой детализацией, 30% средней, 20% высокой.
PRODUCTION_TRAFFIC = [0.1] * 5 + [0.5] * 3 + [0.9] * 2


# ------------------------------------------------------------- normalize_mix
def test_normalize_mix_sums_to_one():
    assert sum(normalize_mix({"text": 2, "video": 3, "caption": 5}).values()) == APPROX(
        1.0
    )


def test_normalize_mix_keeps_proportions():
    assert normalize_mix({"text": 40, "video": 10}) == {"text": 0.8, "video": 0.2}


def test_normalize_mix_leaves_an_already_normalized_mix_alone():
    assert normalize_mix(NATIVE_MIX) == pytest.approx(NATIVE_MIX, abs=1e-12)


def test_normalize_mix_rejects_a_zero_sum():
    """Без этой проверки деление на ноль случится посреди подготовки данных."""
    with pytest.raises(ValueError):
        normalize_mix({"text": 0, "video": 0})


def test_normalize_mix_rejects_a_negative_share():
    with pytest.raises(ValueError):
        normalize_mix({"text": 1.0, "video": -0.2})


# --------------------------------------------------------- sample_modalities
def test_sample_modalities_is_reproducible_for_the_same_seed():
    """Два прогона с одним seed обязаны увидеть одинаковый порядок данных."""
    a = sample_modalities(NATIVE_MIX, 50, random.Random(4))
    b = sample_modalities(NATIVE_MIX, 50, random.Random(4))
    assert a == b


def test_sample_modalities_differs_between_seeds():
    a = sample_modalities(NATIVE_MIX, 50, random.Random(4))
    b = sample_modalities(NATIVE_MIX, 50, random.Random(5))
    assert a != b


def test_sample_modalities_returns_the_requested_count():
    assert len(sample_modalities(NATIVE_MIX, 137, random.Random(0))) == 137


def test_sample_frequencies_converge_to_the_mix():
    draws = sample_modalities(NATIVE_MIX, 20000, random.Random(7))
    for name, share in NATIVE_MIX.items():
        assert draws.count(name) / 20000 == pytest.approx(share, abs=0.02)


def test_a_zero_weight_modality_never_appears():
    draws = sample_modalities({"text": 1.0, "video": 0.0}, 500, random.Random(1))
    assert "video" not in draws


def test_sample_modalities_rejects_a_negative_count():
    with pytest.raises(ValueError):
        sample_modalities(NATIVE_MIX, -1, random.Random(0))


# --------------------------------------------------------- route_resolution
def test_route_resolution_picks_a_tier_per_detail_level():
    assert route_resolution(0.1) == "low"
    assert route_resolution(0.5) == "medium"
    assert route_resolution(0.9) == "high"


def test_route_resolution_sends_the_boundary_to_the_higher_tier():
    """Недокодировать картинку дороже, чем потратить лишние токены."""
    assert route_resolution(0.4) == "medium"
    assert route_resolution(0.7) == "high"


def test_route_resolution_never_goes_down_as_detail_grows():
    order = {"low": 0, "medium": 1, "high": 2}
    tiers = [order[route_resolution(d / 20)] for d in range(21)]
    assert tiers == sorted(tiers)


def test_route_resolution_rejects_detail_outside_the_unit_interval():
    with pytest.raises(ValueError):
        route_resolution(1.5)


# ------------------------------------------------------------- routed_tokens
def test_routed_tokens_averages_the_tier_costs():
    assert routed_tokens([0.1, 0.9]) == APPROX(1152.0)


def test_routed_tokens_stays_between_the_cheapest_and_priciest_tier():
    cost = routed_tokens(PRODUCTION_TRAFFIC)
    assert min(TIER_TOKENS.values()) <= cost <= max(TIER_TOKENS.values())


def test_routed_tokens_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        routed_tokens([])


# ----------------------------------------------------------- routing_speedup
def test_routing_speedup_on_the_production_distribution():
    """Заявленные в уроке «2-3x пропускной способности при равном качестве»."""
    assert 2.0 < routing_speedup(PRODUCTION_TRAFFIC) < 3.0


def test_routing_speedup_is_one_when_every_query_needs_high_resolution():
    assert routing_speedup([0.95] * 10) == APPROX(1.0)


def test_routing_speedup_never_drops_below_one():
    for traffic in ([0.1], [0.9], PRODUCTION_TRAFFIC, [0.5] * 4):
        assert routing_speedup(traffic) >= 1.0


def test_routing_speedup_grows_as_traffic_gets_simpler():
    assert routing_speedup([0.1] * 9 + [0.9]) > routing_speedup([0.1] + [0.9] * 9)


def test_routing_speedup_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        routing_speedup([])


# --------------------------------------------------------------- dvd_speedup
def test_dvd_speedup_doubles_on_balanced_stages():
    assert dvd_speedup(50, 50) == APPROX(2.0)


def test_dvd_speedup_never_exceeds_two():
    """Конвейер из двух стадий физически не может дать больше двукратного."""
    for enc, llm in ((1, 1), (10, 90), (90, 10), (1, 999), (33, 67)):
        assert 1.0 <= dvd_speedup(enc, llm) <= 2.0


def test_dvd_speedup_approaches_one_when_stages_are_imbalanced():
    """Ответ на «когда DvD вредит»: когда разделять нечего."""
    assert dvd_speedup(1, 999) < 1.01


def test_dvd_speedup_is_symmetric_in_its_arguments():
    assert dvd_speedup(10, 90) == APPROX(dvd_speedup(90, 10))


def test_dvd_speedup_rejects_a_zero_stage():
    with pytest.raises(ValueError):
        dvd_speedup(0, 90)


# ------------------------------------------------------------ alignment_debt
def test_alignment_debt_normalizes_the_text_drop_by_vision_gain():
    assert alignment_debt(80.0, 74.0, 12.0) == APPROX(0.5)


def test_alignment_debt_is_zero_without_a_drop():
    assert alignment_debt(80.0, 80.0, 12.0) == APPROX(0.0)


def test_native_pretraining_carries_less_debt_than_post_hoc():
    """Проверяемая формулировка гипотезы alignment debt из урока."""
    native = alignment_debt(80.0, 79.5, 14.0)
    post_hoc = alignment_debt(80.0, 72.0, 12.0)
    assert native < post_hoc


def test_alignment_debt_is_negative_when_text_scores_improve():
    assert alignment_debt(70.0, 73.0, 10.0) < 0


def test_alignment_debt_rejects_a_non_positive_vision_gain():
    with pytest.raises(ValueError):
        alignment_debt(80.0, 74.0, 0.0)
