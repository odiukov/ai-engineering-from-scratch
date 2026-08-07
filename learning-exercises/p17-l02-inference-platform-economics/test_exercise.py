"""Тесты к уроку «Экономика inference-платформ». Правь exercise.py."""

import pytest

from exercise import (
    MINUTES_PER_DAY,
    VENDORS,
    NeverBreakEven,
    ZeroWorkload,
    blended_rate,
    cheapest_vendor,
    effective_rate_per_mtok,
    per_minute_cost,
    per_prediction_cost,
    per_token_cost,
    selfhosted_breakeven_requests,
    utilization_breakeven,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ---------------------------------------------------------- per_token_cost
def test_per_token_cost_charges_per_million():
    assert per_token_cost(2_000_000, 0.90) == APPROX(1.8)


def test_per_token_cost_of_idle_is_zero():
    """Главное свойство модели: за простой не платят вообще."""
    assert per_token_cost(0, 0.90) == APPROX(0.0)


def test_per_token_cost_is_linear():
    assert per_token_cost(50_000_000, 0.88) == APPROX(10 * per_token_cost(5_000_000, 0.88))


# --------------------------------------------------------- per_minute_cost
def test_per_minute_cost_is_dominated_by_the_reserved_floor_at_low_volume():
    assert per_minute_cost(2_000_000, 900_000, 0.55, 1440) == APPROX(792.0)


def test_per_minute_cost_with_a_small_floor_is_much_cheaper_at_the_same_traffic():
    """Тот же трафик, пол 60 минут вместо суток — счёт падает с $792 до $28.80."""
    assert per_minute_cost(2_000_000, 800_000, 0.48, 60) == APPROX(28.8)


def test_per_minute_cost_follows_traffic_once_it_clears_the_floor():
    """125 минут насыщенной работы против пола в 60 — платим за работу."""
    assert per_minute_cost(100_000_000, 800_000, 0.48, 60) == APPROX(60.0)


def test_per_minute_cost_takes_the_max_not_the_sum():
    """Ловушка: резерв не прибавляется к отработанным минутам, он их поглощает."""
    worked = 100_000_000 / 800_000
    assert per_minute_cost(100_000_000, 800_000, 0.48, 60) < (worked + 60) * 0.48


def test_per_minute_cost_of_zero_traffic_still_bills_the_floor():
    assert per_minute_cost(0, 900_000, 0.55, 1440) == APPROX(792.0)


# ----------------------------------------------------- per_prediction_cost
def test_per_prediction_cost_counts_calls_not_tokens():
    assert per_prediction_cost(10_000, 0.006) == APPROX(60.0)


# ------------------------------------------------- effective_rate_per_mtok
def test_effective_rate_normalizes_a_flat_bill_to_dollars_per_mtok():
    assert effective_rate_per_mtok(792.0, 2_000_000) == APPROX(396.0)


def test_effective_rate_falls_as_the_same_bill_covers_more_work():
    """Один и тот же счёт Baseten: $396/M на 2M токенов и $7.92/M на 100M."""
    assert effective_rate_per_mtok(792.0, 100_000_000) == APPROX(7.92)


def test_effective_rate_refuses_a_zero_workload():
    with pytest.raises(ZeroWorkload):
        effective_rate_per_mtok(792.0, 0)


# --------------------------------------------------- utilization_breakeven
def test_utilization_breakeven_for_a_full_day_reservation():
    """Baseten против Fireworks: $792 резерва против $1166.40 полной загрузки."""
    assert utilization_breakeven(0.90, 900_000, 0.55, 1440) == pytest.approx(
        792.0 / 1166.4, abs=1e-9
    )


def test_utilization_breakeven_collapses_when_the_floor_is_small():
    """Modal с полом в час окупается уже на 2.8% загрузки — пол важнее ставки."""
    assert utilization_breakeven(0.90, 800_000, 0.48, 60) == pytest.approx(
        28.8 / 1036.8, abs=1e-9
    )


def test_utilization_breakeven_is_the_point_where_both_bills_match():
    u = utilization_breakeven(0.90, 900_000, 0.55, 1440)
    tokens = 900_000 * MINUTES_PER_DAY * u
    assert per_token_cost(tokens, 0.90) == pytest.approx(
        per_minute_cost(tokens, 900_000, 0.55, 1440), abs=1e-6
    )


def test_utilization_breakeven_rises_with_the_reserved_floor():
    small = utilization_breakeven(0.90, 900_000, 0.55, 120)
    large = utilization_breakeven(0.90, 900_000, 0.55, 1440)
    assert small < large


def test_utilization_breakeven_raises_when_per_minute_never_wins():
    """$2.00/мин при 900k токенов/мин дороже per-token даже при полной загрузке."""
    with pytest.raises(NeverBreakEven):
        utilization_breakeven(0.90, 900_000, 2.00, 1440)


# ------------------------------------------------------------ blended_rate
def test_blended_rate_weights_the_discount_by_share():
    assert blended_rate(0.90, 0.4, 0.5) == APPROX(0.72)


def test_blended_rate_without_batch_traffic_is_the_base_rate():
    assert blended_rate(0.90, 0.0, 0.5) == APPROX(0.90)


def test_blended_rate_with_all_traffic_in_batch_is_the_full_discount():
    assert blended_rate(0.90, 1.0, 0.5) == APPROX(0.45)


def test_blended_rate_is_not_the_discount_subtracted_from_the_share():
    """Ловушка: 50% скидки на 40% трафика — это минус 20%, а не минус 40 или 50."""
    assert blended_rate(1.0, 0.4, 0.5) == APPROX(0.8)


def test_blended_rate_rejects_a_share_outside_zero_one():
    with pytest.raises(ValueError):
        blended_rate(0.90, 40, 0.5)


# ------------------------------------------ selfhosted_breakeven_requests
def test_selfhosted_breakeven_at_a_typical_fixed_cost():
    assert selfhosted_breakeven_requests(0.002, 2000.0, 0.0005) == 1_333_334


def test_selfhosted_breakeven_is_strict_at_an_exact_tie():
    """3000/0.003 — ровно миллион, но на миллионе счета равны; порог на единицу выше."""
    assert selfhosted_breakeven_requests(0.004, 3000.0, 0.001) == 1_000_001


def test_selfhosted_breakeven_below_the_threshold_managed_is_cheaper():
    n = selfhosted_breakeven_requests(0.002, 2000.0, 0.0005)
    below = n - 1
    assert 2000.0 + below * 0.0005 >= below * 0.002


def test_selfhosted_breakeven_at_the_threshold_self_hosted_wins():
    n = selfhosted_breakeven_requests(0.002, 2000.0, 0.0005)
    assert 2000.0 + n * 0.0005 < n * 0.002


def test_selfhosted_never_breaks_even_when_its_variable_cost_is_higher():
    with pytest.raises(NeverBreakEven):
        selfhosted_breakeven_requests(0.002, 2000.0, 0.003)


# --------------------------------------------------------- cheapest_vendor
def test_cheapest_vendor_at_startup_volume_is_a_per_token_provider():
    name, cost = cheapest_vendor(VENDORS, 2_000_000, 10_000)
    assert name == "together"
    assert cost == APPROX(1.76)


def test_cheapest_vendor_at_production_volume_flips_to_per_minute():
    """Тот же каталог, объём в 50 раз выше — побеждает поминутный Modal."""
    name, cost = cheapest_vendor(VENDORS, 100_000_000, 500_000)
    assert name == "modal"
    assert cost == APPROX(60.0)


def test_cheapest_vendor_is_deterministic_on_a_price_tie():
    twins = {
        "zeta": {"per_mtok": 1.0, "per_minute": None, "per_prediction": None,
                 "tokens_per_minute": 1_000, "reserved_minutes_per_day": 0},
        "alpha": {"per_mtok": 1.0, "per_minute": None, "per_prediction": None,
                  "tokens_per_minute": 1_000, "reserved_minutes_per_day": 0},
    }
    assert cheapest_vendor(twins, 1_000_000, 0)[0] == "alpha"
