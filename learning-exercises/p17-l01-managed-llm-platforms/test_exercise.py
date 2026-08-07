"""Тесты к уроку «Managed LLM платформы: on-demand, PTU и выбор под SLA». Правь exercise.py."""

import pytest

from exercise import (
    DAYS_PER_MONTH,
    PLATFORMS,
    SLAUnreachable,
    cheapest_path,
    ondemand_cost,
    pick_platform,
    ptu_breakeven_utilization,
    ptu_cost,
    ptu_units_needed,
    redundancy_uplift,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ----------------------------------------------------------- ondemand_cost
def test_ondemand_cost_sums_input_and_output_at_their_own_rates():
    assert ondemand_cost(3_000_000, 1_000_000, 3.0, 15.0) == APPROX(24.0)


def test_ondemand_cost_of_empty_workload_is_zero():
    assert ondemand_cost(0, 0, 3.0, 15.0) == APPROX(0.0)


def test_ondemand_cost_is_linear_in_volume():
    """Никаких порогов и минимумов: удвоил трафик — удвоил счёт."""
    one = ondemand_cost(1_000_000, 500_000, 2.5, 10.0)
    two = ondemand_cost(2_000_000, 1_000_000, 2.5, 10.0)
    assert two == APPROX(2 * one)


def test_ondemand_cost_divides_volume_not_price():
    """Ловушка единиц: 1M токенов по $3/M — это ровно $3, а не $3e-6 и не $3e6."""
    assert ondemand_cost(1_000_000, 0, 3.0, 15.0) == APPROX(3.0)


# -------------------------------------------------------- ptu_units_needed
def test_ptu_units_needed_fits_workload_into_one_unit():
    assert ptu_units_needed(45_000_000, 2_000_000, 24) == 1


def test_ptu_units_needed_rounds_up_to_a_whole_unit():
    """Половину PTU не продают: 50M при ёмкости 48M — это две единицы."""
    assert ptu_units_needed(50_000_000, 2_000_000, 24) == 2


def test_ptu_units_needed_is_at_least_one_even_without_traffic():
    assert ptu_units_needed(0, 2_000_000, 24) == 1


def test_ptu_units_needed_rejects_a_platform_without_ptu():
    with pytest.raises(ValueError):
        ptu_units_needed(1_000_000, 0, 24)


# ---------------------------------------------------------------- ptu_cost
def test_ptu_cost_of_one_unit_for_a_day():
    assert ptu_cost(45_000_000, 2_000_000, 10.0, 24) == APPROX(240.0)


def test_ptu_cost_does_not_fall_when_traffic_falls():
    """Ты платишь за резерв, а не за трафик — идле стоит столько же."""
    busy = ptu_cost(45_000_000, 2_000_000, 10.0, 24)
    idle = ptu_cost(1_000, 2_000_000, 10.0, 24)
    assert idle == APPROX(busy)


def test_ptu_cost_grows_stepwise_not_smoothly():
    """Цена скачет на границе единицы: 48M и 48M+1 токен стоят по-разному."""
    just_fits = ptu_cost(48_000_000, 2_000_000, 10.0, 24)
    one_over = ptu_cost(48_000_001, 2_000_000, 10.0, 24)
    assert just_fits == APPROX(240.0)
    assert one_over == APPROX(480.0)


# ----------------------------------------------- ptu_breakeven_utilization
def test_ptu_breakeven_at_half_utilization():
    assert ptu_breakeven_utilization(10.0, 2_000_000, 10.0) == APPROX(0.5)


def test_ptu_breakeven_above_one_means_never_pays_off():
    """Bedrock: $21/час за 1.2M токенов, которые on-demand стоят $18. Не окупится."""
    assert ptu_breakeven_utilization(21.0, 1_200_000, 15.0) > 1.0


def test_ptu_breakeven_falls_when_the_reservation_gets_cheaper():
    expensive = ptu_breakeven_utilization(20.0, 2_000_000, 10.0)
    cheap = ptu_breakeven_utilization(5.0, 2_000_000, 10.0)
    assert cheap < expensive


def test_ptu_breakeven_matches_the_direct_cost_comparison():
    """В точке безубыточности стоимость обоих путей за час совпадает."""
    u = ptu_breakeven_utilization(10.0, 2_000_000, 10.0)
    tokens = 2_000_000 * u
    assert ondemand_cost(0, tokens, 0.0, 10.0) == APPROX(10.0)


# ------------------------------------------------------------ cheapest_path
def test_cheapest_path_is_ondemand_when_the_platform_sells_no_ptu():
    assert cheapest_path(PLATFORMS["vertex"], 3_000_000, 1_000_000, 24) == (
        "on-demand",
        APPROX(8.75),
    )


def test_cheapest_path_keeps_ondemand_at_low_volume():
    path, cost = cheapest_path(PLATFORMS["azure"], 30_000_000, 15_000_000, 24)
    assert path == "on-demand"
    assert cost == APPROX(225.0)


def test_cheapest_path_switches_to_ptu_at_high_volume():
    """5 единиц по $10/час = $1200 против $1250 on-demand — резерв выиграл."""
    path, cost = cheapest_path(PLATFORMS["azure"], 100_000_000, 100_000_000, 24)
    assert path == "ptu"
    assert cost == APPROX(1200.0)


def test_cheapest_path_prefers_ondemand_on_a_tie():
    """Одинаковая цена — берём то, что не связывает обязательством."""
    platform = {
        "in_per_mtok": 0.0,
        "out_per_mtok": 10.0,
        "ptu_hourly": 10.0,
        "ptu_tokens_per_hour": 2_000_000,
        "ttft_p99_ondemand_ms": 100.0,
        "ttft_p99_ptu_ms": 50.0,
    }
    # 24M выходных токенов по $10/M = $240; одна PTU за сутки — тоже $240
    assert cheapest_path(platform, 0, 24_000_000, 24) == ("on-demand", APPROX(240.0))


# ----------------------------------------------------------- pick_platform
def test_pick_platform_takes_the_cheapest_that_meets_the_sla():
    assert pick_platform(PLATFORMS, 3_000_000, 1_000_000, 24, 200.0) == (
        "vertex",
        "on-demand",
        APPROX(8.75),
    )


def test_pick_platform_drops_those_that_miss_the_sla():
    """SLA 150 мс отсекает vertex (160) и bedrock (180), остаётся azure (140)."""
    name, path, _ = pick_platform(PLATFORMS, 3_000_000, 1_000_000, 24, 150.0)
    assert (name, path) == ("azure", "on-demand")


def test_pick_platform_judges_the_sla_on_the_chosen_path():
    """Под SLA 60 мс проходит только выделенная мощность — и только там, где она выбрана по цене."""
    name, path, cost = pick_platform(PLATFORMS, 100_000_000, 100_000_000, 24, 60.0)
    assert (name, path) == ("azure", "ptu")
    assert cost == APPROX(1200.0)


def test_pick_platform_raises_its_own_exception_when_nothing_fits():
    with pytest.raises(SLAUnreachable):
        pick_platform(PLATFORMS, 3_000_000, 1_000_000, 24, 10.0)


def test_pick_platform_is_deterministic_on_a_price_tie():
    """Два одинаковых по цене каталога — побеждает меньшее имя, а не порядок ключей."""
    twin = {
        "in_per_mtok": 1.0,
        "out_per_mtok": 1.0,
        "ptu_hourly": None,
        "ptu_tokens_per_hour": 0,
        "ttft_p99_ondemand_ms": 50.0,
        "ttft_p99_ptu_ms": 50.0,
    }
    catalog = {"zeta": dict(twin), "alpha": dict(twin)}
    assert pick_platform(catalog, 1_000_000, 0, 24, 100.0)[0] == "alpha"


# ------------------------------------------------------- redundancy_uplift
def test_redundancy_uplift_adds_gateway_and_headroom():
    assert redundancy_uplift(50.0, 3.0, 10.0) == (APPROX(6.5), APPROX(195.0))


def test_redundancy_uplift_is_zero_without_a_second_provider():
    assert redundancy_uplift(50.0, 0.0, 0.0) == (APPROX(0.0), APPROX(0.0))


def test_redundancy_uplift_monthly_is_the_daily_figure_times_the_month():
    daily, monthly = redundancy_uplift(123.45, 3.0, 10.0)
    assert monthly == APPROX(daily * DAYS_PER_MONTH)
