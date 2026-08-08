"""Тесты к уроку «FinOps для LLM: атрибуция, аномалии, прогноз». Правь exercise.py."""

import pytest

from exercise import (
    LAYERS,
    anomaly_days,
    attribute,
    call_cost,
    daily_totals,
    enforcement_action,
    forecast_month,
    layer_shares,
    zscore,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TRACE = {
    "trace_id": "abc123",
    "user_id": "u_42",
    "tenant_id": "t_7",
    "task_id": "task_classify_doc",
    "route": "haiku",
    "day": "2026-08-03",
    "layers": {"prompt": 1800, "tool": 600, "memory": 400, "response": 150},
}

POLICY = {
    "contracted_daily_usd": 100.0,
    "spend_cap_multiplier": 2.0,
    "kill_z": 4.0,
    "min_history": 5,
}


def august_2026(weekday_usd, weekend_usd, days):
    """Расход по дням августа 2026: 1 августа — суббота."""
    import datetime

    out = {}
    for d in range(1, days + 1):
        day = datetime.date(2026, 8, d)
        out[day.isoformat()] = weekday_usd if day.weekday() < 5 else weekend_usd
    return out


# ---------------------------------------------------------------- call_cost
def test_cost_of_the_reference_trace():
    assert call_cost(TRACE) == APPROX(0.00284)


def test_output_tokens_are_billed_at_the_expensive_rate():
    """Одинаковое число токенов на выходе стоит впятеро дороже, чем на входе."""
    inp = call_cost({"route": "haiku", "layers": {"prompt": 1000}})
    out = call_cost({"route": "haiku", "layers": {"response": 1000}})
    assert out == APPROX(inp * 5)


def test_a_missing_layer_counts_as_zero_tokens():
    """Вызов без tool-слоя — норма для не-агентских задач, а не KeyError."""
    assert call_cost({"route": "haiku", "layers": {"prompt": 1000}}) == APPROX(0.0008)


def test_cached_input_makes_the_input_ten_times_cheaper_and_output_unchanged():
    cached = call_cost(dict(TRACE, cached_input=True))
    assert cached == APPROX(0.000824)


def test_batch_takes_half_off_the_whole_call():
    assert call_cost(dict(TRACE, batch=True)) == APPROX(call_cost(TRACE) / 2)


def test_an_unknown_route_is_an_error_not_a_free_call():
    with pytest.raises(ValueError):
        call_cost({"route": "gpt-guess", "layers": {"prompt": 100}})


# ------------------------------------------------------------- layer_shares
def test_shares_of_the_four_layers_sum_to_one():
    shares = layer_shares([TRACE])
    assert sum(shares.values()) == APPROX(1.0)


def test_every_layer_key_is_present_even_when_never_used():
    """Пропадающая колонка в дашборде хуже, чем колонка нулей."""
    shares = layer_shares([{"layers": {"prompt": 100}}])
    assert sorted(shares) == sorted(LAYERS) and shares["tool"] == APPROX(0.0)


def test_shares_are_computed_over_all_calls_not_per_call():
    calls = [{"layers": {"prompt": 100}}, {"layers": {"response": 300}}]
    assert layer_shares(calls)["prompt"] == APPROX(0.25)


def test_no_calls_means_zeros_not_a_division_by_zero():
    assert layer_shares([]) == {layer: APPROX(0.0) for layer in LAYERS}


# ---------------------------------------------------------------- attribute
def test_the_same_calls_split_differently_per_dimension():
    calls = [
        dict(TRACE, user_id="u1", tenant_id="t1"),
        dict(TRACE, user_id="u2", tenant_id="t1"),
    ]
    assert sorted(attribute(calls, "user_id")) == ["u1", "u2"]
    assert sorted(attribute(calls, "tenant_id")) == ["t1"]


def test_a_tenant_bucket_is_the_sum_of_its_calls():
    calls = [dict(TRACE, tenant_id="t1"), dict(TRACE, tenant_id="t1")]
    assert attribute(calls, "tenant_id")["t1"] == APPROX(call_cost(TRACE) * 2)


def test_untagged_calls_are_shown_and_not_silently_dropped():
    """Цена ретроактивного тегирования: расход есть, владельца у него нет."""
    calls = [dict(TRACE, tenant_id="t1"), {"route": "haiku", "layers": {"prompt": 1000}}]
    assert attribute(calls, "tenant_id")["untagged"] == APPROX(0.0008)


def test_attribution_conserves_the_total_bill():
    calls = [dict(TRACE, tenant_id="t1"), dict(TRACE, tenant_id="t2"), dict(TRACE)]
    total = sum(attribute(calls, "tenant_id").values())
    assert total == APPROX(call_cost(TRACE) * 3)


# ------------------------------------------------------------- daily_totals
def test_calls_of_one_day_land_in_one_bucket():
    calls = [dict(TRACE, day="2026-08-03"), dict(TRACE, day="2026-08-03")]
    assert daily_totals(calls) == {"2026-08-03": APPROX(call_cost(TRACE) * 2)}


def test_days_without_calls_do_not_appear_as_zeros():
    """Ноль расхода и отсутствие телеметрии — разные диагнозы."""
    calls = [dict(TRACE, day="2026-08-01"), dict(TRACE, day="2026-08-03")]
    assert sorted(daily_totals(calls)) == ["2026-08-01", "2026-08-03"]


# ------------------------------------------------------------------ zscore
def test_zscore_measures_distance_in_standard_deviations():
    assert zscore(100.0, [50.0, 52.0, 48.0, 50.0, 50.0]) == pytest.approx(35.355, abs=1e-3)


def test_a_flat_baseline_detects_any_nonzero_deviation():
    assert zscore(50.97, [50.0] * 5) == float("inf")
    assert zscore(49.99, [50.0] * 5) == float("-inf")
    assert zscore(50.0, [50.0] * 5) == APPROX(0.0)


def test_a_history_shorter_than_two_points_is_not_a_baseline():
    """Дисперсии ещё нет — значит и аномалий тоже, а не «всё аномально»."""
    assert zscore(999.0, [50.0]) == APPROX(0.0)
    assert zscore(999.0, []) == APPROX(0.0)


def test_spend_below_the_baseline_gives_a_negative_score():
    assert zscore(10.0, [50.0, 52.0, 48.0, 50.0, 50.0]) < 0


# ------------------------------------------------------------- anomaly_days
def test_a_steady_month_has_no_anomalies():
    assert anomaly_days({"2026-08-%02d" % d: 10.0 for d in range(1, 11)}) == []


def test_a_spike_after_a_stable_baseline_is_caught():
    daily = {"2026-08-%02d" % d: 10.0 + (d % 3) for d in range(1, 8)}
    daily["2026-08-08"] = 500.0
    assert anomaly_days(daily) == ["2026-08-08"]


def test_the_first_days_are_never_flagged_because_there_is_no_baseline_yet():
    daily = {"2026-08-01": 1000.0, "2026-08-02": 1.0, "2026-08-03": 1.0}
    assert anomaly_days(daily) == []


def test_a_relative_spike_on_pocket_change_can_be_muted_by_min_usd():
    """Скачок с $0.60 до $0.97 формально огромен — будить из-за него незачем."""
    daily = {"2026-08-%02d" % d: 0.60 + 0.01 * (d % 3) for d in range(1, 8)}
    daily["2026-08-08"] = 0.97
    assert anomaly_days(daily) != []
    assert anomaly_days(daily, min_usd=5.0) == []


def test_a_day_is_compared_only_with_the_days_before_it():
    """Детектор, подсматривающий в будущее, в проде не работает."""
    daily = {"2026-08-%02d" % d: 10.0 + (d % 3) for d in range(1, 8)}
    daily["2026-08-08"] = 900.0
    # 09-е такое же, но его база уже содержит скачок 08-го: разброс вырос,
    # и второй день той же величины аномалией больше не считается
    daily["2026-08-09"] = 900.0
    assert anomaly_days(daily) == ["2026-08-08"]


# ----------------------------------------------------------- forecast_month
def test_forecast_projects_weekdays_and_weekends_separately():
    daily = august_2026(weekday_usd=100.0, weekend_usd=10.0, days=14)
    assert forecast_month(daily, "2026-08-14", 31) == APPROX(2200.0)


def test_forecast_does_not_extrapolate_linearly():
    """Линейная экстраполяция дала бы 1040 / 14 * 31 = 2302.86."""
    daily = august_2026(weekday_usd=100.0, weekend_usd=10.0, days=14)
    linear = sum(daily.values()) / 14 * 31
    assert forecast_month(daily, "2026-08-14", 31) != pytest.approx(linear, abs=1.0)


def test_a_full_month_forecasts_exactly_what_it_spent():
    daily = august_2026(weekday_usd=100.0, weekend_usd=10.0, days=31)
    assert forecast_month(daily, "2026-08-31", 31) == APPROX(sum(daily.values()))


def test_with_only_one_kind_of_day_observed_the_forecast_uses_the_overall_mean():
    """Выдумывать поведение выходных не из чего — берём то, что есть."""
    # 3 и 4 августа 2026 — понедельник и вторник, выходных в наблюдениях нет.
    # Впереди 5-7 (будни) и 8-9 (суббота с воскресеньем): для выходных берётся
    # общее среднее, потому что своего у них ещё не накопилось.
    daily = {"2026-08-03": 100.0, "2026-08-04": 100.0}
    assert forecast_month(daily, "2026-08-04", 9) == APPROX(700.0)


# ------------------------------------------------------- enforcement_action
def test_spending_inside_the_contract_needs_no_action():
    assert enforcement_action(50.0, [], POLICY) == "ok"


def test_crossing_the_contract_tightens_the_rate_limit():
    assert enforcement_action(150.0, [], POLICY) == "rate_limit"


def test_crossing_the_daily_cap_raises_the_alert():
    assert enforcement_action(250.0, [], POLICY) == "cap_alert"


def test_a_twentyfold_blowup_pauses_the_tenant_instead_of_emailing_them():
    """Kill switch старше cap: сначала пауза, потом переписка."""
    history = [50.0, 52.0, 48.0, 51.0, 49.0, 50.0]
    assert enforcement_action(900.0, history, POLICY) == "kill_switch"


def test_enforcement_detects_a_deviation_from_a_perfectly_flat_history():
    assert enforcement_action(0.97, [0.60] * 6, POLICY) == "kill_switch"


def test_a_short_history_does_not_arm_the_kill_switch():
    assert enforcement_action(900.0, [50.0, 52.0], POLICY) == "cap_alert"
