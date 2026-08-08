"""Тесты к уроку «Бюджеты действий, лимиты итераций и губернаторы стоимости».

Правь exercise.py.
"""

import pytest

from exercise import (
    DEFAULT_LIMITS,
    budget_warnings,
    cap_request_tokens,
    first_breached_cap,
    new_ledger,
    record_turn,
    run_session,
    tokens_to_usd,
    window_velocity,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Траектория из урока: 29 обычных ходов, дальше агент срывается в polling loop.
NORMAL_TOKENS = 2_500
LOOP_TOKENS = 8_000
LOOP_TRAJECTORY = [NORMAL_TOKENS] * 29 + [LOOP_TOKENS] * 200

# Стек, у которого velocity-лимит настроен на реальный масштаб трат этой
# траектории (обычный ход даёт 0.015 $/мин, ход в цикле — 0.048 $/мин).
TIGHT_LIMITS = {
    "max_tokens_per_request": 10_000,
    "max_turns": 200,
    "velocity_usd_per_min": 0.02,
    "velocity_window_min": 10.0,
}


# ----------------------------------------------------------- tokens_to_usd
def test_tokens_to_usd_prices_per_thousand():
    assert tokens_to_usd(1000) == APPROX(0.003)


def test_tokens_to_usd_scales_linearly():
    assert tokens_to_usd(8000) == APPROX(8 * tokens_to_usd(1000))


def test_tokens_to_usd_accepts_another_price():
    assert tokens_to_usd(500, 0.015) == APPROX(0.0075)


def test_tokens_to_usd_of_nothing_is_free():
    assert tokens_to_usd(0) == APPROX(0.0)


# ------------------------------------------------------ cap_request_tokens
def test_cap_lets_a_small_request_through_untouched():
    assert cap_request_tokens(8000, 10_000) == 8000


def test_cap_trims_an_oversized_request():
    assert cap_request_tokens(80_000, 10_000) == 10_000


def test_cap_without_a_ceiling_is_a_passthrough():
    assert cap_request_tokens(80_000, None) == 80_000


def test_cap_rejects_a_negative_request():
    """Отрицательный запрос — баг вызывающего, а не бесплатный ход."""
    with pytest.raises(ValueError):
        cap_request_tokens(-1, 10_000)


# --------------------------------------------------- new_ledger/record_turn
def test_new_ledger_starts_empty():
    led = new_ledger()
    assert (led["turns"], led["tokens"], led["usd"], led["history"]) == (0, 0, 0.0, [])


def test_new_ledger_hands_out_independent_objects():
    """Две сессии не должны делить один список истории."""
    a, b = new_ledger(), new_ledger()
    a = record_turn(a, 1000, 0.5)
    assert b["history"] == []


def test_record_turn_charges_the_tokens():
    led = record_turn(new_ledger(), 1000, 0.5)
    assert (led["turns"], led["tokens"], led["usd"]) == (1, 1000, APPROX(0.003))


def test_record_turn_appends_time_and_running_total():
    led = record_turn(record_turn(new_ledger(), 1000, 0.5), 1000, 1.0)
    assert led["history"] == [(0.5, APPROX(0.003)), (1.0, APPROX(0.006))]


def test_record_turn_does_not_mutate_the_old_ledger():
    """Ветвление сценариев ломается, если книга учёта правится на месте."""
    before = new_ledger()
    record_turn(before, 5000, 0.5)
    assert (before["turns"], before["usd"], before["history"]) == (0, 0.0, [])


# --------------------------------------------------------- window_velocity
def test_velocity_of_an_empty_history_is_zero():
    assert window_velocity([], 5.0, 10.0) == APPROX(0.0)


def test_velocity_is_dollars_over_elapsed_minutes():
    assert window_velocity([(1.0, 0.5), (2.0, 1.0)], 2.0, 10.0) == APPROX(0.5)


def test_velocity_during_warmup_divides_by_elapsed_not_window_width():
    """Ловушка прогрева: за 2 минуты потрачен доллар — это 0.5 $/мин, не 0.1."""
    assert window_velocity([(2.0, 1.0)], 2.0, 10.0) == APPROX(0.5)


def test_velocity_forgets_spending_older_than_the_window():
    """Сотня долларов час назад не должна тянуть текущую скорость вверх."""
    history = [(1.0, 100.0), (61.0, 101.0)]
    assert window_velocity(history, 61.0, 10.0) == APPROX(0.1)


def test_velocity_rises_when_the_agent_enters_a_loop():
    calm = run_session([NORMAL_TOKENS] * 40, {"max_turns": 40})
    busy = run_session([LOOP_TOKENS] * 40, {"max_turns": 40})
    rate_calm = window_velocity(calm["history"], calm["history"][-1][0], 10.0)
    rate_busy = window_velocity(busy["history"], busy["history"][-1][0], 10.0)
    assert rate_busy > 3 * rate_calm


# -------------------------------------------------------- first_breached_cap
def test_nothing_is_breached_on_a_fresh_ledger():
    assert first_breached_cap(new_ledger(), DEFAULT_LIMITS, 0.0) is None


def test_iteration_cap_fires_on_the_last_allowed_turn():
    """Лимит в 3 хода означает, что третий ход последний, а не первый лишний."""
    led = run_session([1000] * 2, {"max_turns": 99})
    assert first_breached_cap(led, {"max_turns": 3}, 1.0) is None
    led = record_turn(led, 1000, 1.5)
    assert first_breached_cap(led, {"max_turns": 3}, 1.5) == "max_turns"


def test_dollar_cap_fires_on_accumulated_spend():
    led = run_session([10_000] * 5, {"max_turns": 99})
    assert first_breached_cap(led, {"max_budget_usd": 0.1}, 2.5) == "max_budget_usd"


def test_a_limit_absent_from_the_config_is_simply_off():
    """Так собирается конфигурация «velocity выключен», а не падение по KeyError."""
    led = run_session([10_000] * 5, {"max_turns": 99})
    assert first_breached_cap(led, {"monthly_cap_usd": 500.0}, 2.5) is None


def test_earlier_layer_wins_when_several_caps_break_at_once():
    led = run_session([10_000] * 5, {"max_turns": 99})
    limits = {"max_turns": 5, "max_budget_usd": 0.01}
    assert first_breached_cap(led, limits, 2.5) == "max_turns"


# --------------------------------------------------------------- run_session
def test_session_stops_on_the_iteration_cap():
    led = run_session([1000, 1000, 1000], {"max_turns": 1})
    assert (led["turns"], led["stopped_by"]) == (1, "max_turns")


def test_request_cap_trims_tokens_before_they_are_charged():
    """Порядок слоёв: сначала режем, потом списываем. Иначе деньги уже ушли."""
    led = run_session([80_000], {"max_tokens_per_request": 10_000})
    assert (led["tokens"], led["usd"]) == (10_000, APPROX(0.03))


def test_dollar_cap_refuses_the_next_turn_before_it_is_charged():
    """Два хода по $0.006 не должны протащить hard cap $0.01 до $0.012."""
    led = run_session([2000, 2000], {"max_budget_usd": 0.01})
    assert led["stopped_by"] == "max_budget_usd"
    assert (led["turns"], led["tokens"], led["usd"]) == (1, 2000, APPROX(0.006))
    assert len(led["history"]) == 1


def test_velocity_limit_catches_the_polling_loop_long_before_the_iteration_cap():
    led = run_session(LOOP_TRAJECTORY, TIGHT_LIMITS)
    assert led["stopped_by"] == "velocity_usd_per_min"
    assert led["turns"] < TIGHT_LIMITS["max_turns"]


def test_turning_the_velocity_limit_off_costs_ten_times_more():
    """Упражнение 1 из урока: измерь, сколько агент потратит без velocity."""
    with_velocity = run_session(LOOP_TRAJECTORY, TIGHT_LIMITS)
    without = dict(TIGHT_LIMITS)
    del without["velocity_usd_per_min"]
    without_velocity = run_session(LOOP_TRAJECTORY, without)
    assert without_velocity["usd"] > 10 * with_velocity["usd"]
    assert without_velocity["stopped_by"] == "max_turns"


def test_session_that_never_breaches_a_cap_reports_no_stopper():
    led = run_session([1000, 1000], {"max_turns": 100, "max_budget_usd": 100.0})
    assert (led["turns"], led["stopped_by"]) == (2, None)


def test_session_is_reproducible():
    """Ни времени из time.time(), ни глобального random — два прогона равны."""
    a = run_session(LOOP_TRAJECTORY, TIGHT_LIMITS)
    b = run_session(LOOP_TRAJECTORY, TIGHT_LIMITS)
    assert (a["turns"], a["tokens"], a["stopped_by"]) == (
        b["turns"], b["tokens"], b["stopped_by"]
    )


# ------------------------------------------------------------ budget_warnings
def test_no_warnings_while_the_session_is_young():
    assert budget_warnings(run_session([1000], {"max_turns": 100}),
                           {"max_turns": 100}) == ()


def test_iteration_cap_warns_at_eighty_percent():
    led = run_session([1000] * 8, {"max_turns": 100})
    assert budget_warnings(led, {"max_turns": 10}) == ("max_turns",)


def test_dollar_cap_warns_before_it_fires():
    led = run_session([1000] * 8, {"max_turns": 100})
    assert budget_warnings(led, {"max_budget_usd": 0.04}, warn_at=0.5) == (
        "max_budget_usd",
    )


def test_a_cap_that_already_fired_is_not_a_warning():
    """Сработавший лимит — это не предупреждение, а срабатывание."""
    led = run_session([1000] * 20, {"max_turns": 10})
    assert "max_turns" not in budget_warnings(led, {"max_turns": 10})
