"""Тесты к уроку «Kill switch, circuit breaker и canary token». Правь exercise.py."""

import random

import pytest

from exercise import (
    BREAKER_CLOSED,
    CANARY_PATHS,
    KILL_SWITCH_OFF,
    breaker_step,
    canary_hits,
    engage_kill_switch,
    ewma,
    ewma_alarm,
    hard_limit_breach,
    release_kill_switch,
    run_trajectory,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def repeat(breaker, key, times, ok=True, start=1.0, **kw):
    """Прогнать один и тот же action_key несколько раз. Вернуть (вердикты, breaker)."""
    verdicts = []
    for i in range(times):
        allowed, breaker = breaker_step(breaker, key, ok, start + i, **kw)
        verdicts.append(allowed)
    return verdicts, breaker


# ------------------------------------------------------- engage_kill_switch
def test_engaging_records_reason_and_time():
    sw = engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    assert (sw["engaged"], sw["reason"], sw["engaged_at"]) == (True, "runaway loop", 10.0)


def test_engaging_without_a_reason_is_refused():
    """Выключатель без записи в журнале нельзя аудировать."""
    with pytest.raises(ValueError):
        engage_kill_switch(KILL_SWITCH_OFF, "", 10.0)


def test_engaging_twice_is_idempotent_and_keeps_the_first_reason():
    """Повторное срабатывание безопасно и не теряет момент остановки."""
    first = engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    second = engage_kill_switch(first, "someone else noticed", 99.0)
    assert (second["engaged"], second["reason"], second["engaged_at"]) == (
        True, "runaway loop", 10.0,
    )


def test_engaging_does_not_mutate_the_external_switch():
    """Общее состояние из модуля обязано пережить вызов нетронутым."""
    engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    assert KILL_SWITCH_OFF["engaged"] is False


# ------------------------------------------------------ release_kill_switch
def test_release_records_who_did_it():
    engaged = engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    sw = release_kill_switch(engaged, "alice", "loop patched, tests green", 20.0)
    assert (sw["engaged"], sw["released_by"], sw["released_at"]) == (
        False, "alice", 20.0,
    )


def test_release_without_an_operator_is_refused():
    """Обратное включение — человеческое действие, а не автоматический таймаут."""
    engaged = engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    with pytest.raises(ValueError):
        release_kill_switch(engaged, "", "loop patched", 20.0)


def test_release_without_a_note_is_refused():
    engaged = engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)
    with pytest.raises(ValueError):
        release_kill_switch(engaged, "alice", "", 20.0)


def test_switch_can_be_engaged_again_after_a_release():
    """Цикл включить-снять-включить не должен ломать состояние."""
    engaged = engage_kill_switch(KILL_SWITCH_OFF, "first incident", 10.0)
    released = release_kill_switch(engaged, "alice", "fixed", 20.0)
    again = engage_kill_switch(released, "second incident", 30.0)
    assert (again["engaged"], again["reason"], again["engaged_at"]) == (
        True, "second incident", 30.0,
    )


# -------------------------------------------------------------- breaker_step
def test_fifth_identical_call_is_the_one_that_gets_blocked():
    verdicts, br = repeat(BREAKER_CLOSED, "read:app.log", 5)
    assert verdicts == [True, True, True, True, False]
    assert br["state"] == "open"


def test_five_consecutive_failures_trip_the_breaker_too():
    """Разные инструменты, один и тот же отказ — системная поломка."""
    br = BREAKER_CLOSED
    verdicts = []
    for i, key in enumerate("abcde"):
        allowed, br = breaker_step(br, key, False, float(i))
        verdicts.append(allowed)
    assert (verdicts[-1], br["state"]) == (False, "open")


def test_varied_successful_calls_never_trip_the_breaker():
    br = BREAKER_CLOSED
    for i in range(50):
        allowed, br = breaker_step(br, f"tool:{i % 7}", True, float(i))
        assert allowed is True
    assert br["state"] == "closed"


def test_open_breaker_keeps_blocking_without_extending_its_own_cooldown():
    """Ловушка: если двигать opened_at на каждой попытке, half_open не наступит."""
    _, br = repeat(BREAKER_CLOSED, "read:app.log", 5)
    opened_at = br["opened_at"]
    for t in (5.1, 6.0, 9.0):
        allowed, br = breaker_step(br, "read:app.log", True, t)
        assert allowed is False
    assert br["opened_at"] == opened_at


def test_probes_walk_the_breaker_from_half_open_back_to_closed():
    _, br = repeat(BREAKER_CLOSED, "read:app.log", 5)
    t = br["opened_at"] + 10.0
    allowed, br = breaker_step(br, "read:app.log", True, t, probes=2)
    assert (allowed, br["state"], br["probes_left"]) == (True, "half_open", 1)
    allowed, br = breaker_step(br, "read:app.log", True, t + 1.0, probes=2)
    assert (allowed, br["state"], br["recent"]) == (True, "closed", ())


def test_failed_probe_is_allowed_but_reopens_the_breaker():
    """Пробный вызов должен выполниться — иначе о провале неоткуда узнать."""
    _, br = repeat(BREAKER_CLOSED, "read:app.log", 5)
    probe_at = br["opened_at"] + 10.0
    allowed, br = breaker_step(br, "read:app.log", False, probe_at)
    assert (allowed, br["state"], br["opened_at"]) == (True, "open", probe_at)


# --------------------------------------------------------------- canary_hits
def test_reading_a_canary_is_reported_with_its_turn_number():
    traj = [
        {"kind": "read", "payload": "README.md"},
        {"kind": "read", "payload": "~/.env.canary"},
    ]
    assert canary_hits(traj) == ((2, "~/.env.canary"),)


def test_ordinary_reads_are_silent():
    assert canary_hits([{"kind": "read", "payload": "src/app.py"}]) == ()


def test_a_canary_the_agent_only_writes_to_is_not_a_read_alarm():
    assert canary_hits([{"kind": "write", "payload": CANARY_PATHS[0]}]) == ()


# ---------------------------------------------------------------------- ewma
def test_ewma_of_a_flat_series_is_flat():
    assert ewma([1.0, 1.0, 1.0], 0.5) == [APPROX(1.0)] * 3


def test_ewma_starts_at_the_first_sample():
    assert ewma([7.0, 0.0], 0.25)[0] == APPROX(7.0)


def test_ewma_moves_only_partway_towards_a_new_value():
    assert ewma([0.0, 1.0], 0.5) == [APPROX(0.0), APPROX(0.5)]


# ---------------------------------------------------------------- ewma_alarm
def test_alarm_fires_on_a_spike_over_a_flat_baseline():
    assert ewma_alarm([1.0] * 10 + [50.0], 0.3, 4.0) == 10


def test_alarm_stays_quiet_on_noise_inside_tolerance():
    """Шум вокруг ровной базы — не событие. Ложная тревога хуже молчания."""
    rng = random.Random(0)
    noise = [10.0 + rng.uniform(-1.0, 1.0) for _ in range(100)]
    assert ewma_alarm(noise, 0.3, 6.0) is None


def test_alarm_fires_when_a_real_spike_sits_inside_the_same_noise():
    rng = random.Random(0)
    noise = [10.0 + rng.uniform(-1.0, 1.0) for _ in range(100)]
    spiked = noise[:50] + [200.0] + noise[51:]
    assert ewma_alarm(spiked, 0.3, 6.0) == 50


def test_alarm_never_fires_on_slow_drift():
    """Дыра статистики: базовая линия уползает вместе с атакующим."""
    drift = [1.0 + 0.05 * i for i in range(100)]
    assert ewma_alarm(drift, 0.3, 4.0) is None


def test_alarm_refuses_a_zero_warmup():
    with pytest.raises(ValueError):
        ewma_alarm([1.0, 2.0, 3.0], 0.3, 4.0, warmup=0)


# ----------------------------------------------------------- hard_limit_breach
def test_a_quiet_log_breaches_nothing():
    assert hard_limit_breach([0.0, 1.0, 2.0], 5, 10.0) is None


def test_a_burst_breaches_on_the_call_that_crosses_the_line():
    assert hard_limit_breach([0.0, 0.1, 0.2, 0.3], 3, 10.0) == 3


def test_the_window_forgets_calls_that_fell_out_of_it():
    """Те же четыре вызова, растянутые по времени, лимит не пробивают."""
    assert hard_limit_breach([0.0, 20.0, 40.0, 60.0], 3, 10.0) is None


def test_hard_limit_catches_the_drift_that_walked_past_the_statistics():
    """Слоение из урока: статистика промолчала, константа сработала."""
    times = [0.0]
    for i in range(199):
        times.append(times[-1] + max(0.1, 1.0 - 0.01 * i))
    rates = [1.0 + 0.05 * i for i in range(100)]
    assert ewma_alarm(rates, 0.3, 4.0) is None
    assert hard_limit_breach(times, 50, 10.0) is not None


# ------------------------------------------------------------- run_trajectory
def test_a_clean_trajectory_runs_to_the_end():
    traj = [{"kind": "tool", "payload": f"read:{i}"} for i in range(6)]
    rep = run_trajectory(traj)
    assert (rep["executed"], rep["stopped_by"]) == (6, None)


def test_engaged_kill_switch_stops_the_agent_before_the_first_action():
    """Выключатель читается перед КАЖДЫМ действием, а не один раз на старте."""
    engaged = engage_kill_switch(KILL_SWITCH_OFF, "operator pulled the plug", 1.0)
    traj = [{"kind": "tool", "payload": "read:a"}] * 3
    rep = run_trajectory(traj, switch=engaged)
    assert (rep["executed"], rep["stopped_by"]) == (0, "kill_switch")


def test_the_breaker_stops_the_repetitive_loop_midway():
    traj = [{"kind": "tool", "payload": "read:logs/app.log"}] * 9
    rep = run_trajectory(traj)
    assert (rep["executed"], rep["stopped_by"]) == (4, "circuit_breaker")


def test_a_canary_read_alerts_but_does_not_halt_the_agent():
    """Приманка — детектор, а не тормоз: она поднимает тревогу и пропускает ход."""
    traj = [
        {"kind": "read", "payload": "src/app.py"},
        {"kind": "read", "payload": "~/.env.canary"},
        {"kind": "read", "payload": "src/other.py"},
    ]
    rep = run_trajectory(traj)
    assert rep["canary"] == ((2, "~/.env.canary"),)
    assert (rep["executed"], rep["stopped_by"]) == (3, None)


def test_run_trajectory_leaves_the_shared_breaker_untouched():
    run_trajectory([{"kind": "tool", "payload": "read:a"}] * 9)
    assert BREAKER_CLOSED["state"] == "closed" and BREAKER_CLOSED["recent"] == ()
