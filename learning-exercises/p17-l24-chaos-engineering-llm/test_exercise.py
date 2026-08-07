"""Тесты к уроку «Chaos engineering для LLM-продакшена». Правь exercise.py."""

import random

import pytest

from exercise import (
    EXPECTED_ERROR_RATE,
    ChaosError,
    burn_rate,
    experiment_report,
    inject_failures,
    route_request,
    run_scenario,
    serve_request,
    should_abort,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

THREE_HEALTHY = [True, True, True]
ONE_DEAD = [True, False, True]
ALL_DEAD = [False, False, False]


# -------------------------------------------------------------- burn_rate
def test_burn_rate_is_a_ratio_to_the_expected_baseline():
    assert burn_rate(0.002, 0.0005) == APPROX(4.0)


def test_burn_rate_of_the_baseline_itself_is_one():
    assert burn_rate(EXPECTED_ERROR_RATE, EXPECTED_ERROR_RATE) == APPROX(1.0)


def test_burn_rate_without_induced_errors_is_zero():
    assert burn_rate(0.0, 0.0005) == APPROX(0.0)


def test_burn_rate_survives_a_zero_baseline_instead_of_dividing_by_zero():
    """SLO ещё не заполнили — safety plane обязан пережить это, а не упасть."""
    assert burn_rate(0.01, 0.0) > 0


# ------------------------------------------------------------ should_abort
def test_abort_needs_both_high_burn_and_wide_blast():
    assert should_abort(30.0, 0.30) is True


def test_narrow_blast_radius_survives_even_an_enormous_burn():
    """80x горения на 10% трафика — эксперимент доигрывают, это его смысл."""
    assert should_abort(80.0, 0.10) is False


def test_wide_blast_radius_without_errors_is_not_a_reason_to_abort():
    assert should_abort(1.0, 0.90) is False


def test_abort_thresholds_are_strict_inequalities():
    """Ровно на пороге ещё не гасим: гейт срабатывает при превышении."""
    assert should_abort(2.0, 0.2) is False


# --------------------------------------------------------- inject_failures
def test_injector_with_zero_probability_never_fails():
    assert inject_failures(4, 0.0, random.Random(0)) == [False] * 4


def test_injector_with_probability_one_always_fails():
    assert inject_failures(4, 1.0, random.Random(0)) == [True] * 4


def test_injector_is_reproducible_for_the_same_seed():
    """Без воспроизводимости постмортем эксперимента невозможен."""
    a = inject_failures(50, 0.3, random.Random(17))
    b = inject_failures(50, 0.3, random.Random(17))
    assert a == b


def test_injector_does_not_read_the_global_random_state():
    """Глобальный random.seed не должен влиять на результат инжектора."""
    random.seed(1)
    a = inject_failures(30, 0.5, random.Random(4))
    random.seed(999)
    b = inject_failures(30, 0.5, random.Random(4))
    assert a == b


def test_injector_roughly_hits_the_requested_rate():
    fails = inject_failures(2000, 0.25, random.Random(3))
    assert 0.20 < sum(fails) / 2000 < 0.30


# ----------------------------------------------------------- route_request
def test_router_spreads_requests_round_robin():
    assert [route_request(THREE_HEALTHY, i) for i in range(4)] == [0, 1, 2, 0]


def test_router_does_not_look_at_health_and_hits_the_dead_replica():
    """L4-балансировщик без health-check — источник половины инцидентов."""
    assert route_request(ONE_DEAD, 1) == 1


def test_router_raises_chaos_error_when_nothing_is_alive():
    with pytest.raises(ChaosError):
        route_request(ALL_DEAD, 0)


def test_router_raises_chaos_error_on_an_empty_replica_set():
    with pytest.raises(ChaosError):
        route_request([], 0)


# ----------------------------------------------------------- serve_request
def test_healthy_replica_serves_on_the_first_attempt():
    assert serve_request(THREE_HEALTHY, 0, False, False) == (True, 1)


def test_dead_replica_without_retry_loses_the_request():
    assert serve_request(ONE_DEAD, 1, False, False) == (False, 1)


def test_retry_reroutes_the_request_to_the_next_live_replica():
    assert serve_request(ONE_DEAD, 1, False, True) == (True, 2)


def test_retry_also_rescues_an_injected_provider_failure():
    """429 от провайдера лечится тем же перезаходом, что и мёртвая реплика."""
    assert serve_request(THREE_HEALTHY, 0, True, True) == (True, 2)


def test_retry_walks_around_the_ring_and_stops_after_one_lap():
    """Две мёртвых из трёх: живая нашлась на втором шаге, третьего не будет."""
    assert serve_request([True, False, False], 1, False, True) == (True, 3)


# ------------------------------------------------------------ run_scenario
def test_healthy_service_serves_everything_with_one_attempt_per_request():
    r = run_scenario(THREE_HEALTHY, 9, 0.0, False, random.Random(0))
    assert (r["served"], r["failed"], r["attempts"]) == (9, 0, 9)


def test_one_dead_replica_without_retry_loses_a_third_of_traffic():
    r = run_scenario(ONE_DEAD, 9, 0.0, False, random.Random(0))
    assert r["error_rate"] == APPROX(1 / 3)


def test_one_dead_replica_with_retry_does_not_take_the_service_down():
    """Главный вывод эксперимента: сервис деградирует, а не падает."""
    r = run_scenario(ONE_DEAD, 9, 0.0, True, random.Random(0))
    assert r["error_rate"] == APPROX(0.0)


def test_retry_pays_for_survival_with_extra_attempts():
    off = run_scenario(ONE_DEAD, 9, 0.0, False, random.Random(0))
    on = run_scenario(ONE_DEAD, 9, 0.0, True, random.Random(0))
    assert on["attempts"] > off["attempts"]


def test_scenario_with_every_replica_dead_raises_chaos_error():
    """Тут уже не деградация: обслуживать некому."""
    with pytest.raises(ChaosError):
        run_scenario(ALL_DEAD, 9, 0.0, True, random.Random(0))


def test_provider_outage_without_retry_matches_the_injected_rate():
    """100% отказов провайдера и выключенный retry — сервис теряет всё."""
    r = run_scenario(THREE_HEALTHY, 20, 1.0, False, random.Random(0))
    assert r["error_rate"] == APPROX(1.0)


# ------------------------------------------------------- experiment_report
def test_report_marks_the_wide_provider_experiment_as_aborted():
    r = experiment_report("provider 429", {"error_rate": 0.015}, 0.30)
    assert (r["burn_rate_x"], r["aborted"]) == (APPROX(30.0), True)


def test_report_lets_the_narrow_tokenizer_experiment_finish():
    """80x горения, но радиус 10% — статус COMPLETED, а не ABORTED."""
    r = experiment_report("tokenizer stall", {"error_rate": 0.040}, 0.10)
    assert (r["burn_rate_x"], r["status"]) == (APPROX(80.0), "COMPLETED")


def test_report_reads_the_error_rate_straight_from_the_scenario():
    scenario = run_scenario(ONE_DEAD, 9, 0.0, False, random.Random(0))
    r = experiment_report("pod kill", scenario, 0.05)
    assert r["error_rate"] == APPROX(scenario["error_rate"])


def test_working_retry_keeps_the_pod_kill_experiment_green():
    """Тот же pod-kill с ретраями не жжёт бюджет вообще."""
    scenario = run_scenario(ONE_DEAD, 9, 0.0, True, random.Random(0))
    r = experiment_report("pod kill", scenario, 0.05)
    assert (r["burn_rate_x"], r["aborted"]) == (APPROX(0.0), False)
