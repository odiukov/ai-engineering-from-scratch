"""Тесты к уроку «Shadow-трафик, канарейка и постепенная выкатка». Правь exercise.py."""

import random

import pytest

from exercise import (
    BASELINE,
    GATES,
    STAGES,
    assign_variant,
    bucket_of,
    gate_breaches,
    rollback_policy,
    run_canary,
    shadow_call,
    split_traffic,
    widen_gates_for_noise,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

USERS = tuple(f"u-{i}" for i in range(2000))


def metrics(**overrides):
    """Метрики кандидата: база, поверх неё — заданные множители."""
    row = dict(BASELINE)
    for name, multiplier in overrides.items():
        row[name] = BASELINE[name] * multiplier
    return row


# --------------------------------------------------------------- bucket_of
def test_bucket_is_inside_the_unit_interval():
    assert all(0.0 <= bucket_of(u) < 1.0 for u in USERS[:200])


def test_bucket_of_the_same_user_never_changes():
    assert bucket_of("u-1") == APPROX(bucket_of("u-1"))


def test_different_users_land_in_different_buckets():
    assert len({bucket_of(u) for u in USERS[:200]}) == 200


def test_salt_reshuffles_the_layout():
    """Разные эксперименты не должны делить пользователей одинаково."""
    assert bucket_of("u-1") != bucket_of("u-1", salt="experiment-b")


def test_buckets_are_spread_over_the_whole_interval():
    """Хеш, забившийся в один угол, сломал бы любую долю трафика."""
    values = [bucket_of(u) for u in USERS]
    assert min(values) < 0.02
    assert max(values) > 0.98
    assert 0.45 < sum(values) / len(values) < 0.55


# ---------------------------------------------------------- assign_variant
def test_zero_share_sends_nobody_to_the_canary():
    assert all(assign_variant(u, 0.0) == "baseline" for u in USERS[:200])


def test_full_share_sends_everybody_to_the_canary():
    assert all(assign_variant(u, 1.0) == "canary" for u in USERS[:200])


def test_stable_hash_keeps_a_user_in_one_branch_while_random_does_not():
    """Ровно та причина, по которой канарейку нельзя делать через random.

    Устойчивый хеш держит пользователя в одной ветке между запросами.
    Наивный random перебрасывает его туда-сюда — и сравнение веток теряет
    смысл: метрика «канарейки» посчитана по разным людям.
    """
    users = USERS[:50]
    stable = [[assign_variant(u, 0.5) for _ in range(5)] for u in users]
    assert all(len(set(row)) == 1 for row in stable)

    rng = random.Random(0)
    naive = [["canary" if rng.random() < 0.5 else "baseline" for _ in range(5)] for _ in users]
    assert any(len(set(row)) > 1 for row in naive)


def test_share_outside_the_unit_interval_is_an_error():
    with pytest.raises(ValueError):
        assign_variant("u-1", 1.5)


# ----------------------------------------------------------- split_traffic
def test_split_covers_every_user_exactly_once():
    canary, baseline = split_traffic(USERS, 0.25)
    assert len(canary) + len(baseline) == len(USERS)
    assert set(canary) | set(baseline) == set(USERS)
    assert set(canary) & set(baseline) == set()


def test_split_keeps_the_input_order_inside_each_branch():
    canary, baseline = split_traffic(USERS[:100], 0.5)
    assert list(canary) == [u for u in USERS[:100] if u in set(canary)]
    assert list(baseline) == [u for u in USERS[:100] if u in set(baseline)]


def test_actual_share_is_close_to_the_requested_one():
    canary, _ = split_traffic(USERS, 0.25)
    assert 0.22 < len(canary) / len(USERS) < 0.28


def test_ramping_up_only_adds_users_to_the_canary():
    """Главное свойство ступенчатой выкатки: состав канарейки не тасуется.

    Кто попал в неё на 10%, остаётся на 25% и на 50%. Иначе «метрика
    ухудшилась» не отличить от «поменялась выборка».
    """
    previous = set()
    for share in (0.01, 0.10, 0.25, 0.50, 0.75, 1.00):
        canary, _ = split_traffic(USERS, share)
        assert previous <= set(canary)
        previous = set(canary)
    assert previous == set(USERS)


# ------------------------------------------------------------- shadow_call
def test_user_gets_the_baseline_answer_untouched():
    base = lambda req: {"text": "prod answer", "tokens": 100}
    cand = lambda req: {"text": "candidate answer", "tokens": 140}
    response, record = shadow_call({"q": "hi"}, base, cand)
    assert response == {"text": "prod answer", "tokens": 100}
    assert record["same_output"] is False


def test_shadow_records_the_token_delta_that_predicts_the_cost_spike():
    """Ради этого shadow и включают: скачок стоимости виден до канарейки."""
    base = lambda req: {"text": "a", "tokens": 100}
    cand = lambda req: {"text": "a", "tokens": 140}
    _, record = shadow_call({}, base, cand)
    assert record["token_delta"] == 40


def test_candidate_crash_never_reaches_the_user():
    def cand(req):
        raise ZeroDivisionError("candidate exploded")

    response, record = shadow_call({}, lambda req: {"text": "prod", "tokens": 10}, cand)
    assert response == {"text": "prod", "tokens": 10}
    assert "ZeroDivisionError" in record["candidate_error"]


def test_baseline_crash_is_a_real_outage_and_is_not_swallowed():
    """Падение прода нельзя прятать за словом shadow."""
    def base(req):
        raise ZeroDivisionError("prod is down")

    with pytest.raises(ZeroDivisionError):
        shadow_call({}, base, lambda req: {"text": "c", "tokens": 1})


def test_identical_outputs_are_reported_as_identical():
    same = lambda req: {"text": "same", "tokens": 10}
    _, record = shadow_call({}, same, same)
    assert record["same_output"] is True
    assert record["token_delta"] == 0


# ----------------------------------------------------------- gate_breaches
def test_candidate_equal_to_baseline_breaches_nothing():
    assert gate_breaches(dict(BASELINE)) == ()


def test_cost_regression_breaches_the_cost_gate():
    assert gate_breaches(metrics(cost_per_req=1.25)) == ("cost_per_req",)


def test_exactly_at_the_threshold_is_not_a_breach():
    """«Не более чем в 1.2 раза» обязано пропускать ровно 1.2."""
    assert gate_breaches(metrics(cost_per_req=1.2)) == ()


def test_several_breaches_are_reported_in_gate_order():
    breaches = gate_breaches(metrics(cost_per_req=1.5, latency_p99_ms=2.0, thumbs_down_rate=3.0))
    assert breaches == tuple(k for k in GATES if k in breaches)
    assert set(breaches) == {"latency_p99_ms", "cost_per_req", "thumbs_down_rate"}


def test_error_rate_gate_is_looser_than_the_cost_gate():
    """Ошибки шумят сильнее денег — множитель 2.0 против 1.2."""
    assert gate_breaches(metrics(error_rate=1.9)) == ()
    assert gate_breaches(metrics(cost_per_req=1.9)) == ("cost_per_req",)


def test_missing_measurement_is_not_a_pass():
    incomplete = dict(BASELINE)
    del incomplete["thumbs_down_rate"]
    with pytest.raises(KeyError):
        gate_breaches(incomplete)


# --------------------------------------------------- widen_gates_for_noise
def test_widening_scales_every_gate():
    widened = widen_gates_for_noise({"cost_per_req": 1.2}, 0.07)
    assert widened["cost_per_req"] == APPROX(1.284)


def test_zero_noise_leaves_gates_alone():
    assert widen_gates_for_noise(GATES, 0.0) == pytest.approx(GATES)


def test_widened_gates_stop_the_false_alarm_but_still_catch_the_real_one():
    """Гейт ниже шумового пола останавливает здоровые выкатки. Раз за разом.

    Кандидат на 25% дороже базы: при ±7% недетерминизма это ещё шум,
    при явных 60% — уже регрессия, и её широкий гейт всё равно ловит.
    """
    noisy = widen_gates_for_noise(GATES, 0.07)
    assert gate_breaches(metrics(cost_per_req=1.25)) == ("cost_per_req",)
    assert gate_breaches(metrics(cost_per_req=1.25), gates=noisy) == ()
    assert gate_breaches(metrics(cost_per_req=1.60), gates=noisy) == ("cost_per_req",)


def test_negative_noise_is_an_error():
    with pytest.raises(ValueError):
        widen_gates_for_noise(GATES, -0.1)


# -------------------------------------------------------------- run_canary
def test_clean_candidate_is_promoted_to_full_traffic():
    report = run_canary(lambda share: dict(BASELINE))
    assert report["promoted"] is True
    assert report["exposed_share"] == APPROX(1.0)
    assert len(report["history"]) == len(STAGES)


def test_rollback_fires_at_the_first_stage_and_spares_almost_everyone():
    """Ради этого числа канарейка и существует: 1% вместо 100%."""
    report = run_canary(lambda share: metrics(cost_per_req=1.5))
    assert report["promoted"] is False
    assert report["halted_at"] == APPROX(0.01)
    assert report["exposed_share"] < 0.02
    assert report["breaches"] == ("cost_per_req",)


def test_load_dependent_regression_halts_mid_progression():
    """Деградация, которая проявляется только под нагрузкой, всё равно ловится
    раньше, чем доедет до всех."""
    def measure(share):
        return metrics(latency_p99_ms=1.0 + 2.0 * share)

    report = run_canary(measure)
    assert report["promoted"] is False
    assert report["breaches"] == ("latency_p99_ms",)
    assert 0.01 < report["exposed_share"] < 1.0


def test_history_stops_at_the_breach_and_does_not_go_further():
    report = run_canary(lambda share: metrics(error_rate=3.0))
    assert len(report["history"]) == 1
    assert report["history"][0][0] == APPROX(STAGES[0])


def test_empty_progression_is_an_error_not_a_promotion():
    with pytest.raises(ValueError):
        run_canary(lambda share: dict(BASELINE), stages=())


# ---------------------------------------------------------- rollback_policy
def test_rollback_zeroes_the_share_and_pins_the_previous_digest():
    rolled = rollback_policy({"canary_share": 0.25, "model_digest": "sha256:new"}, "sha256:old")
    assert rolled["canary_share"] == APPROX(0.0)
    assert rolled["model_digest"] == "sha256:old"
    assert rolled["rolled_back"] is True


def test_rollback_is_a_config_flip_not_a_redeploy():
    """Если для отката нужен передеплой, откат займёт часы вместо секунд."""
    assert rollback_policy({"canary_share": 0.5}, "sha256:old")["requires_redeploy"] is False


def test_rollback_leaves_the_old_policy_intact_for_the_post_mortem():
    policy = {"canary_share": 0.25, "model_digest": "sha256:new"}
    rollback_policy(policy, "sha256:old")
    assert policy == {"canary_share": 0.25, "model_digest": "sha256:new"}


def test_rollback_without_a_pin_is_refused():
    with pytest.raises(ValueError):
        rollback_policy({"canary_share": 0.25}, "")


def test_after_rollback_nobody_is_in_the_canary():
    rolled = rollback_policy({"canary_share": 0.75}, "sha256:old")
    canary, baseline = split_traffic(USERS[:200], rolled["canary_share"])
    assert canary == ()
    assert len(baseline) == 200
