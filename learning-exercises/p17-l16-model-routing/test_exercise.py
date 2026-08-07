"""Тесты к уроку «Роутинг моделей: каскад, эскалация и цена качества». Правь exercise.py."""

import pytest

from exercise import (
    CASCADE_THRESHOLD,
    HARD_SIMILARITY,
    LONG_PROMPT_TOKENS,
    MAX_ESCALATION_RATE,
    RoutingError,
    call_cost,
    cascade,
    cascade_break_even_rate,
    drift_alarm,
    make_request,
    pre_route,
    route_request,
    run_workload,
)

APPROX = lambda x: pytest.approx(x, abs=1e-12)
ROUGH = lambda x: pytest.approx(x, rel=1e-9)


def req(**kw):
    """Обычный лёгкий запрос, поля перекрываются именованными аргументами."""
    base = dict(req_id="q", task="chat", prompt_tokens=1000, output_tokens=200)
    base.update(kw)
    return make_request(**base)


def stream(n, **kw):
    return [req(req_id=f"q{i}", **kw) for i in range(n)]


# -------------------------------------------------------------- make_request
def test_request_keeps_every_routing_signal():
    r = req(task="code", similarity=0.9, cheap_confidence=0.4, cheap_correct=False)
    assert (r["task"], r["similarity"], r["cheap_confidence"], r["cheap_correct"]) == (
        "code", 0.9, 0.4, False)


def test_confidence_and_correctness_are_independent_fields():
    """Модель бывает уверенно неправа — каскад этого не видит и не чинит."""
    r = req(cheap_confidence=1.0, cheap_correct=False)
    assert cascade(r)["escalated"] is False
    assert cascade(r)["correct"] is False


def test_similarity_outside_the_unit_interval_is_rejected():
    with pytest.raises(RoutingError):
        make_request("q", "chat", 300, 80, similarity=1.7)


def test_confidence_outside_the_unit_interval_is_rejected():
    with pytest.raises(RoutingError):
        make_request("q", "chat", 300, 80, cheap_confidence=-0.1)


def test_empty_prompt_is_rejected():
    with pytest.raises(RoutingError):
        make_request("q", "chat", 0, 80)


# ----------------------------------------------------------------- call_cost
def test_cheap_call_price():
    assert call_cost("cheap", 1000, 200) == APPROX(0.0008)


def test_frontier_call_price():
    assert call_cost("frontier", 1000, 200) == APPROX(0.006)


def test_price_ratio_is_the_same_for_any_token_mix():
    """Выход у обеих моделей ровно в 5 раз дороже входа — отношение не плавает."""
    long_out = call_cost("cheap", 100, 9000) / call_cost("frontier", 100, 9000)
    long_in = call_cost("cheap", 9000, 100) / call_cost("frontier", 9000, 100)
    assert long_out == ROUGH(long_in)
    assert long_out == ROUGH(2 / 15)


def test_unknown_model_is_rejected():
    with pytest.raises(RoutingError):
        call_cost("gpt-9", 1000, 200)


# ----------------------------------------------------------------- pre_route
def test_plain_short_chat_goes_cheap():
    assert pre_route(req()) == "cheap"


def test_hard_task_class_goes_frontier():
    assert pre_route(req(task="math")) == "frontier"


def test_long_prompt_goes_frontier():
    assert pre_route(req(prompt_tokens=LONG_PROMPT_TOKENS + 1)) == "frontier"
    assert pre_route(req(prompt_tokens=LONG_PROMPT_TOKENS)) == "cheap"


def test_similarity_to_the_hard_bucket_goes_frontier():
    assert pre_route(req(similarity=HARD_SIMILARITY)) == "frontier"
    assert pre_route(req(similarity=HARD_SIMILARITY - 0.01)) == "cheap"


def test_one_hard_signal_outvotes_two_easy_ones():
    """Сигналы не голосуют большинством: цена ошибки несимметрична."""
    assert pre_route(req(task="chat", prompt_tokens=10, similarity=0.95)) == "frontier"


def test_pre_route_ignores_confidence_because_it_does_not_exist_yet():
    """Уверенность появляется только после вызова — классификатор её не видит."""
    assert pre_route(req(cheap_confidence=0.0)) == "cheap"


# ------------------------------------------------------------------- cascade
def test_confident_cheap_answer_costs_one_cheap_call():
    out = cascade(req(cheap_confidence=0.9))
    assert out["models"] == ["cheap"]
    assert out["cost_usd"] == APPROX(0.0008)


def test_escalation_pays_for_both_calls():
    """Ловушка: первый прогон оплачен и выброшен, вход посчитан дважды."""
    out = cascade(req(cheap_confidence=0.4))
    assert out["models"] == ["cheap", "frontier"]
    assert out["cost_usd"] == APPROX(0.0068)
    assert out["cost_usd"] > call_cost("frontier", 1000, 200)


def test_escalation_repairs_the_answer():
    assert cascade(req(cheap_confidence=0.1, cheap_correct=False))["correct"] is True


def test_confidence_exactly_on_the_threshold_does_not_escalate():
    assert cascade(req(cheap_confidence=CASCADE_THRESHOLD))["escalated"] is False


def test_threshold_outside_the_unit_interval_is_rejected():
    with pytest.raises(RoutingError):
        cascade(req(), threshold=1.5)


# ------------------------------------------------- cascade_break_even_rate
def test_break_even_rate_is_thirteen_fifteenths():
    assert cascade_break_even_rate(4000, 300) == ROUGH(13 / 15)


def test_break_even_rate_does_not_depend_on_the_token_mix():
    assert cascade_break_even_rate(100, 5000) == ROUGH(cascade_break_even_rate(9000, 10))


def test_break_even_rate_is_far_above_the_drift_alarm():
    """30% эскалаций — порог по качеству и задержке, а не по деньгам."""
    assert cascade_break_even_rate(1000, 200) > MAX_ESCALATION_RATE


def test_free_request_has_no_break_even_rate():
    with pytest.raises(RoutingError):
        cascade_break_even_rate(0, 0)


# ------------------------------------------------------------- route_request
def test_frontier_only_always_calls_the_expensive_model():
    assert route_request(req(task="chat"), "frontier_only")["models"] == ["frontier"]


def test_pre_route_makes_exactly_one_call():
    out = route_request(req(task="math"), "pre_route")
    assert out["models"] == ["frontier"]
    assert out["escalated"] is False


def test_pre_route_cannot_repair_a_misclassified_request():
    """Классификатор ошибся — исправлять нечем, второго круга нет."""
    assert route_request(req(task="chat", cheap_correct=False), "pre_route")["correct"] is False


def test_unknown_strategy_is_rejected():
    with pytest.raises(RoutingError):
        route_request(req(), "vibes")


# --------------------------------------------------------------- run_workload
def test_cascade_saves_on_a_stream_of_easy_requests():
    report = run_workload(stream(100, cheap_confidence=0.9), "cascade")
    assert report["saving_usd"] > 0
    assert report["saving_pct"] == ROUGH(100 * (1 - 2 / 15))
    assert report["escalation_rate"] == APPROX(0.0)


def test_cascade_loses_money_on_a_stream_of_hard_requests():
    """Обратная сторона: дешёвый прогон оплачен и выброшен на каждом запросе."""
    report = run_workload(stream(100, cheap_confidence=0.1), "cascade")
    assert report["saving_usd"] < 0
    assert report["total_cost_usd"] > report["baseline_cost_usd"]
    assert report["escalation_rate"] == APPROX(1.0)


def test_the_break_even_rate_predicts_where_the_sign_flips():
    """Ниже порога каскад ещё экономит, выше — уже нет. Порог из формулы."""
    rate = cascade_break_even_rate(1000, 200)
    below = stream(86, cheap_confidence=0.1) + stream(14, cheap_confidence=0.9)
    above = stream(90, cheap_confidence=0.1) + stream(10, cheap_confidence=0.9)
    assert run_workload(below, "cascade")["escalation_rate"] < rate
    assert run_workload(below, "cascade")["saving_usd"] > 0
    assert run_workload(above, "cascade")["escalation_rate"] > rate
    assert run_workload(above, "cascade")["saving_usd"] < 0


def test_cheap_only_is_the_cheapest_and_the_worst():
    easy = stream(50, cheap_confidence=0.9, cheap_correct=True)
    wrong = stream(50, cheap_confidence=0.9, cheap_correct=False)
    cheap = run_workload(easy + wrong, "cheap_only")
    frontier = run_workload(easy + wrong, "frontier_only")
    assert cheap["total_cost_usd"] < frontier["total_cost_usd"]
    assert cheap["accuracy"] < frontier["accuracy"]


def test_pre_route_never_scores_worse_than_cheap_only():
    """Маршрут на frontier может только починить ответ, испортить не может."""
    mixed = stream(50, task="math", cheap_correct=False) + stream(50, task="chat")
    assert (run_workload(mixed, "pre_route")["accuracy"]
            >= run_workload(mixed, "cheap_only")["accuracy"])


def test_frontier_only_is_its_own_baseline():
    report = run_workload(stream(30), "frontier_only")
    assert report["saving_usd"] == APPROX(0.0)
    assert report["quality_loss_pct"] == APPROX(0.0)


def test_empty_stream_does_not_divide_by_zero():
    report = run_workload([], "cascade")
    assert report["saving_pct"] == APPROX(0.0)
    assert report["escalation_rate"] == APPROX(0.0)


def test_cheap_share_counts_requests_served_without_the_frontier():
    mixed = stream(70, task="chat") + stream(30, task="code")
    assert run_workload(mixed, "pre_route")["cheap_share"] == APPROX(0.7)


# ---------------------------------------------------------------- drift_alarm
def test_healthy_cascade_raises_nothing():
    assert drift_alarm(run_workload(stream(100, cheap_confidence=0.9), "cascade")) == []


def test_over_routing_raises_the_escalation_alarm_alone():
    """Эскалируют все, но качество не пострадало: одна причина, не две."""
    report = run_workload(stream(100, cheap_confidence=0.1), "cascade")
    assert drift_alarm(report) == ["escalation_rate"]


def test_confident_wrong_answers_raise_the_quality_alarm_alone():
    report = run_workload(stream(100, cheap_confidence=0.9, cheap_correct=False), "cascade")
    assert drift_alarm(report) == ["quality_loss"]


def test_cheap_creep_raises_both_reasons():
    creep = (stream(50, cheap_confidence=0.1)
             + stream(50, cheap_confidence=0.9, cheap_correct=False))
    assert drift_alarm(run_workload(creep, "cascade")) == ["escalation_rate", "quality_loss"]


def test_alarm_is_silent_exactly_on_the_threshold():
    """Сравнение строгое: ровно на пороге тревоги ещё нет."""
    report = run_workload(stream(100, cheap_confidence=0.9), "cascade")
    report["escalation_rate"] = MAX_ESCALATION_RATE
    assert drift_alarm(report) == []
