"""Тесты к уроку «SRE для AI: триаж агентами и error budget». Правь exercise.py."""

import pytest

from exercise import (
    FAST_BURN,
    SAFE_ACTIONS,
    SLOW_BURN,
    UNSAFE_ACTIONS,
    adversarial_review,
    bad_minutes,
    error_budget,
    is_safe_action,
    normalize_cause,
    release_decision,
    retrieve_runbook,
    triage,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)

MONTH_S = 30 * 24 * 3600
HOUR_S = 3600

RUNBOOKS = {
    "RB-017": {"symptom": "KV cache OOM under burst concurrency", "action": "restart_pod"},
    "RB-004": {"symptom": "gateway 429 storm from a single tenant", "action": "enable_preapproved_flag"},
    "RB-031": {"symptom": "checkout latency spike after deploy", "action": "revert_deploy"},
    "RB-055": {"symptom": "database connection pool exhausted", "action": "alter_database"},
}

OOM = "KV cache OOM under burst concurrency"


def agent(name, root_cause, confidence):
    return lambda incident: {"agent": name, "root_cause": root_cause, "confidence": confidence}


# ----------------------------------------------------------- normalize_cause
def test_same_cause_in_different_words_gives_the_same_key():
    assert normalize_cause("vLLM OOM caused by a KV cache spike") == normalize_cause(
        "KV-cache spike, vLLM OOM"
    )


def test_key_is_sorted_and_deduplicated():
    assert normalize_cause("cache cache OOM spike") == ("cache", "oom", "spike")


def test_key_ignores_case_and_punctuation():
    assert normalize_cause("Gateway/429 STORM!") == normalize_cause("gateway 429 storm")


# ---------------------------------------------------------- retrieve_runbook
def test_verbatim_symptom_matches_its_runbook_exactly():
    assert retrieve_runbook(OOM, RUNBOOKS) == ("RB-017", APPROX(1.0))


def test_the_same_incident_in_other_words_still_finds_the_runbook():
    """Operational memory обязана работать на формулировке дежурного,
    а не только на формулировке автора раннбука."""
    runbook_id, score = retrieve_runbook("vLLM OOM from KV cache spike on /api/llm", RUNBOOKS)
    assert runbook_id == "RB-017"
    assert 0.0 < score < 1.0


def test_unknown_symptom_matches_nothing():
    assert retrieve_runbook("unrelated gibberish tokens", RUNBOOKS) == (None, 0.0)


def test_empty_symptom_matches_nothing():
    assert retrieve_runbook("", RUNBOOKS) == (None, 0.0)


def test_a_wordy_runbook_does_not_win_on_volume_alone():
    """Жаккар делит на объединение — иначе длинный раннбук выигрывал бы всегда."""
    padded = dict(RUNBOOKS)
    padded["RB-999"] = {
        "symptom": OOM + " and also latency deploy gateway tenant database pool checkout",
        "action": "restart_pod",
    }
    assert retrieve_runbook(OOM, padded)[0] == "RB-017"


def test_retrieval_does_not_depend_on_dictionary_order():
    reversed_books = dict(reversed(list(RUNBOOKS.items())))
    assert retrieve_runbook(OOM, RUNBOOKS) == retrieve_runbook(OOM, reversed_books)


# -------------------------------------------------------- adversarial_review
def test_two_agents_agreeing_raise_confidence():
    review = adversarial_review([
        {"agent": "LogAgent", "root_cause": "vLLM OOM from KV cache spike", "confidence": 0.78},
        {"agent": "RunbookAgent", "root_cause": "KV cache spike, vLLM OOM", "confidence": 0.88},
    ])
    assert review["agreement"] is True
    assert review["escalate"] is False
    assert set(review["supporters"]) == {"LogAgent", "RunbookAgent"}


def test_disagreement_escalates_with_both_hypotheses_visible():
    """Расхождение — не ошибка, а фильтр против галлюцинированной первопричины."""
    review = adversarial_review([
        {"agent": "LogAgent", "root_cause": "vLLM OOM from KV cache spike", "confidence": 0.78},
        {"agent": "MetricAgent", "root_cause": "upstream DNS resolution failure", "confidence": 0.75},
    ])
    assert review["escalate"] is True
    assert review["alternatives"] != ()


def test_a_lone_agent_never_counts_as_agreement():
    """Одна модель не может согласиться сама с собой."""
    review = adversarial_review([
        {"agent": "LogAgent", "root_cause": "vLLM OOM", "confidence": 0.99},
    ])
    assert review["agreement"] is False
    assert review["escalate"] is True


def test_confidence_is_averaged_not_summed():
    """Иначе три неуверенных агента обошли бы двух уверенных, а «уверенность»
    перестала бы быть числом от нуля до единицы."""
    review = adversarial_review([
        {"agent": "a", "root_cause": "vLLM OOM", "confidence": 0.8},
        {"agent": "b", "root_cause": "vLLM OOM spike", "confidence": 0.8},
        {"agent": "c", "root_cause": "vLLM OOM spike again", "confidence": 0.8},
    ])
    assert review["confidence"] <= 1.0


def test_majority_wins_over_a_single_loud_agent():
    review = adversarial_review([
        {"agent": "a", "root_cause": "vLLM OOM from KV cache spike", "confidence": 0.6},
        {"agent": "b", "root_cause": "KV cache spike, vLLM OOM", "confidence": 0.6},
        {"agent": "c", "root_cause": "upstream DNS resolution failure", "confidence": 0.95},
    ])
    assert set(review["supporters"]) == {"a", "b"}
    assert review["agreement"] is True


def test_review_does_not_depend_on_the_order_of_hypotheses():
    """Отчёт не должен зависеть от того, какой агент ответил быстрее."""
    hypotheses = [
        {"agent": "a", "root_cause": "vLLM OOM from KV cache spike", "confidence": 0.6},
        {"agent": "b", "root_cause": "KV cache spike, vLLM OOM", "confidence": 0.6},
        {"agent": "c", "root_cause": "upstream DNS resolution failure", "confidence": 0.95},
    ]
    assert adversarial_review(hypotheses) == adversarial_review(list(reversed(hypotheses)))


def test_no_hypotheses_at_all_is_a_broken_collector():
    with pytest.raises(ValueError):
        adversarial_review([])


# ------------------------------------------------------------- is_safe_action
def test_restarting_a_pod_is_safe():
    assert is_safe_action("restart_pod") is True


def test_changing_topology_is_not_safe():
    assert is_safe_action("change_topology") is False


def test_the_safe_set_and_the_unsafe_set_do_not_overlap():
    """Ни одно действие не должно быть одновременно разрешено и запрещено."""
    assert SAFE_ACTIONS & UNSAFE_ACTIONS == frozenset()
    assert all(is_safe_action(a) is True for a in SAFE_ACTIONS)
    assert all(is_safe_action(a) is False for a in UNSAFE_ACTIONS)


def test_an_invented_action_fails_loudly_instead_of_returning_false():
    """False значит «знаем и не разрешаем». Незнакомая строка значит
    «действие придумали на ходу», и это обязано ломаться громко."""
    with pytest.raises(ValueError):
        is_safe_action("rm_rf_slash")


# -------------------------------------------------------------------- triage
def test_agreement_plus_runbook_plus_safe_action_runs_itself():
    result = triage(
        "High error rate in /checkout/generate-summary",
        [
            agent("LogAgent", OOM, 0.78),
            agent("RunbookAgent", "Under burst concurrency: KV-cache OOM", 0.88),
        ],
        RUNBOOKS,
    )
    assert result["agreement"] is True
    assert result["runbook"] == "RB-017"
    assert result["action"] == "restart_pod"
    assert result["auto_approved"] is True


def test_disagreeing_agents_never_get_auto_approval():
    result = triage(
        "incident",
        [agent("LogAgent", OOM, 0.78), agent("MetricAgent", "upstream DNS resolution failure", 0.90)],
        RUNBOOKS,
    )
    assert result["auto_approved"] is False
    assert "disagree" in result["reason"]


def test_no_matching_runbook_means_no_pre_approved_action():
    """Агент, сочиняющий действие на ходу, — то самое, из-за чего
    авторемедиацию держат узкой."""
    result = triage(
        "incident",
        [agent("a", "quantum flux inversion", 0.9), agent("b", "quantum flux inversion", 0.9)],
        RUNBOOKS,
    )
    assert result["runbook"] is None
    assert result["action"] is None
    assert result["auto_approved"] is False


def test_a_matched_runbook_with_an_unsafe_action_still_needs_a_human():
    result = triage(
        "incident",
        [
            agent("a", "database connection pool exhausted", 0.9),
            agent("b", "connection pool exhausted database", 0.9),
        ],
        RUNBOOKS,
    )
    assert result["runbook"] == "RB-055"
    assert result["action"] == "alter_database"
    assert result["auto_approved"] is False
    assert "safe set" in result["reason"]


def test_triage_is_reproducible():
    agents = [agent("a", OOM, 0.8), agent("b", "KV cache OOM burst", 0.7)]
    assert triage("incident", agents, RUNBOOKS) == triage("incident", agents, RUNBOOKS)


# --------------------------------------------------------------- bad_minutes
def test_one_five_minute_outage_costs_five_minutes():
    assert bad_minutes([(0.0, 300.0)]) == APPROX(5.0)


def test_no_outages_cost_nothing():
    assert bad_minutes([]) == APPROX(0.0)


def test_overlapping_alerts_do_not_double_charge_the_same_minute():
    assert bad_minutes([(0.0, 300.0), (60.0, 60.0)]) == APPROX(5.0)


def test_short_frequent_outages_burn_twice_as_fast_at_minute_granularity():
    """Ключевой факт урока про SLI, и он контринтуитивен.

    Один сбой на 300 секунд и десять сбоев по 30 секунд — одна и та же
    суммарная недоступность. При поминутном SLI второй сценарий стоит вдвое
    дороже: минута плохая целиком, даже если лежало полминуты.
    """
    one_long = [(0.0, 300.0)]
    ten_short = [(i * 600.0, 30.0) for i in range(10)]
    assert sum(d for _, d in one_long) == APPROX(sum(d for _, d in ten_short))
    assert bad_minutes(one_long) == APPROX(5.0)
    assert bad_minutes(ten_short) == APPROX(10.0)


def test_per_second_sli_makes_the_two_scenarios_equal_again():
    """Разница целиком в ОПРЕДЕЛЕНИИ SLI, а не в поведении сервиса.

    Это и есть ответ на вопрос «частые короткие хуже или лучше одного
    длинного»: зависит от того, что вы называете плохой единицей.
    """
    one_long = [(0.0, 300.0)]
    ten_short = [(i * 600.0, 30.0) for i in range(10)]
    assert bad_minutes(one_long, granularity_s=1.0) == APPROX(300.0)
    assert bad_minutes(ten_short, granularity_s=1.0) == APPROX(300.0)


def test_an_outage_of_zero_length_is_a_data_error():
    with pytest.raises(ValueError):
        bad_minutes([(0.0, 0.0)])


# -------------------------------------------------------------- error_budget
def test_three_nines_give_43_2_minutes_a_month():
    budget = error_budget(0.999, MONTH_S, [])
    assert budget["allowed"] == ROUGH(43.2)
    assert budget["spent"] == APPROX(0.0)
    assert budget["burn_rate"] == APPROX(0.0)


def test_a_five_minute_outage_eats_an_eighth_of_the_monthly_budget():
    budget = error_budget(0.999, MONTH_S, [(0.0, 300.0)])
    assert budget["spent"] == APPROX(5.0)
    assert budget["remaining"] == ROUGH(38.2)
    assert budget["burn_rate"] == ROUGH(5.0 / 43.2)


def test_the_budget_can_go_negative_and_that_is_the_point():
    budget = error_budget(0.999, MONTH_S, [(0.0, 3600.0)])
    assert budget["remaining"] < 0
    assert budget["burn_rate"] > 1.0


def test_the_same_outage_reads_differently_on_a_short_window():
    """Одна минута простоя: 0.023 месячного бюджета и 16.7 часового.

    Оба числа верны. Именно поэтому решение принимается по двум окнам сразу.
    """
    month = error_budget(0.999, MONTH_S, [(0.0, 60.0)])
    hour = error_budget(0.999, HOUR_S, [(0.0, 60.0)])
    assert month["burn_rate"] == ROUGH(1.0 / 43.2)
    assert hour["burn_rate"] == ROUGH(1.0 / 0.06)


def test_a_stricter_slo_leaves_a_smaller_budget():
    assert error_budget(0.9999, MONTH_S, [])["allowed"] < error_budget(0.999, MONTH_S, [])["allowed"]


def test_an_slo_of_one_hundred_percent_is_refused():
    """Нулевой бюджет — это не SLO, это обещание, которое нельзя измерить."""
    with pytest.raises(ValueError):
        error_budget(1.0, MONTH_S, [])


# ----------------------------------------------------------- release_decision
def test_calm_service_ships():
    budget = error_budget(0.999, MONTH_S, [(0.0, 300.0)])
    verdict, _ = release_decision(budget, 1.0, 1.0)
    assert verdict == "ship"


def test_both_windows_burning_freezes_releases():
    budget = error_budget(0.999, MONTH_S, [(0.0, 300.0)])
    verdict, reason = release_decision(budget, FAST_BURN + 1, SLOW_BURN + 1)
    assert verdict == "freeze"
    assert "both windows" in reason


def test_a_short_spike_alone_only_slows_releases_down():
    """Одно быстрое окно — обычно всплеск, который уже кончился. Замораживать
    по нему значит перестать выпускать вообще."""
    budget = error_budget(0.999, MONTH_S, [(0.0, 300.0)])
    verdict, _ = release_decision(budget, FAST_BURN + 5, 1.0)
    assert verdict == "slow"


def test_a_calm_fast_window_ships_even_if_the_long_one_is_elevated():
    budget = error_budget(0.999, MONTH_S, [(0.0, 300.0)])
    verdict, _ = release_decision(budget, 2.0, SLOW_BURN + 1)
    assert verdict == "ship"


def test_an_exhausted_budget_freezes_regardless_of_current_burn():
    """Бюджет — это про месяц, а не про последний спокойный час."""
    budget = error_budget(0.999, MONTH_S, [(0.0, 3600.0)])
    verdict, reason = release_decision(budget, 0.0, 0.0)
    assert verdict == "freeze"
    assert "exhausted" in reason


def test_the_short_frequent_outages_are_what_actually_freeze_the_release():
    """Складываем два факта урока: SLI поминутный, сбои короткие и частые.

    Суммарная недоступность одна и та же, а решение — разное. Именно так
    команда узнаёт, что определение SLI это не формальность.
    """
    one_long = error_budget(0.999, MONTH_S, [(0.0, 40 * 60.0)])
    many_short = error_budget(0.999, MONTH_S, [(i * 600.0, 30.0) for i in range(80)])
    assert one_long["spent"] == APPROX(40.0)
    assert many_short["spent"] == APPROX(80.0)
    assert release_decision(one_long, 1.0, 1.0)[0] == "ship"
    assert release_decision(many_short, 1.0, 1.0)[0] == "freeze"
