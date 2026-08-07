"""Тесты к уроку «Eval-driven разработка агентов». Правь exercise.py."""

import pytest

from exercise import (
    CASE_LAYERS,
    ci_gate,
    coverage_gaps,
    detect_regression,
    evaluator_optimizer,
    flaky_cases,
    run_case,
    run_suite,
    summarize,
)

CITE_CASE = {
    "id": "custom_c001",
    "layer": "custom",
    "prompt": "перескажи ReAct",
    "expect": "arXiv",
    "topics": ("citations",),
}
PII_CASE = {
    "id": "online_o001",
    "layer": "online",
    "prompt": "перешли ssn 123-45-6789",
    "expect": "refused",
    "topics": ("guardrails",),
}


def _result(cid, layer, passed):
    """Готовый результат кейса — чтобы не гонять агента там, где важна аналитика."""
    return {"id": cid, "layer": layer, "passed": passed, "answer": "", "error": None}


# ------------------------------------------------------------------ run_case
def test_case_passes_when_the_expected_text_appears():
    result = run_case(CITE_CASE, lambda prompt: "см. arXiv:2210.03629")
    assert (result["passed"], result["id"], result["error"]) == (
        True,
        "custom_c001",
        None,
    )


def test_case_fails_when_the_expectation_is_missing():
    result = run_case(CITE_CASE, lambda prompt: "ReAct это про рассуждения")
    assert result["passed"] is False


def test_crashing_agent_fails_the_case_instead_of_the_suite():
    """Один кривой кейс не имеет права уронить прогон в CI."""

    def broken_agent(prompt):
        raise LookupError("инструмент недоступен")

    result = run_case(CITE_CASE, broken_agent)
    assert result["passed"] is False
    assert result["answer"] == ""
    assert "LookupError" in result["error"]


def test_unknown_layer_is_value_error():
    with pytest.raises(ValueError):
        run_case({**CITE_CASE, "layer": "vibes"}, lambda prompt: "arXiv")


# ----------------------------------------------------------------- run_suite
def test_suite_keeps_the_case_order():
    results = run_suite([CITE_CASE, PII_CASE], lambda prompt: "arXiv и refused")
    assert [r["id"] for r in results] == ["custom_c001", "online_o001"]


def test_one_failing_case_does_not_stop_the_others():
    results = run_suite([CITE_CASE, PII_CASE], lambda prompt: "refused")
    assert [r["passed"] for r in results] == [False, True]


def test_duplicate_case_ids_are_value_error():
    """Базовая линия хранится по id: близнецы затрут друг друга молча."""
    with pytest.raises(ValueError):
        run_suite([CITE_CASE, dict(CITE_CASE)], lambda prompt: "arXiv")


def test_empty_suite_gives_empty_results():
    assert run_suite([], lambda prompt: "arXiv") == []


# ----------------------------------------------------------------- summarize
def test_summary_counts_passes():
    report = summarize([_result("a", "custom", True), _result("b", "custom", False)])
    assert (report["total"], report["passed"], report["rate"]) == (2, 1, 0.5)


def test_summary_splits_by_layer():
    report = summarize(
        [
            _result("a", "custom", True),
            _result("b", "online", False),
            _result("c", "online", True),
        ]
    )
    assert report["by_layer"] == {
        "custom": {"total": 1, "passed": 1},
        "online": {"total": 2, "passed": 1},
    }


def test_layer_without_cases_is_absent_not_green():
    """Пустой слой — отсутствие проверки, а не сто процентов прохождения."""
    report = summarize([_result("a", "custom", True)])
    assert "benchmark" not in report["by_layer"]


def test_empty_suite_is_value_error():
    with pytest.raises(ValueError):
        summarize([])


# ---------------------------------------------------------- detect_regression
def test_case_that_went_red_is_reported_as_broken():
    diff = detect_regression({"c1": True}, [_result("c1", "custom", False)])
    assert diff["broken"] == ["c1"]


def test_average_can_stay_flat_while_a_case_breaks():
    """Средняя доля прохождения компенсирует поломку чужим успехом. Кейсы — нет."""
    baseline = {"c1": True, "c2": False}
    results = [_result("c1", "custom", False), _result("c2", "custom", True)]
    assert summarize(results)["rate"] == 0.5 == sum(baseline.values()) / len(baseline)
    diff = detect_regression(baseline, results)
    assert (diff["broken"], diff["fixed"]) == (["c1"], ["c2"])


def test_fixed_case_is_not_a_regression():
    diff = detect_regression({"c1": False}, [_result("c1", "custom", True)])
    assert diff["broken"] == []
    assert diff["fixed"] == ["c1"]


def test_deleted_case_is_reported_as_missing():
    """Убрать красный кейс из суиты дешевле, чем починить, — так и делают."""
    diff = detect_regression({"c1": True, "c2": True}, [_result("c1", "custom", True)])
    assert diff["missing"] == ["c2"]


def test_brand_new_case_is_reported_as_new():
    diff = detect_regression({}, [_result("c9", "benchmark", False)])
    assert (diff["new"], diff["broken"]) == (["c9"], [])


# ------------------------------------------------------------------- ci_gate
def test_gate_allows_a_clean_run():
    gate = ci_gate([_result("c1", "custom", True)], {"c1": True})
    assert gate["allowed"] is True


def test_gate_blocks_a_broken_case_even_when_the_average_holds():
    gate = ci_gate(
        [_result("c1", "custom", False), _result("c2", "custom", True)],
        {"c1": True, "c2": False},
    )
    assert gate["allowed"] is False
    assert "c1" in gate["reason"]


def test_gate_blocks_a_rate_drop_from_new_red_cases():
    """broken пуст — просели новые кейсы. Порог ловит именно это."""
    results = [_result("c1", "custom", True)] + [
        _result(cid, "custom", False) for cid in ("c2", "c3", "c4")
    ]
    assert ci_gate(results, {"c1": True})["allowed"] is False


def test_gate_blocks_an_empty_run():
    assert ci_gate([], {"c1": True})["allowed"] is False


def test_negative_threshold_is_value_error():
    with pytest.raises(ValueError):
        ci_gate([_result("c1", "custom", True)], {"c1": True}, max_rate_drop=-0.1)


# -------------------------------------------------------- evaluator_optimizer
def test_loop_stops_as_soon_as_the_judge_passes():
    calls = []

    def propose(feedback):
        calls.append(feedback)
        return "ответ с arXiv"

    loop = evaluator_optimizer(propose, lambda c: ("arXiv" in c, "ссылка есть"))
    assert (loop["passed"], loop["rounds"]) == (True, 1)
    assert calls == [None]


def test_feedback_from_the_judge_reaches_the_next_proposal():
    """Без передачи замечания это не петля, а три попытки наугад."""
    seen = []

    def propose(feedback):
        seen.append(feedback)
        return "ответ с arXiv" if feedback == "нет ссылки" else "ответ без ссылки"

    def judge(candidate):
        return ("arXiv" in candidate, "ссылка есть" if "arXiv" in candidate else "нет ссылки")

    loop = evaluator_optimizer(propose, judge)
    assert seen == [None, "нет ссылки"]
    assert (loop["passed"], loop["rounds"]) == (True, 2)


def test_loop_gives_up_after_max_rounds():
    loop = evaluator_optimizer(lambda fb: "мимо", lambda c: (False, "всё ещё мимо"), 4)
    assert (loop["passed"], loop["rounds"], loop["reason"]) == (False, 4, "всё ещё мимо")


def test_history_records_every_candidate():
    counter = [0]

    def propose(feedback):
        counter[0] += 1
        return f"попытка {counter[0]}"

    loop = evaluator_optimizer(propose, lambda c: (False, "нет"), max_rounds=3)
    assert loop["history"] == ["попытка 1", "попытка 2", "попытка 3"]


def test_max_rounds_below_one_is_value_error():
    with pytest.raises(ValueError):
        evaluator_optimizer(lambda fb: "x", lambda c: (True, "ok"), max_rounds=0)


# --------------------------------------------------------------- flaky_cases
def test_case_with_mixed_outcomes_is_flaky():
    assert flaky_cases([{"c1": True, "c2": True}, {"c1": False, "c2": True}]) == ["c1"]


def test_stable_failure_is_a_regression_not_flakiness():
    """Стабильно красный кейс лечится кодом, мигающий — фиксацией seed."""
    assert flaky_cases([{"c1": False}, {"c1": False}, {"c1": False}]) == []


def test_single_run_cannot_prove_flakiness():
    assert flaky_cases([{"c1": True, "c2": False}]) == []


def test_flaky_ids_come_back_sorted():
    runs = [{"zebra": True, "alpha": True}, {"zebra": False, "alpha": False}]
    assert flaky_cases(runs) == ["alpha", "zebra"]


# ------------------------------------------------------------ coverage_gaps
def test_topic_without_a_case_is_a_gap():
    gaps = coverage_gaps([CITE_CASE], ("citations", "memory", "budget"))
    assert gaps["uncovered_topics"] == ["budget", "memory"]


def test_covered_topic_is_not_a_gap():
    gaps = coverage_gaps([CITE_CASE, PII_CASE], ("citations", "guardrails"))
    assert gaps["uncovered_topics"] == []


def test_layer_without_a_case_is_reported():
    """Стопроцентная суита из одного слоя проверяет один слой, а не всё."""
    assert coverage_gaps([CITE_CASE], ()) == {
        "uncovered_topics": [],
        "empty_layers": ["benchmark", "online"],
    }


def test_every_layer_covered_leaves_no_empty_layer():
    cases = [
        {"id": f"c{i}", "layer": layer, "prompt": "", "expect": ""}
        for i, layer in enumerate(CASE_LAYERS)
    ]
    assert coverage_gaps(cases, ())["empty_layers"] == []
