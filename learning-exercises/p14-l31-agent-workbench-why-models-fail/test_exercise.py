"""Тесты к уроку «Воркбенч агента: почему сильные модели всё равно ошибаются».

Правь exercise.py.
"""

import pytest

from exercise import (
    FAILURE_MODES,
    SURFACES,
    acceptance_status,
    classify_failures,
    failure_report,
    missing_surfaces,
    off_scope_writes,
    repeated_steps,
    surfaces_to_fix,
    weakest_surface,
)


def step(action, target, ok=True):
    """Короткий конструктор шага трейса."""
    return {"action": action, "target": target, "ok": ok}


ALLOWED = ["app.py", "test_app.py"]
ACCEPT = ["pytest test_app.py"]


# -------------------------------------------------------- missing_surfaces
def test_missing_surfaces_lists_everything_absent():
    assert missing_surfaces(["scope", "state"]) == [
        "instructions",
        "feedback",
        "verification",
        "review",
        "handoff",
    ]


def test_full_workbench_misses_nothing():
    assert missing_surfaces(SURFACES) == []


def test_missing_surfaces_keeps_canonical_order_not_alphabetical():
    """Порядок таблицы урока, а не буквенный: instructions раньше feedback."""
    result = missing_surfaces([])
    assert result == list(SURFACES)
    assert result != sorted(result)


# --------------------------------------------------------- weakest_surface
def test_weakest_surface_picks_the_lowest_score():
    assert weakest_surface({"scope": 2, "state": 0, "review": 1}) == "state"


def test_weakest_surface_breaks_ties_by_canonical_order():
    """Ничья 1:1 — чиним ту, что раньше в SURFACES, а не раньше в алфавите."""
    assert weakest_surface({"handoff": 1, "state": 1}) == "state"


def test_weakest_surface_rejects_unknown_surface():
    with pytest.raises(ValueError):
        weakest_surface({"vibes": 0})


def test_weakest_surface_rejects_empty_audit():
    with pytest.raises(ValueError):
        weakest_surface({})


# ---------------------------------------------------------- repeated_steps
def test_repeated_steps_flags_three_identical_reads():
    trace = [step("read", "app.py")] * 3
    assert repeated_steps(trace) == [("read", "app.py")]


def test_two_repeats_are_iteration_not_a_loop():
    trace = [step("read", "app.py")] * 2
    assert repeated_steps(trace) == []


def test_repeated_steps_separates_action_from_target():
    """Три чтения и три записи одного файла — два разных зацикливания."""
    trace = [step("read", "app.py")] * 3 + [step("write", "app.py")] * 3
    assert repeated_steps(trace) == [("read", "app.py"), ("write", "app.py")]


def test_repeated_steps_counts_non_adjacent_occurrences():
    """Зацикливание не обязано быть подряд: агент кружит между двумя файлами."""
    trace = [step("read", "a.py"), step("read", "b.py")] * 3
    assert repeated_steps(trace) == [("read", "a.py"), ("read", "b.py")]


def test_repeated_steps_threshold_is_tunable():
    trace = [step("read", "app.py")] * 2
    assert repeated_steps(trace, threshold=2) == [("read", "app.py")]


# --------------------------------------------------------- off_scope_writes
def test_off_scope_write_is_caught():
    trace = [step("write", "app.py"), step("write", "README.md")]
    assert off_scope_writes(trace, ALLOWED) == ["README.md"]


def test_reading_a_file_outside_scope_is_not_a_violation():
    """Область ограничивает запись, а не чтение."""
    assert off_scope_writes([step("read", "scripts/release.sh")], ALLOWED) == []


def test_off_scope_writes_deduplicate_but_keep_first_seen_order():
    trace = [
        step("write", "scripts/release.sh"),
        step("write", "README.md"),
        step("write", "scripts/release.sh"),
    ]
    assert off_scope_writes(trace, ALLOWED) == ["scripts/release.sh", "README.md"]


def test_a_run_entirely_inside_scope_reports_nothing():
    trace = [step("write", "app.py"), step("write", "test_app.py")]
    assert off_scope_writes(trace, ALLOWED) == []


# ------------------------------------------------------- acceptance_status
def test_criterion_never_run_is_not_run():
    assert acceptance_status([], ACCEPT) == {"pytest test_app.py": "not_run"}


def test_criterion_that_failed_is_failed():
    trace = [step("run", "pytest test_app.py", ok=False)]
    assert acceptance_status(trace, ACCEPT) == {"pytest test_app.py": "failed"}


def test_a_green_rerun_after_a_red_one_counts_as_passed():
    """Агент чинил и починил — итог зелёный, а не красный."""
    trace = [
        step("run", "pytest test_app.py", ok=False),
        step("run", "pytest test_app.py", ok=True),
    ]
    assert acceptance_status(trace, ACCEPT) == {"pytest test_app.py": "passed"}


def test_empty_acceptance_gives_empty_status():
    assert acceptance_status([step("run", "pytest", ok=True)], []) == {}


# ------------------------------------------------------- classify_failures
def test_stopping_without_running_the_tests_is_premature_stop():
    trace = [step("write", "app.py"), step("stop", "done")]
    assert classify_failures(trace, ALLOWED, ACCEPT) == ["premature_stop"]


def test_declaring_success_on_red_tests_is_unverified_success():
    trace = [
        step("run", "pytest test_app.py", ok=False),
        step("stop", "done", ok=True),
    ]
    assert classify_failures(trace, ALLOWED, ACCEPT) == ["unverified_success"]


def test_premature_stop_and_unverified_success_never_fire_together():
    """Разные поломки: «не проверял» и «проверил и соврал» исключают друг друга."""
    trace = [
        step("run", "pytest test_app.py", ok=False),
        step("stop", "done", ok=True),
    ]
    modes = classify_failures(trace, ALLOWED, ACCEPT + ["ruff check"])
    assert modes == ["premature_stop"]


def test_looping_is_told_apart_from_stopping_early():
    """Зацикливание видно и в прогоне, который довёл приёмку до зелёного."""
    trace = [step("read", "app.py")] * 3 + [
        step("run", "pytest test_app.py", ok=True),
        step("stop", "done", ok=True),
    ]
    assert classify_failures(trace, ALLOWED, ACCEPT) == ["loop"]


def test_an_unfinished_trace_is_not_accused_of_stopping_early():
    """Нет шага stop — прогон ещё идёт, обвинять не в чем."""
    assert classify_failures([step("write", "app.py")], ALLOWED, ACCEPT) == []


def test_several_modes_come_back_in_canonical_order():
    trace = [step("read", "app.py")] * 3 + [
        step("write", "scripts/release.sh"),
        step("stop", "done"),
    ]
    assert classify_failures(trace, ALLOWED, ACCEPT) == [
        "loop",
        "premature_stop",
        "scope_creep",
    ]


def test_a_healthy_run_has_no_failure_modes():
    trace = [
        step("write", "app.py"),
        step("write", "test_app.py"),
        step("run", "pytest test_app.py", ok=True),
        step("stop", "done", ok=True),
    ]
    assert classify_failures(trace, ALLOWED, ACCEPT) == []


# ---------------------------------------------------------- surfaces_to_fix
def test_each_failure_mode_maps_to_a_surface():
    assert surfaces_to_fix(["scope_creep", "loop"]) == ["state", "scope"]


def test_no_failures_means_nothing_to_fix():
    assert surfaces_to_fix([]) == []


def test_every_declared_failure_mode_has_a_surface():
    """Ни один режим из FAILURE_MODES не должен остаться без поверхности."""
    assert len(surfaces_to_fix(list(FAILURE_MODES))) == 4


def test_unknown_failure_mode_is_refused():
    with pytest.raises(ValueError):
        surfaces_to_fix(["vibes_off"])


# ------------------------------------------------------------ failure_report
def test_report_of_a_healthy_run_is_clean():
    trace = [
        step("write", "app.py"),
        step("run", "pytest test_app.py", ok=True),
        step("stop", "done", ok=True),
    ]
    assert failure_report(trace, ALLOWED, ACCEPT)["clean"] is True


def test_report_names_both_the_symptom_and_the_surface():
    trace = [step("write", "scripts/release.sh"), step("stop", "done")]
    report = failure_report(trace, ALLOWED, ACCEPT)
    assert "scope_creep" in report["modes"]
    assert "scope" in report["surfaces_to_fix"]
    assert report["off_scope_writes"] == ["scripts/release.sh"]


def test_prompt_only_run_fails_on_more_surfaces_than_the_workbench_run():
    """Тот же агент, та же задача: без поверхностей ломается больше мест."""
    prompt_only = [
        step("read", "app.py"),
        step("read", "app.py"),
        step("read", "app.py"),
        step("write", "app.py"),
        step("write", "README.md"),
        step("stop", "done", ok=True),
    ]
    workbench = [
        step("write", "app.py"),
        step("write", "test_app.py"),
        step("run", "pytest test_app.py", ok=True),
        step("stop", "done", ok=True),
    ]
    weak = failure_report(prompt_only, ALLOWED, ACCEPT)
    strong = failure_report(workbench, ALLOWED, ACCEPT)
    assert len(weak["surfaces_to_fix"]) > len(strong["surfaces_to_fix"])
    assert strong["clean"] is True


def test_report_carries_the_acceptance_breakdown():
    trace = [step("run", "pytest test_app.py", ok=False), step("stop", "done")]
    assert failure_report(trace, ALLOWED, ACCEPT)["acceptance"] == {
        "pytest test_app.py": "failed"
    }
