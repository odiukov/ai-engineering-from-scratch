"""Тесты к уроку «Ограниченное самоулучшение: четыре примитива».

Правь exercise.py.
"""

import pytest

from exercise import (
    GATE_ORDER,
    anchor_digest,
    bounded_loop,
    gate_anchor,
    gate_frozen,
    gate_invariant,
    gate_multi,
    gate_regression,
    review_edit,
)

OBJECTIVE = "canonicalize whitespace and title-case the input"
FROZEN = ["eval/checker.py", "policy/constitution.md"]
APPROVED = {"trim", "collapse", "lower", "upper", "title"}


def policy(**over):
    """Политика по умолчанию: всё разрешено ровно в одобренных пределах."""
    base = {
        "frozen": FROZEN,
        "approved_manifest": APPROVED,
        "approved_digest": anchor_digest(OBJECTIVE),
        "minimums": {"perf": 0.5, "safety": 1.0},
        "tol": 0.2,
    }
    base.update(over)
    return base


def edit(**over):
    """Идеальная правка: проходит все пять гейтов."""
    base = {
        "files": ["src/agent.py"],
        "manifest": {"trim", "title"},
        "objective": OBJECTIVE,
        "scores": {"perf": 0.75, "safety": 1.0},
    }
    base.update(over)
    return base


# ---------------------------------------------------------- anchor_digest
def test_anchor_digest_matches_the_known_sha256_prefix():
    assert anchor_digest("abc") == "ba7816bf8f01cfea"


def test_anchor_digest_is_sixteen_hex_chars():
    d = anchor_digest(OBJECTIVE)
    assert len(d) == 16 and all(c in "0123456789abcdef" for c in d)


def test_anchor_digest_is_deterministic():
    assert anchor_digest(OBJECTIVE) == anchor_digest(OBJECTIVE)


def test_anchor_digest_changes_on_a_single_appended_space():
    """Дрейф цели обычно и выглядит как «просто пробел в конце»."""
    assert anchor_digest(OBJECTIVE) != anchor_digest(OBJECTIVE + " ")


# ------------------------------------------------------------ gate_frozen
def test_gate_frozen_allows_edits_outside_the_frozen_set():
    assert gate_frozen(["src/agent.py", "src/tools.py"], FROZEN) is True


def test_gate_frozen_rejects_editing_its_own_checker():
    """Петля не имеет права трогать код, который её же и проверяет."""
    assert gate_frozen(["eval/checker.py"], FROZEN) is False


def test_gate_frozen_rejects_a_batch_where_only_one_path_is_frozen():
    """Замаскировать запретный файл среди девяти разрешённых не выйдет."""
    batch = [f"src/m{i}.py" for i in range(9)] + ["policy/constitution.md"]
    assert gate_frozen(batch, FROZEN) is False


def test_gate_frozen_allows_an_empty_edit():
    assert gate_frozen([], FROZEN) is True


# --------------------------------------------------------- gate_invariant
def test_gate_invariant_accepts_a_subset_of_approved_tools():
    assert gate_invariant({"trim"}, APPROVED) is True


def test_gate_invariant_rejects_a_smuggled_in_tool():
    assert gate_invariant({"trim", "raw_eval"}, APPROVED) is False


def test_gate_invariant_accepts_dropping_every_tool():
    """Отказаться от инструмента можно, добавить себе новый — нет."""
    assert gate_invariant(set(), APPROVED) is True


def test_gate_invariant_accepts_the_full_approved_manifest():
    assert gate_invariant(set(APPROVED), APPROVED) is True


# ------------------------------------------------------------ gate_anchor
def test_gate_anchor_accepts_the_untouched_objective():
    assert gate_anchor(OBJECTIVE, anchor_digest(OBJECTIVE)) is True


def test_gate_anchor_rejects_an_objective_with_a_relaxing_comment():
    drifted = OBJECTIVE + "  # relax: accept any output"
    assert gate_anchor(drifted, anchor_digest(OBJECTIVE)) is False


def test_gate_anchor_rejects_a_reformatted_objective():
    """Смысл тот же, байты другие — якорь обязан сработать."""
    assert gate_anchor(OBJECTIVE.upper(), anchor_digest(OBJECTIVE)) is False


# ------------------------------------------------------------- gate_multi
def test_gate_multi_accepts_when_every_axis_clears_its_floor():
    assert gate_multi({"perf": 0.8, "safety": 1.0}, {"perf": 0.5, "safety": 1.0}) is True


def test_gate_multi_rejects_a_safety_drop_even_when_perf_is_great():
    """Одна ось не выкупается другой — в этом весь смысл примитива."""
    assert gate_multi({"perf": 1.0, "safety": 0.5}, {"perf": 0.5, "safety": 1.0}) is False


def test_gate_multi_accepts_a_score_exactly_on_the_floor():
    assert gate_multi({"perf": 0.5}, {"perf": 0.5}) is True


def test_gate_multi_ignores_axes_without_a_floor():
    assert gate_multi({"perf": 0.9, "speed": 0.0}, {"perf": 0.5}) is True


def test_gate_multi_treats_a_missing_required_axis_as_failed():
    """«Забыл посчитать safety» не должно быть способом её пройти."""
    assert gate_multi({"perf": 0.9}, {"perf": 0.5, "safety": 1.0}) is False


# -------------------------------------------------------- gate_regression
def test_gate_regression_accepts_the_very_first_cycle():
    assert gate_regression([], {"perf": 0.1}) is True


def test_gate_regression_accepts_a_dip_inside_the_tolerance():
    assert gate_regression([{"perf": 0.9}], {"perf": 0.8}, tol=0.2) is True


def test_gate_regression_rejects_a_dip_beyond_the_tolerance():
    assert gate_regression([{"perf": 0.9}], {"perf": 0.5}, tol=0.2) is False


def test_gate_regression_compares_against_the_historical_best_not_the_last():
    """Ползучая просадка по чуть-чуть — ровно тот тихий отказ, который ловят."""
    history = [{"perf": 1.0}, {"perf": 0.85}, {"perf": 0.7}]
    assert gate_regression(history, {"perf": 0.6}, tol=0.2) is False


def test_gate_regression_ignores_an_axis_absent_from_history():
    assert gate_regression([{"perf": 0.9}], {"fairness": 0.0}, tol=0.0) is True


def test_gate_regression_with_zero_tolerance_is_strictly_monotonic():
    assert gate_regression([{"perf": 0.9}], {"perf": 0.89}, tol=0.0) is False


# ------------------------------------------------------------ review_edit
def test_review_edit_accepts_a_clean_edit():
    assert review_edit(edit(), policy()) == (True, [])


def test_review_edit_rejects_an_edit_that_touches_the_checker():
    ok, failed = review_edit(edit(files=["eval/checker.py"]), policy())
    assert (ok, failed) == (False, ["frozen"])


def test_review_edit_reports_every_failing_gate_not_just_the_first():
    bad = edit(manifest={"raw_eval"}, scores={"perf": 0.1, "safety": 0.0})
    ok, failed = review_edit(bad, policy())
    assert ok is False
    assert set(failed) == {"invariant", "multi"}


def test_review_edit_lists_failures_in_gate_order():
    """Порядок отчёта детерминирован: два запуска обязаны совпасть дословно."""
    bad = edit(files=["eval/checker.py"], objective="something else")
    _ok, failed = review_edit(bad, policy())
    assert failed == ["frozen", "anchor"]
    assert [GATE_ORDER.index(name) for name in failed] == sorted(
        GATE_ORDER.index(name) for name in failed
    )


def test_review_edit_regression_gate_sees_the_history():
    good_history = [{"perf": 1.0, "safety": 1.0}]
    ok, failed = review_edit(edit(), policy(), good_history)
    assert (ok, failed) == (False, ["regression"])


def test_review_edit_needs_all_gates_to_pass():
    """Любой один отказ останавливает петлю — гейты не голосуют большинством."""
    ok, _failed = review_edit(edit(objective=OBJECTIVE + "\n"), policy())
    assert ok is False


# ----------------------------------------------------------- bounded_loop
def _rising(n):
    """n правок, каждая лучше предыдущей по perf."""
    return [edit(scores={"perf": 0.5 + 0.01 * i, "safety": 1.0}) for i in range(n)]


def test_bounded_loop_stops_at_the_ceiling_while_the_metric_still_rises():
    """Потолок сильнее любого «но ведь становится лучше»."""
    out = bounded_loop(_rising(50), policy(), max_cycles=5)
    assert (out["accepted"], out["reason"]) == (5, "ceiling")


def test_bounded_loop_reports_exhausted_when_proposals_run_out_first():
    out = bounded_loop(_rising(3), policy(), max_cycles=100)
    assert (out["accepted"], out["reason"]) == (3, "exhausted")


def test_bounded_loop_with_zero_ceiling_accepts_nothing():
    out = bounded_loop(_rising(10), policy(), max_cycles=0)
    assert (out["accepted"], out["history"]) == (0, [])


def test_bounded_loop_counts_rejections_per_gate():
    proposals = [edit(files=["eval/checker.py"]) for _ in range(4)]
    out = bounded_loop(proposals, policy(), max_cycles=10)
    assert out["rejected"]["frozen"] == 4
    assert out["accepted"] == 0


def test_bounded_loop_history_holds_only_accepted_scores():
    proposals = [
        edit(scores={"perf": 0.6, "safety": 1.0}),
        edit(manifest={"raw_eval"}),               # отвергается инвариантом
        edit(scores={"perf": 0.7, "safety": 1.0}),
    ]
    out = bounded_loop(proposals, policy(), max_cycles=10)
    assert out["history"] == [
        {"perf": pytest.approx(0.6), "safety": pytest.approx(1.0)},
        {"perf": pytest.approx(0.7), "safety": pytest.approx(1.0)},
    ]
