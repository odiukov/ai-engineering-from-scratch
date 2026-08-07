"""Тесты к уроку «Хендофф между сессиями». Правь exercise.py."""

import pytest

from exercise import (
    CLEAN_CHECKS,
    HANDOFF_FIELDS,
    build_handoff,
    choose_next_action,
    clean_state_issues,
    derive_open_risks,
    render_markdown,
    resume_blockers,
    select_active_handoff,
    trim_feedback,
)

NOW = "2026-08-07T10:00:00"


def _snapshot(**over):
    snap = {
        "task_id": "T-17",
        "topic": "signup-validation",
        "last_known_good_commit": "abc1234",
        "state": {
            "summary": "Добавил валидацию пароля в /signup.",
            "commands_run": ["pytest -q", "ruff check ."],
            "failed_attempts": ["пробовал pydantic v1 API — его нет в проекте"],
        },
        "verdict": {"status": "pass", "report_path": "outputs/verification_report.json",
                    "findings": [{"severity": "info", "detail": "3 теста"}]},
        "review": {"report_path": "outputs/review_report.json", "findings": []},
        "feedback": [{"cmd": "pytest -q", "exit_code": 0}],
        "diff_summary": {"changed": ["app.py", "test_app.py"]},
    }
    snap.update(over)
    return snap


def _workbench(**over):
    wb = {
        "uncommitted_files": [],
        "stash_note": None,
        "temp_artifacts": [],
        "tests": {"status": "green", "failure": ""},
        "feature_board": [{"id": "F1", "status": "done", "actual_done": True}],
        "branch": "feat/signup",
        "expected_branch": "feat/signup",
        "orphan_branches": [],
    }
    wb.update(over)
    return wb


# ------------------------------------------------------------ trim_feedback
def test_trim_feedback_keeps_the_last_k_records():
    recs = [{"cmd": c, "exit_code": 0} for c in "abcde"]
    assert trim_feedback(recs, 2) == [{"cmd": "d", "exit_code": 0}, {"cmd": "e", "exit_code": 0}]


def test_trim_feedback_keeps_old_failures_outside_the_tail():
    """Провал в начале журнала — ровно то, ради чего пакет и читают."""
    recs = [{"cmd": "a", "exit_code": 1}] + [{"cmd": c, "exit_code": 0} for c in "bcdefg"]
    kept = trim_feedback(recs, 2)
    assert kept[0] == {"cmd": "a", "exit_code": 1}
    assert [r["cmd"] for r in kept] == ["a", "f", "g"]


def test_trim_feedback_does_not_duplicate_a_failure_inside_the_tail():
    recs = [{"cmd": "a", "exit_code": 0}, {"cmd": "b", "exit_code": 2}]
    assert trim_feedback(recs, 2) == recs


def test_trim_feedback_rejects_negative_tail():
    with pytest.raises(ValueError):
        trim_feedback([{"cmd": "a", "exit_code": 0}], -1)


# -------------------------------------------------------- derive_open_risks
def test_open_risks_put_blockers_before_warnings():
    verdict = {"findings": [{"severity": "warn", "detail": "slow test"}]}
    review = {"findings": [{"severity": "block", "detail": "no rollback"}]}
    risks = derive_open_risks(verdict, review)
    assert [r["severity"] for r in risks] == ["block", "warn"]


def test_open_risks_drop_info_findings():
    verdict = {"findings": [{"severity": "info", "detail": "3 файла"}]}
    assert derive_open_risks(verdict, {}) == []


def test_open_risks_tag_the_report_they_came_from():
    verdict = {"findings": [{"severity": "warn", "detail": "x"}]}
    review = {"findings": [{"severity": "warn", "detail": "y"}]}
    sources = {r["detail"]: r["source"] for r in derive_open_risks(verdict, review)}
    assert sources == {"x": "verification", "y": "review"}


def test_open_risks_order_does_not_depend_on_input_order():
    """Одинаковая severity — порядок всё равно обязан быть фиксированным."""
    a = {"severity": "warn", "detail": "aaa"}
    b = {"severity": "warn", "detail": "bbb"}
    assert derive_open_risks({"findings": [a, b]}, {}) == derive_open_risks(
        {"findings": [b, a]}, {}
    )


# ------------------------------------------------------- choose_next_action
def test_next_action_prefers_a_blocking_risk():
    risks = [{"severity": "block", "detail": "no rollback", "source": "review"}]
    board = [{"id": "F1", "status": "todo", "title": "валидация"}]
    assert "no rollback" in choose_next_action({"status": "pass"}, risks, board)


def test_next_action_falls_back_to_reverification_when_verdict_is_not_pass():
    verdict = {"status": "fail", "report_path": "outputs/verification_report.json"}
    action = choose_next_action(verdict, [], [{"id": "F1", "status": "todo"}])
    assert "outputs/verification_report.json" in action


def test_next_action_continues_in_progress_before_starting_todo():
    board = [
        {"id": "F1", "status": "todo", "title": "первая"},
        {"id": "F2", "status": "in_progress", "title": "вторая"},
    ]
    assert "F2" in choose_next_action({"status": "pass"}, [], board)


def test_next_action_mentions_the_warning_when_only_warnings_remain():
    risks = [{"severity": "warn", "detail": "тест на 40 секунд", "source": "verification"}]
    assert "тест на 40 секунд" in choose_next_action({"status": "pass"}, risks, [])


def test_next_action_is_never_empty():
    """Пакет без next_action — статус-репорт, а не хендофф."""
    assert choose_next_action({"status": "pass"}, [], []).strip() != ""


# ------------------------------------------------------- clean_state_issues
def test_clean_workbench_has_no_issues():
    assert clean_state_issues(_workbench()) == []


def test_uncommitted_files_block_unless_stashed_with_a_note():
    dirty = _workbench(uncommitted_files=["app.py"])
    assert [i["check"] for i in clean_state_issues(dirty)] == ["working_tree"]
    excused = _workbench(uncommitted_files=["app.py"], stash_note="отложено до ревью")
    assert clean_state_issues(excused) == []


def test_red_test_named_in_open_risks_is_not_a_blocker():
    """Урок разрешает уйти с красным тестом, но только честно записанным."""
    wb = _workbench(tests={"status": "red", "failure": "test_signup_422"})
    assert [i["check"] for i in clean_state_issues(wb)] == ["tests"]
    named = [{"severity": "warn", "detail": "test_signup_422", "source": "verification"}]
    assert clean_state_issues(wb, named) == []


def test_stale_feature_board_is_detected_in_both_directions():
    lying_done = _workbench(feature_board=[{"id": "F1", "status": "done", "actual_done": False}])
    assert [i["check"] for i in clean_state_issues(lying_done)] == ["feature_board"]
    lying_todo = _workbench(feature_board=[{"id": "F2", "status": "todo", "actual_done": True}])
    assert [i["check"] for i in clean_state_issues(lying_todo)] == ["feature_board"]


def test_issues_follow_the_documented_check_order():
    wb = _workbench(
        uncommitted_files=["app.py"],
        temp_artifacts=["scratch.tmp"],
        branch="main",
        expected_branch="feat/signup",
    )
    checks = [i["check"] for i in clean_state_issues(wb)]
    assert checks == sorted(checks, key=CLEAN_CHECKS.index)
    assert checks == ["working_tree", "temp_artifacts", "branch"]


# ------------------------------------------------------------ build_handoff
def test_build_handoff_fills_all_seven_fields():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    assert all(field in packet for field in HANDOFF_FIELDS)
    assert packet["changed_files"] == ["app.py", "test_app.py"]


def test_build_handoff_refuses_a_dirty_workbench():
    """Хендофф на грязном дереве — переадресованный беспорядок, не хендофф."""
    with pytest.raises(ValueError):
        build_handoff(_snapshot(), _workbench(temp_artifacts=["scratch.tmp"]), NOW)


def test_build_handoff_takes_the_clock_as_an_argument():
    packet = build_handoff(_snapshot(), _workbench(), "2020-01-01T00:00:00")
    assert packet["generated_at"] == "2020-01-01T00:00:00"


def test_build_handoff_is_idempotent_for_the_same_clock():
    first = build_handoff(_snapshot(), _workbench(), NOW)
    second = build_handoff(_snapshot(), _workbench(), NOW)
    assert first == second


def test_build_handoff_packet_is_enough_to_resume_without_the_first_session():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    assert resume_blockers(packet) == []


# ----------------------------------------------------------- render_markdown
def test_markdown_has_a_section_for_every_handoff_field():
    text = render_markdown(build_handoff(_snapshot(), _workbench(), NOW))
    assert text.startswith("# Handoff T-17")
    for field in HANDOFF_FIELDS:
        assert ("## " + field) in text


def test_markdown_shows_none_for_empty_fields_instead_of_hiding_them():
    snap = _snapshot(diff_summary={"changed": []})
    text = render_markdown(build_handoff(snap, _workbench(), NOW))
    body = text.split("## changed_files")[1].split("##")[0]
    assert "_none_" in body


def test_markdown_is_stable_across_calls():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    assert render_markdown(packet) == render_markdown(packet)


# ----------------------------------------------------------- resume_blockers
def test_status_report_without_next_action_is_not_resumable():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    packet["next_action"] = ""
    assert resume_blockers(packet) == ["поле next_action пустое"]


def test_dropped_field_is_reported_by_name():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    del packet["open_risks"]
    assert "нет поля open_risks" in resume_blockers(packet)


def test_missing_verdict_pointer_link_is_a_blocker():
    snap = _snapshot(review={"report_path": "", "findings": []})
    blockers = resume_blockers(build_handoff(snap, _workbench(), NOW))
    assert "verdict_pointer без ссылки review" in blockers


def test_missing_branch_is_a_blocker():
    packet = build_handoff(_snapshot(), _workbench(), NOW)
    packet["branch"] = None
    assert "нет branch" in resume_blockers(packet)


# ----------------------------------------------------- select_active_handoff
def _packet(task_id, generated_at, status="active", branch="main", topic="auth"):
    return {"task_id": task_id, "generated_at": generated_at, "status": status,
            "branch": branch, "topic": topic}


def test_newest_packet_becomes_active_and_the_rest_superseded():
    packets = [_packet("T-1", "2026-08-01"), _packet("T-2", "2026-08-05")]
    out = select_active_handoff(packets, "main", "auth")
    assert {p["task_id"]: p["status"] for p in out} == {"T-1": "superseded", "T-2": "active"}


def test_archived_packet_never_becomes_active():
    packets = [_packet("T-1", "2026-08-01"), _packet("T-9", "2026-08-09", status="archived")]
    out = select_active_handoff(packets, "main", "auth")
    assert {p["task_id"]: p["status"] for p in out} == {"T-1": "active", "T-9": "archived"}


def test_packets_of_other_branches_are_untouched_and_input_is_not_mutated():
    other = _packet("T-3", "2026-08-09", branch="feat/x")
    packets = [_packet("T-1", "2026-08-01"), _packet("T-2", "2026-08-05"), other]
    out = select_active_handoff(packets, "main", "auth")
    assert {p["task_id"]: p["status"] for p in out} == {
        "T-1": "superseded",
        "T-2": "active",
        "T-3": "active",
    }
    assert packets[0]["status"] == "active"


def test_tie_on_timestamp_is_resolved_deterministically():
    a, b = _packet("T-A", "2026-08-05"), _packet("T-B", "2026-08-05")
    forward = select_active_handoff([a, b], "main", "auth")
    backward = select_active_handoff([b, a], "main", "auth")
    assert {p["task_id"]: p["status"] for p in forward} == {
        p["task_id"]: p["status"] for p in backward
    }
