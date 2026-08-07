"""Тесты к уроку «Контракты области изменений». Правь exercise.py."""

import pytest

from exercise import (
    SEVERITIES,
    classify_write,
    contract_gaps,
    merge_contracts,
    merge_egress,
    path_matches,
    pick_feature,
    scope_check,
)

CONTRACT = {
    "task_id": "T-001",
    "goal": "добавить валидацию в /signup",
    "allowed_files": ["app.py", "tests/**/*.py"],
    "forbidden_files": ["migrations/**", "scripts/**"],
    "soft_files": ["docs/**", "README.md"],
    "acceptance_criteria": ["pytest -q tests/test_signup.py"],
    "rollback_plan": "откатить коммит и задеплоить прошлый тег",
    "approvals_required": [],
    "time_budget_minutes": 30,
    "network_egress": ["api.anthropic.com"],
    "violation_budget": 0,
}
CLEAN_RUN = {
    "touched_files": ["app.py", "tests/test_signup.py"],
    "commands_run": ["pytest -q tests/test_signup.py"],
    "elapsed_minutes": 12.0,
    "network_hosts": ["api.anthropic.com"],
}
FEATURES = {
    "project": "knowledge-base",
    "active": "",
    "features": [
        {"id": "import-pdf", "status": "done", "goal": "импорт PDF"},
        {"id": "full-text-search", "status": "todo", "goal": "поиск по тексту"},
        {"id": "cite-answers", "status": "todo", "goal": "ссылки на источники"},
    ],
}


# -------------------------------------------------------------- path_matches
def test_exact_path_matches():
    assert path_matches("app.py", "app.py") is True


def test_star_does_not_cross_a_slash():
    """Голый fnmatch про "/" не знает и разрешил бы правку в подпапке."""
    assert path_matches("app/x.py", "app/*.py") is True
    assert path_matches("app/sub/x.py", "app/*.py") is False


def test_double_star_spans_directories():
    assert path_matches("app/a/b/x.py", "app/**/*.py") is True


def test_double_star_matches_zero_directories():
    assert path_matches("app/x.py", "app/**/*.py") is True


def test_bare_markdown_pattern_does_not_reach_into_folders():
    assert path_matches("README.md", "*.md") is True
    assert path_matches("docs/guide.md", "*.md") is False
    assert path_matches("docs/guide.md", "**/*.md") is True


def test_unrelated_path_does_not_match():
    assert path_matches("driver.c", "tests/**/*.py") is False


# ------------------------------------------------------------ classify_write
def test_allowed_file_is_in_scope():
    assert classify_write("tests/test_signup.py", CONTRACT) == "allowed"


def test_unlisted_file_is_off_scope():
    assert classify_write("driver.c", CONTRACT) == "off_scope"


def test_forbidden_beats_a_wide_allow_glob():
    """Широкий allowed не должен тихо разрешать то, что контракт запретил явно."""
    wide = {**CONTRACT, "allowed_files": ["**/*.sql", "**/*.py"]}
    assert classify_write("migrations/001_init.sql", wide) == "forbidden"


def test_documentation_is_soft():
    assert classify_write("docs/api.md", CONTRACT) == "soft"


def test_contract_without_soft_files_treats_docs_as_off_scope():
    bare = {k: v for k, v in CONTRACT.items() if k != "soft_files"}
    assert classify_write("docs/api.md", bare) == "off_scope"


# ------------------------------------------------------------- contract_gaps
def test_complete_contract_has_no_gaps():
    assert contract_gaps(CONTRACT) == []


def test_empty_forbidden_files_is_a_gap():
    """Негативное пространство — половина контракта."""
    assert contract_gaps({**CONTRACT, "forbidden_files": []}) == ["forbidden_files"]


def test_absent_rollback_plan_is_a_gap():
    without = {k: v for k, v in CONTRACT.items() if k != "rollback_plan"}
    assert contract_gaps(without) == ["rollback_plan"]


def test_gaps_come_back_sorted():
    broken = {**CONTRACT, "task_id": "", "goal": "", "acceptance_criteria": []}
    assert contract_gaps(broken) == ["acceptance_criteria", "goal", "task_id"]


# -------------------------------------------------------------- merge_egress
def test_both_sides_unenforced_stays_unenforced():
    assert merge_egress(None, None) is None


def test_unenforced_side_defers_to_the_enforcing_one():
    """«Я не проверяю» не имеет права ослаблять того, кто проверяет."""
    assert merge_egress(None, ["api.anthropic.com"]) == ["api.anthropic.com"]
    assert merge_egress(["api.anthropic.com"], None) == ["api.anthropic.com"]


def test_deny_all_stays_deny_all():
    assert merge_egress([], ["api.anthropic.com"]) == []


def test_two_allowlists_intersect():
    assert merge_egress(["a", "b"], ["b", "c"]) == ["b"]


def test_intersection_result_is_sorted():
    assert merge_egress(["z", "m", "a"], ["a", "m", "z"]) == ["a", "m", "z"]


# ----------------------------------------------------------- merge_contracts
def test_allowed_files_are_intersected():
    """Объединять allowed — самая дорогая ошибка урока: права из ниоткуда."""
    parent = {**CONTRACT, "allowed_files": ["app.py", "lib/**"]}
    child = {**CONTRACT, "allowed_files": ["app.py", "docs/**"]}
    assert merge_contracts(parent, child)["allowed_files"] == ["app.py"]


def test_forbidden_files_are_unioned():
    parent = {**CONTRACT, "forbidden_files": ["scripts/**"]}
    child = {**CONTRACT, "forbidden_files": ["migrations/**"]}
    assert merge_contracts(parent, child)["forbidden_files"] == [
        "migrations/**",
        "scripts/**",
    ]


def test_narrowest_time_budget_wins():
    parent = {**CONTRACT, "time_budget_minutes": 60}
    child = {**CONTRACT, "time_budget_minutes": 30}
    assert merge_contracts(parent, child)["time_budget_minutes"] == 30
    child_open = {**CONTRACT, "time_budget_minutes": None}
    assert merge_contracts(parent, child_open)["time_budget_minutes"] == 60


def test_approvals_accumulate_without_duplicates():
    parent = {**CONTRACT, "approvals_required": ["новая зависимость", "миграция"]}
    child = {**CONTRACT, "approvals_required": ["миграция", "смена схемы"]}
    assert merge_contracts(parent, child)["approvals_required"] == [
        "новая зависимость",
        "миграция",
        "смена схемы",
    ]


def test_egress_and_violation_budget_take_the_stricter_side():
    parent = {**CONTRACT, "network_egress": None, "violation_budget": 3}
    child = {**CONTRACT, "network_egress": ["api.anthropic.com"], "violation_budget": 0}
    merged = merge_contracts(parent, child)
    assert merged["network_egress"] == ["api.anthropic.com"]
    assert merged["violation_budget"] == 0


# --------------------------------------------------------------- scope_check
def test_clean_run_passes():
    report = scope_check(CONTRACT, CLEAN_RUN)
    assert report["passed"] is True
    assert report["in_scope"] == ["app.py", "tests/test_signup.py"]
    assert report["findings"] == []


def test_off_scope_write_is_a_warning_that_eats_the_budget():
    run = {**CLEAN_RUN, "touched_files": CLEAN_RUN["touched_files"] + ["email_helper.py"]}
    report = scope_check(CONTRACT, run)
    assert report["off_scope"] == ["email_helper.py"]
    assert (report["warnings"], report["over_budget"], report["passed"]) == (1, True, False)
    assert all(f["severity"] in SEVERITIES for f in report["findings"])


def test_violation_budget_tolerates_one_slip():
    """Гейт, блокирующий за правку README, отключит первая же раздражённая команда."""
    contract = {**CONTRACT, "violation_budget": 1}
    run = {**CLEAN_RUN, "touched_files": CLEAN_RUN["touched_files"] + ["email_helper.py"]}
    report = scope_check(contract, run)
    assert (report["warnings"], report["over_budget"], report["passed"]) == (1, False, True)


def test_forbidden_write_blocks_whatever_the_budget():
    contract = {**CONTRACT, "violation_budget": 99}
    run = {**CLEAN_RUN, "touched_files": ["app.py", "scripts/release.sh"]}
    report = scope_check(contract, run)
    assert report["forbidden"] == ["scripts/release.sh"]
    assert report["passed"] is False


def test_unproven_work_blocks():
    """Критерии приёмки, которые никто не запускал, — это «не сделано»."""
    report = scope_check(CONTRACT, {**CLEAN_RUN, "commands_run": []})
    assert report["missing_acceptance"] == ["pytest -q tests/test_signup.py"]
    assert report["passed"] is False


def test_time_and_network_are_scope_dimensions_too():
    slow = scope_check(CONTRACT, {**CLEAN_RUN, "elapsed_minutes": 42.0})
    leaky = scope_check(CONTRACT, {**CLEAN_RUN, "network_hosts": ["evil.example"]})
    assert slow["passed"] is False
    assert leaky["passed"] is False
    assert {f["code"] for f in leaky["findings"]} == {"network.unallowed_host"}


# --------------------------------------------------------------- pick_feature
def test_active_feature_is_respected():
    assert pick_feature({**FEATURES, "active": "cite-answers"}) == "cite-answers"


def test_empty_active_takes_the_first_todo_in_file_order():
    """Порядок в файле — это приоритет, сортировать по id нельзя."""
    assert pick_feature(FEATURES) == "full-text-search"


def test_no_todo_left_gives_none():
    closed = {
        **FEATURES,
        "features": [{"id": "import-pdf", "status": "done", "goal": "импорт PDF"}],
    }
    assert pick_feature(closed) is None


def test_two_features_in_progress_is_value_error():
    """Две начатые фичи означают, что прошлая сессия закончилась не там, где думают."""
    doubled = {
        **FEATURES,
        "features": [
            {"id": "a", "status": "in_progress", "goal": "a"},
            {"id": "b", "status": "in_progress", "goal": "b"},
        ],
    }
    with pytest.raises(ValueError):
        pick_feature(doubled)
    with pytest.raises(ValueError):
        pick_feature({**FEATURES, "active": "нет-такой-фичи"})
