"""Тесты к уроку «Скрипты инициализации агента». Правь exercise.py."""

import json

import pytest

from exercise import (
    LOCK_PATH,
    PROBE_ORDER,
    REPORT_PATH,
    STATE_PATH,
    deps_fingerprint,
    lock_is_fresh,
    probe_dependencies,
    probe_env,
    probe_lkg_diff,
    probe_runtime,
    probe_state_freshness,
    run_init,
)

CONFIG = {
    "python": (3, 10),
    "deps": ["pytest", "ruff"],
    "env": ["AGENT_WORKDIR"],
    "test_command": "pytest -q",
}
HEALTHY = {
    "version": (3, 12),
    "installed": ["pytest", "ruff", "mypy"],
    "environ": {"AGENT_WORKDIR": "/work"},
    "changed_files": ["agent.py"],
}
NOW = 10_000


# -------------------------------------------------------------- probe_runtime
def test_matching_runtime_passes():
    assert probe_runtime((3, 12))["status"] == "pass"


def test_old_runtime_fails():
    """Строковое сравнение сказало бы, что "3.9" >= "3.10". Кортежи не врут."""
    assert probe_runtime((3, 9), required=(3, 10))["status"] == "fail"


def test_patch_version_does_not_break_the_comparison():
    assert probe_runtime((3, 10, 7), required=(3, 10))["status"] == "pass"


# --------------------------------------------------------- probe_dependencies
def test_all_dependencies_present_passes():
    assert probe_dependencies(["pytest", "ruff"])["status"] == "pass"


def test_missing_dependency_is_named_in_the_detail():
    """Отчёт без имён заставляет человека запускать пробу заново руками."""
    probe = probe_dependencies(["pytest"], required=("pytest", "ruff"))
    assert probe["status"] == "fail"
    assert "ruff" in probe["detail"]


def test_extra_packages_are_not_a_problem():
    assert probe_dependencies(["pytest", "ruff", "numpy"])["status"] == "pass"


# ------------------------------------------------------------------ probe_env
def test_present_env_vars_pass():
    assert probe_env({"AGENT_WORKDIR": "/work"})["status"] == "pass"


def test_empty_value_counts_as_missing():
    """Объявленная пустой переменная хуже отсутствующей: падение уедет в середину сессии."""
    assert probe_env({"AGENT_WORKDIR": "   "})["status"] == "fail"
    assert probe_env({})["status"] == "fail"


def test_detail_never_leaks_the_value():
    """init_report.json уходит в логи CI, а в переменных лежат ключи."""
    probe = probe_env({"API_KEY": "sk-secret-42"}, required=("API_KEY",))
    assert probe["status"] == "pass"
    assert "sk-secret-42" not in probe["detail"]


# ----------------------------------------------------- probe_state_freshness
def test_absent_state_is_a_warning_not_a_failure():
    assert probe_state_freshness(None, now=100)["status"] == "warn"


def test_fresh_state_passes():
    assert probe_state_freshness({"written_at": 90}, now=100)["status"] == "pass"


def test_stale_state_warns():
    assert probe_state_freshness({"written_at": 0}, now=10**6)["status"] == "warn"


def test_state_written_in_the_future_is_value_error():
    with pytest.raises(ValueError):
        probe_state_freshness({"written_at": 500}, now=100)


# ---------------------------------------------------------- probe_lkg_diff
def test_small_diff_passes():
    assert probe_lkg_diff(["a.py", "b.py"])["status"] == "pass"


def test_diff_over_the_budget_fails():
    assert probe_lkg_diff([f"f{i}.py" for i in range(11)], budget=10)["status"] == "fail"


def test_exactly_the_budget_still_passes():
    assert probe_lkg_diff([f"f{i}.py" for i in range(10)], budget=10)["status"] == "pass"


def test_missing_baseline_is_a_warning():
    """Базовая линия не закреплена — это повод её закрепить, а не отказ стартовать."""
    assert probe_lkg_diff(None)["status"] == "warn"


def test_negative_budget_is_value_error():
    with pytest.raises(ValueError):
        probe_lkg_diff([], budget=-1)


# ------------------------------------------------------------ deps_fingerprint
def test_reordered_lists_give_the_same_fingerprint():
    """Порядок строк в манифесте не меняет окружение — не должен менять и отпечаток."""
    shuffled = {**CONFIG, "deps": ["ruff", "pytest"]}
    assert deps_fingerprint(shuffled) == deps_fingerprint(CONFIG)


def test_added_dependency_changes_the_fingerprint():
    grown = {**CONFIG, "deps": CONFIG["deps"] + ["httpx"]}
    assert deps_fingerprint(grown) != deps_fingerprint(CONFIG)


def test_incomplete_config_is_value_error():
    with pytest.raises(ValueError):
        deps_fingerprint({"python": (3, 10), "deps": [], "env": []})


# --------------------------------------------------------------- lock_is_fresh
def test_absent_lock_is_not_trusted():
    assert lock_is_fresh(None, CONFIG, now=0) is False


def test_fresh_lock_is_trusted():
    lock = {"fingerprint": deps_fingerprint(CONFIG), "written_at": 0}
    assert lock_is_fresh(lock, CONFIG, now=10) is True


def test_expired_lock_is_not_trusted():
    lock = {"fingerprint": deps_fingerprint(CONFIG), "written_at": 0}
    assert lock_is_fresh(lock, CONFIG, now=10, ttl=10) is False


def test_changed_manifest_invalidates_the_lock():
    """Добавили зависимость — прошлый прогон проб больше ничего не доказывает."""
    lock = {"fingerprint": deps_fingerprint(CONFIG), "written_at": 0}
    grown = {**CONFIG, "deps": CONFIG["deps"] + ["httpx"]}
    assert lock_is_fresh(lock, grown, now=10) is False


def test_lock_from_the_future_is_not_trusted():
    lock = {"fingerprint": deps_fingerprint(CONFIG), "written_at": 5_000}
    assert lock_is_fresh(lock, CONFIG, now=100) is False


# -------------------------------------------------------------------- run_init
def test_healthy_init_writes_report_and_lock():
    fs = {}
    result = run_init(fs, HEALTHY, CONFIG, now=NOW)
    assert result["started"] is True and result["skipped"] is False
    assert REPORT_PATH in result["fs"] and LOCK_PATH in result["fs"]
    assert fs == {}, "входную файловую систему трогать нельзя"


def test_second_run_changes_nothing_but_the_timestamp():
    """Идемпотентность — то, что позволяет повесить скрипт на хук и в CI."""
    first = run_init({}, HEALTHY, CONFIG, now=NOW, use_cache=False)
    same_clock = run_init(first["fs"], HEALTHY, CONFIG, now=NOW, use_cache=False)
    later = run_init(first["fs"], HEALTHY, CONFIG, now=NOW + 999, use_cache=False)

    assert same_clock["fs"] == first["fs"]
    assert later["report"]["probes"] == first["report"]["probes"]
    assert later["report"]["timestamp"] != first["report"]["timestamp"]
    assert json.loads(later["fs"][LOCK_PATH])["fingerprint"] == (
        json.loads(first["fs"][LOCK_PATH])["fingerprint"]
    )


def test_failed_probe_refuses_to_start_and_writes_no_lock():
    """Падать надо громко: агент не должен стартовать на битом рабочем месте."""
    broken = {**HEALTHY, "installed": ["pytest"]}
    result = run_init({}, broken, CONFIG, now=NOW)
    assert result["started"] is False
    assert result["report"]["blocking"] == ["dependencies"]
    assert LOCK_PATH not in result["fs"]
    assert REPORT_PATH in result["fs"], "смотреть человеку всё равно куда-то нужно"


def test_fresh_lock_short_circuits_the_probes():
    fs = {
        LOCK_PATH: json.dumps(
            {"fingerprint": deps_fingerprint(CONFIG), "written_at": NOW - 10}
        )
    }
    result = run_init(fs, HEALTHY, CONFIG, now=NOW)
    assert (result["skipped"], result["report"]) == (True, None)
    assert result["fs"] == fs


def test_stale_state_only_warns_and_still_starts():
    fs = {STATE_PATH: json.dumps({"written_at": 0})}
    result = run_init(fs, HEALTHY, CONFIG, now=NOW * 100)
    statuses = {p["name"]: p["status"] for p in result["report"]["probes"]}
    assert statuses["state_freshness"] == "warn"
    assert result["started"] is True


def test_probes_come_back_in_a_fixed_order():
    """Отчёт диффают между запусками — перетасованные строки диффать нельзя."""
    result = run_init({}, HEALTHY, CONFIG, now=NOW)
    assert tuple(p["name"] for p in result["report"]["probes"]) == PROBE_ORDER
