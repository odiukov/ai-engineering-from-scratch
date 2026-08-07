"""Тесты к уроку «Сборка полного LLM-пайплайна». Правь exercise.py."""

import pytest

from exercise import (
    chain_violations,
    descendants,
    estimate_cost_usd,
    gate_failures,
    plan,
    rollback_set,
    stable_hash,
    topological_order,
)

# три стадии в линию: a -> b -> c
LINE = {"a": [], "b": ["a"], "c": ["b"]}

# ромб: 07 и 08 параллельны, 09 ждёт обоих
DIAMOND = {"04": [], "07": ["04"], "08": ["04"], "09": ["07", "08"]}


def good_pipeline():
    """Целая цепочка хэшей на трёх стадиях."""
    return {
        "a": {"deps": [], "inputs": {}, "output": "h_a"},
        "b": {"deps": ["a"], "inputs": {"a": "h_a"}, "output": "h_b"},
        "c": {"deps": ["b"], "inputs": {"b": "h_b"}, "output": "h_c"},
    }


def good_manifest():
    """Манифест, который обязан дать SHIP."""
    return {
        "stages": good_pipeline(),
        "gates": {"mmlu": (">=", 0.65), "safety": ("<=", 0.05)},
        "metrics": {"mmlu": 0.71, "safety": 0.02},
        "budget_usd": 1e9,
        "pretrain": {
            "params": 7e9,
            "tokens": 2e12,
            "peak_flops": 989e12,
            "mfu": 0.4,
            "usd_per_gpu_hour": 2.5,
        },
    }


# ------------------------------------------------------------- stable_hash
def test_hash_ignores_key_order():
    """Тот же артефакт, другой порядок ключей — тот же адрес."""
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_hash_changes_when_a_value_changes():
    assert stable_hash({"seed": 42}) != stable_hash({"seed": 43})


def test_hash_is_a_full_sha256_hex_digest():
    digest = stable_hash({"anything": [1, 2, 3]})
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_hash_distinguishes_nested_structures():
    """Вложенность — часть содержимого, а не оформление."""
    assert stable_hash({"a": [1, 2]}) != stable_hash({"a": [2, 1]})


# -------------------------------------------------------- topological_order
def test_dependencies_run_before_dependents():
    assert topological_order(LINE) == ["a", "b", "c"]


def test_independent_stages_are_ordered_alphabetically():
    """Детерминизм: два запуска одного манифеста дают один и тот же лог."""
    assert topological_order({"c": [], "a": [], "b": []}) == ["a", "b", "c"]


def test_parallel_stages_both_precede_their_joint_dependent():
    order = topological_order(DIAMOND)
    assert order.index("07") < order.index("09")
    assert order.index("08") < order.index("09")
    assert order[0] == "04"


def test_cycle_is_rejected():
    with pytest.raises(ValueError):
        topological_order({"a": ["b"], "b": ["a"]})


def test_unknown_dependency_is_rejected():
    """Ссылка в никуда — потерянная зависимость, а не пустое место."""
    with pytest.raises(ValueError):
        topological_order({"a": ["ghost"]})


def test_empty_pipeline_has_empty_order():
    assert topological_order({}) == []


# --------------------------------------------------- descendants / rollback
def test_descendants_are_transitive():
    assert descendants(LINE, "a") == ["b", "c"]


def test_a_leaf_has_no_descendants():
    assert descendants(LINE, "c") == []


def test_rollback_includes_the_failed_stage_itself():
    assert rollback_set(LINE, "b") == ["b", "c"]


def test_rollback_of_the_last_stage_is_cheap():
    """Падение квантизации не заставляет переучивать модель."""
    assert rollback_set(LINE, "c") == ["c"]


def test_rollback_of_a_shared_ancestor_hits_both_branches():
    assert rollback_set(DIAMOND, "04") == ["04", "07", "08", "09"]


# -------------------------------------------------------- chain_violations
def test_intact_hash_chain_has_no_violations():
    assert chain_violations(good_pipeline()) == []


def test_stale_input_hash_is_reported():
    stages = good_pipeline()
    stages["b"]["inputs"]["a"] = "h_OLD"
    assert chain_violations(stages) == [("b", "a")]


def test_missing_input_record_is_a_violation_not_a_pass():
    """«Не записали ожидаемый хэш» — это разрыв цепочки, а не мелочь."""
    stages = good_pipeline()
    stages["c"]["inputs"] = {}
    assert chain_violations(stages) == [("c", "b")]


def test_a_changed_output_invalidates_only_its_consumer():
    stages = good_pipeline()
    stages["b"]["output"] = "h_b_NEW"
    assert chain_violations(stages) == [("c", "b")]


# ------------------------------------------------------------ gate_failures
def test_all_gates_pass():
    gates = {"mmlu": (">=", 0.65), "cost_usd": ("<=", 50000)}
    assert gate_failures(gates, {"mmlu": 0.70, "cost_usd": 40000}) == []


def test_regression_below_threshold_fails():
    gates = {"mmlu": (">=", 0.65)}
    assert gate_failures(gates, {"mmlu": 0.60}) == ["mmlu"]


def test_threshold_is_inclusive():
    """«Не хуже базы» значит «>=», ровное попадание в порог проходит."""
    assert gate_failures({"mmlu": (">=", 0.65)}, {"mmlu": 0.65}) == []


def test_unmeasured_metric_fails_the_gate():
    assert gate_failures({"kl": ("<=", 25.0)}, {}) == ["kl"]


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError):
        gate_failures({"mmlu": ("~=", 0.65)}, {"mmlu": 0.65})


# -------------------------------------------------------- estimate_cost_usd
def test_cost_matches_the_six_p_t_formula():
    assert estimate_cost_usd(1e9, 1e9, 1e12, 0.5, 2.0) == pytest.approx(6666.667, rel=1e-6)


def test_cost_of_a_seven_b_on_two_t_tokens():
    """Цифры упражнения 3 из урока: 7B, 2T токенов, H100, MFU 40%."""
    got = estimate_cost_usd(7e9, 2e12, 989e12, 0.4, 2.5)
    assert got == pytest.approx(147455.0, rel=1e-4)


def test_cost_is_linear_in_tokens():
    a = estimate_cost_usd(1e9, 1e9, 1e12, 0.5, 2.0)
    b = estimate_cost_usd(1e9, 2e9, 1e12, 0.5, 2.0)
    assert b == pytest.approx(2 * a)


def test_better_mfu_makes_the_run_cheaper():
    slow = estimate_cost_usd(1e9, 1e9, 1e12, 0.2, 2.0)
    fast = estimate_cost_usd(1e9, 1e9, 1e12, 0.5, 2.0)
    assert fast < slow


def test_impossible_mfu_is_rejected():
    with pytest.raises(ValueError):
        estimate_cost_usd(1e9, 1e9, 1e12, 1.5, 2.0)


# --------------------------------------------------------------------- plan
def test_clean_manifest_ships():
    result = plan(good_manifest())
    assert result["decision"] == "SHIP"
    assert result["order"] == ["a", "b", "c"]
    assert result["violations"] == []
    assert result["failed_gates"] == []


def test_broken_hash_chain_holds_the_run():
    manifest = good_manifest()
    manifest["stages"]["b"]["inputs"]["a"] = "h_OLD"
    result = plan(manifest)
    assert result["decision"] == "HOLD"
    assert result["violations"] == [("b", "a")]


def test_failing_eval_gate_holds_the_run():
    manifest = good_manifest()
    manifest["metrics"]["mmlu"] = 0.10
    result = plan(manifest)
    assert result["decision"] == "HOLD"
    assert result["failed_gates"] == ["mmlu"]


def test_budget_overrun_holds_the_run_before_it_starts():
    """Гейт бюджета срабатывает на `plan`, до единого потраченного часа."""
    manifest = good_manifest()
    manifest["budget_usd"] = 1000.0
    result = plan(manifest)
    assert result["decision"] == "HOLD"
    assert result["cost_usd"] > manifest["budget_usd"]


def test_plan_reports_the_cost_it_used_for_the_decision():
    manifest = good_manifest()
    expected = estimate_cost_usd(**manifest["pretrain"])
    assert plan(manifest)["cost_usd"] == pytest.approx(expected)
