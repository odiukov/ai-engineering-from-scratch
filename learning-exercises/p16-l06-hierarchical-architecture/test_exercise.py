"""Тесты к уроку «Иерархическая архитектура и её отказ». Правь exercise.py."""

import pytest

from exercise import (
    DEPTH_CEILING,
    OrgError,
    aggregate,
    build_org,
    delegate,
    depth,
    leaves,
    provenance,
    too_deep,
    validate_org,
)

# Деревья записаны литералами, а не собраны через build_org: модуль,
# зовущий непройденную функцию на импорте, падал бы на коллекции целиком.
ORG = {
    "vp-eng": ["eng-manager", "legal-manager", "finance-manager"],
    "eng-manager": ["fe", "be"],
    "legal-manager": ["lawyer"],
    "finance-manager": ["finance"],
    "fe": [],
    "be": [],
    "lawyer": [],
    "finance": [],
}
FLAT = {"vp-eng": ["fe", "be"], "fe": [], "be": []}
DEEP = {"top": ["sub"], "sub": ["subsub"], "subsub": ["w"], "w": []}
CYCLE = {"a": ["b"], "b": ["a"]}

ANSWERS = {
    "fe": "React component audited; 2 issues.",
    "be": "API endpoints audited; 1 issue.",
    "lawyer": "Contract clauses A and B are non-compliant.",
    "finance": "Projected cost $42k/month.",
}


def counting_summary(node, parts):
    return f"{node}:{len(parts)}"


# ---------------------------------------------------------------- build_org
def test_build_org_collects_reports_in_order():
    assert build_org([("top", "a"), ("top", "b")])["top"] == ["a", "b"]


def test_build_org_registers_leaves_as_childless_nodes():
    assert build_org([("top", "a")])["a"] == []


def test_build_org_of_nothing_is_empty():
    assert build_org([]) == {}


def test_build_org_does_not_judge_the_structure():
    """Сборка честно соберёт и цикл — проверяет его validate_org."""
    assert build_org([("a", "b"), ("b", "a")]) == {"a": ["b"], "b": ["a"]}


# ------------------------------------------------------------- validate_org
def test_validate_returns_nodes_top_down():
    assert validate_org(FLAT, "vp-eng") == ["vp-eng", "fe", "be"]


def test_validate_catches_a_cycle_in_the_chain_of_command():
    """Цикл подчинения — это reconciliation loop из урока в чистом виде."""
    with pytest.raises(OrgError):
        validate_org(CYCLE, "a")


def test_validate_catches_a_self_managing_node():
    with pytest.raises(OrgError):
        validate_org({"a": ["a"]}, "a")


def test_validate_catches_two_managers_over_one_report():
    with pytest.raises(OrgError):
        validate_org({"t": ["a", "b"], "a": ["w"], "b": ["w"], "w": []}, "t")


def test_validate_catches_a_node_nobody_can_reach():
    with pytest.raises(OrgError):
        validate_org({"t": ["a"], "a": [], "orphan": []}, "t")


def test_validate_refuses_a_root_outside_the_org():
    with pytest.raises(OrgError):
        validate_org(FLAT, "ceo")


def test_validate_accepts_the_three_level_org():
    assert validate_org(ORG, "vp-eng")[0] == "vp-eng"


# -------------------------------------------------------------------- depth
def test_a_lone_node_has_depth_zero():
    assert depth({"top": []}, "top") == 0


def test_manager_over_workers_is_one_level():
    """Ловушка: считаются рёбра, а не узлы."""
    assert depth(FLAT, "vp-eng") == 1


def test_the_lesson_org_is_two_levels():
    assert depth(ORG, "vp-eng") == 2


def test_depth_takes_the_longest_branch():
    org = {"t": ["short", "long"], "short": [], "long": ["w"], "w": []}
    assert depth(org, "t") == 2


# ------------------------------------------------------------------- leaves
def test_only_leaves_are_reported_as_workers():
    assert leaves(ORG, "vp-eng") == ["fe", "be", "lawyer", "finance"]


def test_leaves_are_listed_left_to_right():
    assert leaves(FLAT, "vp-eng") == ["fe", "be"]


def test_a_lone_node_is_its_own_leaf():
    assert leaves({"top": []}, "top") == ["top"]


def test_a_sub_manager_never_counts_as_a_worker():
    """Внутренние узлы только планируют и сводят, работу делают листья."""
    assert "eng-manager" not in leaves(ORG, "vp-eng")


# --------------------------------------------------------------- provenance
def test_provenance_traces_a_leaf_back_to_the_top():
    assert provenance(ORG, "vp-eng", "fe") == ["vp-eng", "eng-manager", "fe"]


def test_provenance_of_the_root_is_the_root():
    assert provenance(ORG, "vp-eng", "vp-eng") == ["vp-eng"]


def test_provenance_of_an_unrelated_node_is_refused():
    with pytest.raises(OrgError):
        provenance(ORG, "vp-eng", "ghost")


def test_provenance_length_matches_the_level():
    assert len(provenance(DEEP, "top", "w")) == depth(DEEP, "top") + 1


# ----------------------------------------------------------------- too_deep
def test_a_two_level_org_is_within_the_ceiling():
    assert too_deep(ORG, "vp-eng") == []


def test_the_third_level_is_flagged():
    assert too_deep(DEEP, "top") == ["w"]


def test_the_default_ceiling_is_the_lesson_value():
    assert DEPTH_CEILING == 2
    # Вызов без ceiling обязан вести себя ровно как вызов с DEPTH_CEILING,
    # а не как захардкоженное внутри число.
    assert too_deep(DEEP, "top") == too_deep(DEEP, "top", ceiling=DEPTH_CEILING)
    assert too_deep(DEEP, "top") != too_deep(DEEP, "top", ceiling=DEPTH_CEILING + 1)


def test_a_stricter_ceiling_flags_more_nodes():
    assert too_deep(ORG, "vp-eng", ceiling=1) == ["fe", "be", "lawyer", "finance"]


# ----------------------------------------------------------------- delegate
def test_delegation_to_real_branches_is_clean():
    got = delegate(ORG, "vp-eng", ["eng-manager", "legal-manager"],
                   ["eng-manager", "legal-manager"])
    assert got == {"delegated": ["eng-manager", "legal-manager"],
                   "unknown": [], "uncovered": []}


def test_decomposition_drift_shows_up_as_an_uncovered_branch():
    """Подмена legal на finance: работа сделана честно, вопрос без ответа."""
    got = delegate(ORG, "vp-eng", ["eng-manager", "finance-manager"],
                   ["eng-manager", "legal-manager"])
    assert got["unknown"] == [] and got["uncovered"] == ["legal-manager"]


def test_delegating_to_a_branch_that_does_not_exist():
    got = delegate(ORG, "vp-eng", ["marketing"], ["eng-manager"])
    assert got["unknown"] == ["marketing"] and got["delegated"] == []


def test_delegating_to_nobody_leaves_everything_uncovered():
    got = delegate(ORG, "vp-eng", [], ["eng-manager", "legal-manager"])
    assert got["uncovered"] == ["eng-manager", "legal-manager"]


# ---------------------------------------------------------------- aggregate
def test_aggregate_builds_the_answer_bottom_up():
    org = {"t": ["s"], "s": ["w1", "w2"], "w1": [], "w2": []}
    assert aggregate(org, "t", {"w1": "A", "w2": "B"}) == "[t] [s] A | B"


def test_a_leaf_returns_its_own_answer():
    assert aggregate(ORG, "fe", ANSWERS) == ANSWERS["fe"]


def test_every_leaf_answer_reaches_the_top():
    text = aggregate(ORG, "vp-eng", ANSWERS)
    assert all(answer in text for answer in ANSWERS.values())


def test_the_top_manager_sees_summaries_not_raw_worker_output():
    """Локальная сводка: наверх приходит три сводки, а не четыре ответа."""
    assert aggregate(ORG, "vp-eng", ANSWERS, counting_summary) == "vp-eng:3"
    assert len(leaves(ORG, "vp-eng")) == 4


def test_a_silent_worker_is_marked_not_hidden():
    org = {"t": ["w1", "w2"], "w1": [], "w2": []}
    assert aggregate(org, "t", {"w1": "A"}) == "[t] A | [no answer from w2]"


def test_a_custom_summary_reaches_every_level():
    assert aggregate(DEEP, "top", {"w": "x"}, counting_summary) == "top:1"
