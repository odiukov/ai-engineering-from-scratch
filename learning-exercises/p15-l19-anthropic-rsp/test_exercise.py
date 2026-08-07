"""Тесты к уроку «Anthropic Responsible Scaling Policy v3.0». Правь exercise.py."""

import pytest

from exercise import (
    ASL_LEVELS,
    RSP_V2,
    RSP_V3,
    SAFEGUARD_SCHEDULE,
    affirmative_case_sections,
    capability_level,
    deployment_decision,
    diff_policies,
    missing_safeguards,
    policy_score,
    required_safeguards,
    unilateral_commitments,
)

# Claude Opus 4.6 по заявлению из анонса v3.0: порог не пересечён.
OPUS_4_6 = {
    "rd_automation_share": 0.30,
    "metr_horizon_hours": 14.0,
    "cyber_uplift": 0.10,
}
NEAR = {
    "rd_automation_share": 0.40,
    "metr_horizon_hours": 22.0,
    "cyber_uplift": 0.10,
}
OVER = {
    "rd_automation_share": 0.70,
    "metr_horizon_hours": 48.0,
    "cyber_uplift": 0.60,
}

ASL2_KIT = ["model_card", "usage_policy"]


# --------------------------------------------------------- capability_level
def test_level_of_a_model_below_every_threshold_is_the_base_level():
    assert capability_level(OPUS_4_6) == "ASL-2"


def test_two_triggers_raise_the_level():
    assert capability_level(NEAR) == "ASL-3"


def test_a_single_trigger_is_not_enough():
    """Одно измерение — шум. Лестница поднимается на двух независимых."""
    one = {"rd_automation_share": 0.99, "metr_horizon_hours": 1.0,
           "cyber_uplift": 0.0}
    assert capability_level(one) == "ASL-2"


def test_the_highest_reached_level_wins():
    assert capability_level(OVER) == "ASL-4"


def test_raising_a_measurement_never_lowers_the_level():
    """Монотонность — то, ради чего лестница вообще существует."""
    order = {lvl: i for i, lvl in enumerate(ASL_LEVELS)}
    previous = -1
    reached = set()
    # измерения растут покоординатно — только на таком ряду монотонность
    # вообще что-то утверждает
    for share, horizon in ((0.0, 1.0), (0.20, 10.0), (0.36, 22.0),
                           (0.50, 30.0), (0.65, 45.0), (0.90, 60.0)):
        m = {"rd_automation_share": share, "metr_horizon_hours": horizon,
             "cyber_uplift": 0.0}
        current = order[capability_level(m)]
        reached.add(current)
        assert current >= previous
        previous = current
    # ряд обязан реально пройти лестницу, иначе монотонность выполнилась бы
    # тривиально на одном уровне
    assert reached == {0, 1, 2}


def test_a_missing_measurement_counts_as_zero():
    assert capability_level({}) == "ASL-2"


# ------------------------------------------------------ required_safeguards
def test_base_level_requires_the_base_kit():
    assert required_safeguards("ASL-2") == list(SAFEGUARD_SCHEDULE["ASL-2"])


def test_the_ladder_is_cumulative():
    """Модель на ASL-3 не перестаёт нуждаться в model card."""
    lower = required_safeguards("ASL-2")
    higher = required_safeguards("ASL-3")
    assert higher[: len(lower)] == lower
    assert set(SAFEGUARD_SCHEDULE["ASL-3"]) <= set(higher)


def test_each_step_adds_requirements_and_removes_none():
    seen = set()
    for level in ASL_LEVELS:
        current = set(required_safeguards(level))
        assert seen <= current
        seen = current


def test_an_unknown_level_is_an_error_not_an_empty_kit():
    """Пустой список означал бы «мер не требуется» — противоположный смысл."""
    with pytest.raises(ValueError):
        required_safeguards("ASL-5")


# ------------------------------------------------------- missing_safeguards
def test_nothing_is_missing_when_the_kit_is_complete():
    assert missing_safeguards("ASL-2", ASL2_KIT) == []


def test_the_same_kit_is_incomplete_one_level_up():
    gaps = missing_safeguards("ASL-3", ASL2_KIT)
    assert gaps == list(SAFEGUARD_SCHEDULE["ASL-3"])


def test_extra_safeguards_are_not_reported():
    assert missing_safeguards("ASL-2", ASL2_KIT + ["external_review"]) == []


def test_missing_list_keeps_the_ladder_order():
    gaps = missing_safeguards("ASL-4", [])
    expected = required_safeguards("ASL-4")
    assert gaps == expected


# ------------------------------------------------------ deployment_decision
def test_deployment_is_allowed_when_the_level_kit_is_in_place():
    d = deployment_decision(OPUS_4_6, ASL2_KIT)
    assert d["level"] == "ASL-2"
    assert d["allowed"] is True
    assert d["missing"] == []


def test_higher_capability_blocks_the_same_safeguard_kit():
    """Рост возможностей поднимает планку; развёртывание без мер невозможно."""
    low = deployment_decision(OPUS_4_6, ASL2_KIT)
    high = deployment_decision(NEAR, ASL2_KIT)
    assert low["allowed"] is True
    assert high["allowed"] is False
    assert "weights_security" in high["missing"]


def test_adding_the_missing_safeguards_reopens_deployment():
    kit = ASL2_KIT + list(SAFEGUARD_SCHEDULE["ASL-3"])
    d = deployment_decision(NEAR, kit)
    assert d["level"] == "ASL-3"
    assert d["allowed"] is True


def test_the_reason_names_the_level_and_the_gaps():
    """Решение, которое нельзя прочитать и оспорить, — не решение."""
    d = deployment_decision(OVER, ASL2_KIT)
    assert d["level"] in d["reason"]
    for gap in d["missing"]:
        assert gap in d["reason"]


def test_top_level_deployment_needs_the_whole_ladder():
    d = deployment_decision(OVER, ASL2_KIT)
    assert d["allowed"] is False
    assert "rand_sl4_security" in d["missing"]


# --------------------------------------------------- unilateral_commitments
def test_only_the_unilateral_column_counts_as_a_commitment():
    assert unilateral_commitments({"a": "unilateral", "b": "industry"}) == ["a"]


def test_rand_sl4_is_a_commitment_in_v2_but_a_recommendation_in_v3():
    """Мера в колонке industry — не обещание лаборатории, а её пожелание."""
    assert "rand_sl4_security" in unilateral_commitments(RSP_V2)
    assert "rand_sl4_security" not in unilateral_commitments(RSP_V3)


def test_an_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        unilateral_commitments({"a": "optional"})


# ------------------------------------------------------------ diff_policies
def test_diff_reports_the_removed_pause_commitment():
    d = diff_policies(RSP_V2, RSP_V3)
    assert "pause_on_threshold" in d["removed"]


def test_diff_reports_what_v3_added():
    d = diff_policies(RSP_V2, RSP_V3)
    assert "frontier_safety_roadmap" in d["added"]
    assert "risk_report" in d["added"]


def test_a_retiered_commitment_is_not_reported_as_removed():
    """Перенос в колонку industry не должен маскироваться под «без изменений»."""
    d = diff_policies(RSP_V2, RSP_V3)
    assert ("rand_sl4_security", "unilateral", "industry") in d["retiered"]
    assert "rand_sl4_security" not in d["removed"]
    assert "rand_sl4_security" not in d["added"]


def test_diff_of_a_policy_with_itself_is_empty():
    d = diff_policies(RSP_V3, RSP_V3)
    assert d == {"added": [], "removed": [], "retiered": []}


# ------------------------------------------------ affirmative_case_sections
def test_no_affirmative_case_below_the_threshold_level():
    assert affirmative_case_sections("ASL-2") == []
    assert affirmative_case_sections("ASL-3") == []


def test_the_threshold_level_requires_the_full_section_list():
    sections = affirmative_case_sections("ASL-4")
    assert "misalignment_risk_analysis" in sections
    assert "residual_risk" in sections
    assert len(sections) == 6


def test_high_gaming_rate_adds_the_adjusted_estimate_section():
    plain = affirmative_case_sections("ASL-4", 0.10)
    gamed = affirmative_case_sections("ASL-4", 0.28)
    assert len(gamed) == len(plain) + 1
    assert gamed[-1] == "gaming_adjusted_capability_estimate"


def test_a_negative_gaming_rate_is_an_error():
    with pytest.raises(ValueError):
        affirmative_case_sections("ASL-4", -0.1)


# -------------------------------------------------------------- policy_score
def test_v2_reproduces_the_published_score():
    r = policy_score({"quantitative_thresholds", "pause_commitment",
                      "declared_cadence", "published_risk_reports"})
    assert r["score"] == pytest.approx(2.2)


def test_v3_reproduces_the_published_score():
    r = policy_score({"declared_cadence", "published_risk_reports",
                      "frontier_safety_roadmap"})
    assert r["score"] == pytest.approx(1.9)


def test_the_downgrade_crosses_a_band_boundary():
    """2.2 -> 1.9 не косметика: политика меняет категорию на «weak»."""
    v2 = policy_score({"quantitative_thresholds", "pause_commitment",
                       "declared_cadence", "published_risk_reports"})
    v3 = policy_score({"declared_cadence", "published_risk_reports",
                       "frontier_safety_roadmap"})
    assert v2["band"] == "moderate"
    assert v3["band"] == "weak"


def test_restoring_the_pause_commitment_lifts_the_score():
    without = policy_score({"declared_cadence", "published_risk_reports",
                            "frontier_safety_roadmap"})
    with_pause = policy_score({"declared_cadence", "published_risk_reports",
                               "frontier_safety_roadmap", "pause_commitment"})
    assert with_pause["score"] - without["score"] == pytest.approx(0.3)
    assert with_pause["band"] == "moderate"


def test_an_unknown_rubric_criterion_is_an_error():
    with pytest.raises(ValueError):
        policy_score({"vibes"})
