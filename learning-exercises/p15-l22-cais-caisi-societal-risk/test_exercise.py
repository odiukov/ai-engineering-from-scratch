"""Тесты к уроку «CAIS, CAISI и риск общественного масштаба».
Правь exercise.py."""

import pytest

from exercise import (
    MITIGATIONS,
    SB53_FRONTIER_COMPUTE_OPS,
    SB53_GENERAL_REPORT_HOURS,
    SB53_IMMINENT_REPORT_HOURS,
    SB53_LARGE_REVENUE_USD,
    SOCIETAL_STACK,
    aggregate_risk,
    identify_organization,
    incident_report_status,
    mitigation_checklist,
    sb53_obligations,
    stack_assessment,
    tag_risks,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Внутренний помощник по рефакторингу: закрытый доступ, все меры на месте.
CLEAN = {
    "public_facing": False,
    "harmful_capability_labels": (),
    "competitive_pressure": False,
    "independent_audit": True,
    "multi_layer_defense": True,
    "information_security": True,
    "agent_autonomy_hours": 1.0,
    "training_compute_ops": 0.0,
    "annual_gross_revenue_usd": 0.0,
}
# Фронтирный автономный агент: публичный, с метками, без мер, на 48 часов.
FRONTIER = {
    "public_facing": True,
    "harmful_capability_labels": ("cyber", "cbrn"),
    "competitive_pressure": True,
    "independent_audit": False,
    "multi_layer_defense": False,
    "information_security": False,
    "agent_autonomy_hours": 48.0,
}


# ----------------------------------------------------------------- tag_risks
def test_a_clean_internal_deployment_tags_nothing():
    assert tag_risks(CLEAN) == []


def test_a_frontier_deployment_tags_all_four_categories():
    """Категории не взаимоисключающие: один выкат бывает всеми четырьмя."""
    assert tag_risks(FRONTIER) == [
        "ai_races",
        "malicious_use",
        "organizational_risks",
        "rogue_ais",
    ]


def test_harm_labels_only_count_when_the_deployment_is_public():
    private = dict(CLEAN, harmful_capability_labels=("cyber",))
    public = dict(private, public_facing=True)
    assert tag_risks(private) == []
    assert tag_risks(public) == ["malicious_use"]


def test_one_missing_organizational_lever_is_enough():
    """CAIS называет категорию отдельно именно потому, что хватает одного."""
    for lever in ("independent_audit", "multi_layer_defense", "information_security"):
        assert tag_risks(dict(CLEAN, **{lever: False})) == ["organizational_risks"]


def test_an_unknown_feature_or_harm_label_is_an_error():
    """Опечатка в признаке иначе тихо уйдёт в безопасное умолчание."""
    with pytest.raises(ValueError):
        tag_risks({"publicfacing": True})
    with pytest.raises(ValueError):
        tag_risks({"harmful_capability_labels": ("weaponized_prose",)})


# ------------------------------------------------------- mitigation_checklist
def test_an_untagged_deployment_gets_an_empty_checklist():
    """Чеклист, где всё перечислено всегда, не читается никак."""
    assert mitigation_checklist(CLEAN) == {}


def test_every_tagged_category_brings_its_own_measures():
    checklist = mitigation_checklist(FRONTIER)
    assert set(checklist) == set(tag_risks(FRONTIER))
    for risk, measures in checklist.items():
        assert measures == list(MITIGATIONS[risk])


def test_the_checklist_does_not_alias_the_mitigation_catalogue():
    checklist = mitigation_checklist(dict(CLEAN, competitive_pressure=True))
    checklist["ai_races"].append("ship it and hope")
    assert "ship it and hope" not in MITIGATIONS["ai_races"]


# ------------------------------------------------------------- aggregate_risk
def test_a_uniformly_good_set_scores_at_its_mean():
    got = aggregate_risk({"a": 0.9, "b": 0.9})
    assert got["mean"] == APPROX(0.9)
    assert got["score"] == APPROX(0.9)
    assert got["critical"] == []
    assert got["band"] == "strong"


def test_a_single_critical_indicator_takes_over_the_aggregate():
    """Провал по инфобезу не лечится успехами по остальным осям."""
    got = aggregate_risk({"a": 0.95, "b": 0.95, "c": 0.95, "d": 0.1})
    assert got["mean"] == APPROX(0.7375)
    assert got["worst"] == APPROX(0.1)
    assert got["critical"] == ["d"]
    assert got["score"] == APPROX(0.1)
    assert got["band"] == "critical"


def test_more_good_indicators_cannot_wash_out_a_critical_one():
    """Арифметика сойдётся, риск останется. Это и есть весь смысл функции."""
    base = {"a": 0.95, "b": 0.95, "c": 0.95, "d": 0.1}
    padded = dict(base, **{f"pad_{i}": 1.0 for i in range(10)})
    assert aggregate_risk(padded)["mean"] > aggregate_risk(base)["mean"]
    assert aggregate_risk(padded)["score"] == APPROX(aggregate_risk(base)["score"])
    assert aggregate_risk(padded)["band"] == "critical"


def test_weights_move_the_mean_but_never_the_critical_rule():
    weighted = aggregate_risk({"a": 1.0, "b": 0.6}, weights={"a": 3.0})
    assert weighted["mean"] == APPROX(0.9)
    assert weighted["score"] == APPROX(0.9)

    with_failure = aggregate_risk({"a": 1.0, "b": 0.2}, weights={"a": 9.0})
    assert with_failure["mean"] == APPROX(0.92)
    assert with_failure["score"] == APPROX(0.2)


def test_an_empty_score_set_is_an_error():
    """Агрегат по нулю показателей всё равно читался бы как «мы посчитали»."""
    with pytest.raises(ValueError):
        aggregate_risk({})


def test_a_score_off_the_scale_or_a_bad_weight_is_an_error():
    with pytest.raises(ValueError):
        aggregate_risk({"a": 1.5})
    with pytest.raises(ValueError):
        aggregate_risk({"a": 0.9}, weights={"b": 2.0})
    with pytest.raises(ValueError):
        aggregate_risk({"a": 0.9}, weights={"a": 0.0})


# ----------------------------------------------------------- stack_assessment
def test_a_full_strong_stack_is_complete_and_strong():
    got = stack_assessment({layer: 0.9 for layer in SOCIETAL_STACK})
    assert got["missing"] == []
    assert got["complete"] is True
    assert got["aggregate"]["band"] == "strong"


def test_one_perfect_layer_does_not_rescue_a_gappy_stack():
    """Финальный вывод фазы: полнота стека важнее силы одного слоя."""
    got = stack_assessment({"lab_scaling_policy": 1.0})
    assert got["complete"] is False
    assert got["aggregate"]["score"] == APPROX(0.0)
    assert got["aggregate"]["band"] == "critical"


def test_missing_lists_exactly_the_absent_layers():
    present = {"lab_scaling_policy": 0.8, "practitioner_controls": 0.8}
    got = stack_assessment(present)
    assert got["missing"] == sorted(set(SOCIETAL_STACK) - set(present))


def test_a_complete_stack_can_still_be_inadequate():
    """Полнота и достаточность — разные вопросы, и ответы бывают разные."""
    strengths = {layer: 0.9 for layer in SOCIETAL_STACK}
    strengths["practitioner_controls"] = 0.2
    got = stack_assessment(strengths)
    assert got["complete"] is True
    assert got["aggregate"]["critical"] == ["practitioner_controls"]
    assert got["aggregate"]["band"] == "critical"


def test_an_unknown_layer_is_an_error():
    with pytest.raises(ValueError):
        stack_assessment({"vibes_based_oversight": 1.0})


# ------------------------------------------------------ identify_organization
def test_the_nonprofit_and_the_nist_center_are_told_apart_by_host():
    assert identify_organization("https://safe.ai/ai-risk") == "CAIS"
    assert identify_organization("https://www.nist.gov/caisi") == "CAISI"


def test_the_acronym_in_the_path_proves_nothing():
    """Акронимы совпадают почти целиком; буквы в пути ничего не доказывают."""
    with pytest.raises(ValueError):
        identify_organization("https://example.com/caisi")


def test_host_case_does_not_matter():
    assert identify_organization("https://SAFE.AI/statement-on-ai-risk") == "CAIS"


def test_a_url_without_a_scheme_is_an_error():
    with pytest.raises(ValueError):
        identify_organization("safe.ai/ai-risk")


# ------------------------------------------------------------ sb53_obligations
def test_autonomy_and_harm_labels_do_not_invent_sb53_scope():
    deployment = dict(
        CLEAN,
        agent_autonomy_hours=10_000.0,
        harmful_capability_labels=("cbrn", "cyber"),
    )
    assert sb53_obligations(deployment) == []


def test_frontier_scope_is_strictly_greater_than_ten_to_the_twenty_six_ops():
    at_boundary = dict(CLEAN, training_compute_ops=SB53_FRONTIER_COMPUTE_OPS)
    above = dict(CLEAN, training_compute_ops=SB53_FRONTIER_COMPUTE_OPS + 1)
    assert sb53_obligations(at_boundary) == []
    assert sb53_obligations(above) == [
        "incident_reporting",
        "model_transparency_report",
        "whistleblower_protection",
    ]


def test_revenue_alone_does_not_make_a_nonfrontier_developer_large_frontier():
    deployment = dict(CLEAN, annual_gross_revenue_usd=10 * SB53_LARGE_REVENUE_USD)
    assert sb53_obligations(deployment) == []


def test_large_frontier_scope_requires_revenue_strictly_above_five_hundred_million():
    frontier = dict(CLEAN, training_compute_ops=SB53_FRONTIER_COMPUTE_OPS + 1)
    boundary = dict(frontier, annual_gross_revenue_usd=SB53_LARGE_REVENUE_USD)
    large = dict(frontier, annual_gross_revenue_usd=SB53_LARGE_REVENUE_USD + 1)
    assert "frontier_ai_framework" not in sb53_obligations(boundary)
    assert sb53_obligations(large) == [
        "anonymous_internal_reporting",
        "enhanced_transparency_report",
        "frontier_ai_framework",
        "incident_reporting",
        "internal_risk_assessment_reporting",
        "model_transparency_report",
        "whistleblower_protection",
    ]


def test_an_unknown_feature_is_still_an_error_here():
    with pytest.raises(ValueError):
        sb53_obligations({"autonomy_hours": 12.0})


def test_negative_compute_or_revenue_is_an_error():
    with pytest.raises(ValueError):
        sb53_obligations(dict(CLEAN, training_compute_ops=-1))
    with pytest.raises(ValueError):
        sb53_obligations(dict(CLEAN, annual_gross_revenue_usd=-1))


# -------------------------------------------------------- incident_report_status
def test_a_fresh_incident_has_the_full_window_left():
    got = incident_report_status(100.0, 100.0)
    assert got["deadline_at"] == APPROX(100.0 + SB53_GENERAL_REPORT_HOURS)
    assert got["hours_remaining"] == APPROX(SB53_GENERAL_REPORT_HOURS)
    assert got["overdue"] is False


def test_the_closing_moment_of_the_window_is_still_compliant():
    """Общий срок — 15 дней, и последний момент окна ещё compliant."""
    got = incident_report_status(100.0, 100.0 + SB53_GENERAL_REPORT_HOURS)
    assert got["hours_remaining"] == APPROX(0.0)
    assert got["overdue"] is False


def test_past_the_deadline_the_report_is_overdue():
    got = incident_report_status(100.0, 100.0 + SB53_GENERAL_REPORT_HOURS + 6.0)
    assert got["hours_remaining"] == APPROX(-6.0)
    assert got["overdue"] is True


def test_imminent_death_or_serious_injury_uses_the_24_hour_deadline():
    got = incident_report_status(100.0, 110.0, True)
    assert got["deadline_hours"] == APPROX(SB53_IMMINENT_REPORT_HOURS)
    assert got["deadline_at"] == APPROX(124.0)
    assert got["hours_remaining"] == APPROX(14.0)


def test_the_verdict_moves_with_now_and_nothing_else():
    """Отчёт о просрочке, зависящий от момента запуска, нельзя проверить."""
    assert incident_report_status(100.0, 110.0) == incident_report_status(100.0, 110.0)
    assert incident_report_status(100.0, 110.0)["overdue"] is False
    assert incident_report_status(100.0, 500.0)["overdue"] is True


def test_a_now_before_the_incident_or_a_bad_window_is_an_error():
    with pytest.raises(ValueError):
        incident_report_status(100.0, 99.0)
    with pytest.raises(ValueError):
        incident_report_status(100.0, 110.0, "false")
