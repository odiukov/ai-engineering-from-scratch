"""Тесты к уроку «OpenAI Preparedness Framework и DeepMind Frontier Safety
Framework». Правь exercise.py."""

import pytest

from exercise import (
    ANTHROPIC_RSP_V3,
    CAPABILITY_AXES,
    DEEPMIND_FSF_V3,
    OPENAI_PF_V2,
    POLICIES,
    ccl_reached,
    classify,
    compare,
    coverage_report,
    gating_divergence,
    is_gated,
    required_artifacts,
    sandbagging_correction,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Модель на уровне ML R&D autonomy level 1: пайплайн автоматизирован целиком
# и дешевле человека с инструментами.
AT_LEVEL_1 = {"rnd_pipeline_automation_share": 1.0, "cost_ratio_vs_human": 0.8}
# Та же автоматизация, но втрое дороже человека — это не level 1.
EXPENSIVE = {"rnd_pipeline_automation_share": 1.0, "cost_ratio_vs_human": 3.0}
# Модель, которая занижает себя ровно настолько, чтобы не пересечь порог.
SANDBAGGED = {"rnd_pipeline_automation_share": 0.7, "cost_ratio_vs_human": 1.4}


# ------------------------------------------------------------------ classify
def test_tracked_and_research_are_read_from_the_same_table():
    assert classify(OPENAI_PF_V2, "rnd_automation")["classification"] == "Tracked"
    assert classify(OPENAI_PF_V2, "long_range_autonomy")["classification"] == "Research"


def test_a_capability_absent_from_a_policy_reads_as_not_covered():
    """Locate: чего в документе не находится, того документ не покрывает."""
    row = classify(ANTHROPIC_RSP_V3, "autonomous_replication")
    assert row["covered"] is False
    assert row["classification"] is None
    assert row["action"] is None


def test_the_row_names_the_policy_and_the_axis_it_answers_about():
    """Строка без подписи, о ком и о чём она, в сравнении бесполезна."""
    row = classify(DEEPMIND_FSF_V3, "cyber_uplift")
    assert row["policy"] == DEEPMIND_FSF_V3["name"]
    assert row["capability"] == "cyber_uplift"
    assert row["covered"] is True


def test_a_misspelled_axis_is_an_error_not_a_policy_gap():
    with pytest.raises(ValueError):
        classify(OPENAI_PF_V2, "long_range_autonmy")


# ------------------------------------------------------------------ is_gated
def test_tracked_gates_but_research_only_observes():
    """«Potential mitigations» — не обязательство. Это и есть весь делёж."""
    assert is_gated(OPENAI_PF_V2, "cyber_uplift") is True
    assert is_gated(OPENAI_PF_V2, "long_range_autonomy") is False


def test_monitoring_only_classification_does_not_gate():
    """DeepMind сама пишет, что мониторинга долго не хватит: он не gate."""
    assert classify(DEEPMIND_FSF_V3, "sandbagging")["covered"] is True
    assert is_gated(DEEPMIND_FSF_V3, "sandbagging") is False


def test_an_uncovered_capability_gates_nothing():
    assert is_gated(ANTHROPIC_RSP_V3, "autonomous_replication") is False


def test_every_policy_gates_the_shared_high_stakes_axes():
    """Сходство трёх документов проверяемо, а не только декларируемо."""
    for axis in ("rnd_automation", "cyber_uplift", "bio_uplift"):
        assert all(is_gated(p, axis) for p in POLICIES)


# --------------------------------------------------------- required_artifacts
def test_a_gated_capability_requires_the_full_report_set():
    assert required_artifacts(OPENAI_PF_V2, "cyber_uplift") == [
        "capabilities_report",
        "safeguards_report",
        "sag_review",
    ]


def test_a_research_capability_requires_no_reports():
    assert required_artifacts(OPENAI_PF_V2, "long_range_autonomy") == []


def test_the_returned_list_does_not_alias_the_policy():
    """Отчёт ушёл наружу; правка списка не должна менять саму политику."""
    got = required_artifacts(OPENAI_PF_V2, "bio_uplift")
    got.append("vibes")
    assert "vibes" not in OPENAI_PF_V2["artifacts"]
    assert required_artifacts(OPENAI_PF_V2, "bio_uplift") == [
        "capabilities_report",
        "safeguards_report",
        "sag_review",
    ]


# ------------------------------------------------------------------- compare
def test_comparison_covers_every_policy_on_the_same_axis():
    rows = compare("rnd_automation")
    assert set(rows) == {p["name"] for p in POLICIES}
    assert all(row["gated"] is True for row in rows.values())


def test_the_same_capability_gets_three_different_verdicts():
    """Long-range Autonomy: Research у OpenAI, gate у двух остальных."""
    rows = compare("long_range_autonomy")
    assert rows[OPENAI_PF_V2["name"]]["gated"] is False
    assert rows[ANTHROPIC_RSP_V3["name"]]["gated"] is True
    assert rows[DEEPMIND_FSF_V3["name"]]["gated"] is True


def test_a_policy_missing_the_axis_still_appears_in_the_comparison():
    """Пропуск оси обязан быть в таблице, а не выпадать из неё молча."""
    rows = compare("autonomous_replication")
    assert set(rows) == {p["name"] for p in POLICIES}
    gap = rows[ANTHROPIC_RSP_V3["name"]]
    assert gap["covered"] is False
    assert gap["gated"] is False


def test_comparison_keeps_the_action_text_for_reading():
    rows = compare("undermining_safeguards")
    assert rows[ANTHROPIC_RSP_V3["name"]]["action"] == "refuses training / deploy"
    assert rows[OPENAI_PF_V2["name"]]["action"] == "observed; potential mitigations"


# ------------------------------------------------------------ coverage_report
def test_a_policy_covering_every_axis_reports_no_gaps():
    report = coverage_report(OPENAI_PF_V2)
    assert report["policy"] == OPENAI_PF_V2["name"]
    assert report["uncovered"] == []
    assert report["covered"] == sorted(CAPABILITY_AXES)


def test_a_skipped_axis_shows_up_in_the_report():
    assert coverage_report(ANTHROPIC_RSP_V3)["uncovered"] == ["autonomous_replication"]


def test_covered_and_uncovered_partition_the_axes():
    """Ось не может ни потеряться, ни попасть в оба списка сразу."""
    for policy in POLICIES:
        report = coverage_report(policy)
        assert sorted(report["covered"] + report["uncovered"]) == sorted(CAPABILITY_AXES)
        assert not set(report["covered"]) & set(report["uncovered"])


# ---------------------------------------------------------- gating_divergence
def test_divergence_lists_exactly_the_axes_with_conflicting_consequences():
    assert gating_divergence() == [
        "autonomous_replication",
        "long_range_autonomy",
        "undermining_safeguards",
    ]


def test_unanimous_non_gating_is_not_a_divergence():
    """Все три не гейтят sandbagging — согласие, хоть и неутешительное."""
    assert "sandbagging" not in gating_divergence()
    assert not any(is_gated(p, "sandbagging") for p in POLICIES)


def test_a_single_policy_never_diverges_from_itself():
    for policy in POLICIES:
        assert gating_divergence(policies=(policy,)) == []


def test_divergence_is_computed_from_consequences_not_from_labels():
    """Названия разрядов у трёх лабораторий разные всегда — это шум."""
    labels = {classify(p, "cyber_uplift")["classification"] for p in POLICIES}
    assert len(labels) == 3
    assert "cyber_uplift" not in gating_divergence()


# --------------------------------------------------------------- ccl_reached
def test_a_single_domain_threshold_fires_alone():
    assert ccl_reached({"cyber_uplift": 0.6}) == ["cyber"]


def test_ml_rnd_level_one_needs_full_automation_and_competitive_cost():
    assert ccl_reached(AT_LEVEL_1) == ["ml_rnd"]


def test_full_automation_at_a_bad_price_is_not_level_one():
    """Автоматизация втрое дороже человека — не ML R&D autonomy level 1."""
    assert ccl_reached(EXPENSIVE) == []


def test_missing_measurements_never_count_towards_a_threshold():
    assert ccl_reached({}) == []
    assert ccl_reached({"rnd_pipeline_automation_share": 1.0}) == []


def test_several_domains_can_fire_at_once_and_come_back_sorted():
    everything = dict(AT_LEVEL_1, cyber_uplift=0.9, bio_uplift=0.7)
    assert ccl_reached(everything) == ["bio", "cyber", "ml_rnd"]


def test_a_domain_without_thresholds_is_an_error_not_a_free_pass():
    """all([]) == True: пустое правило иначе «пересекается» на любых данных."""
    with pytest.raises(ValueError):
        ccl_reached({"cyber_uplift": 0.0},
                    thresholds={"ghost": {"at_least": {}, "at_most": {}}})


# ------------------------------------------------------ sandbagging_correction
def test_capability_metrics_are_corrected_upwards():
    assert sandbagging_correction({"cyber_uplift": 0.4}, 0.0)["cyber_uplift"] == APPROX(0.4)
    assert sandbagging_correction({"cyber_uplift": 0.4}, 0.5)["cyber_uplift"] == APPROX(0.8)


def test_cost_metrics_are_corrected_in_the_opposite_direction():
    """Показная дороговизна — тоже занижение возможностей."""
    got = sandbagging_correction({"cost_ratio_vs_human": 2.0}, 0.5)
    assert got["cost_ratio_vs_human"] == APPROX(1.0)


def test_correction_can_reveal_a_threshold_the_raw_numbers_missed():
    """Рамка работает ровно настолько, насколько работает замер."""
    assert ccl_reached(SANDBAGGED) == []
    assert ccl_reached(sandbagging_correction(SANDBAGGED, 0.4)) == ["ml_rnd"]


def test_correction_never_removes_a_reached_domain():
    """Поправка не имеет права успокаивать отчёт."""
    for rate in (0.0, 0.1, 0.3, 0.6):
        raw = set(ccl_reached(AT_LEVEL_1))
        fixed = set(ccl_reached(sandbagging_correction(AT_LEVEL_1, rate)))
        assert raw <= fixed


def test_correction_does_not_mutate_the_input():
    raw = dict(SANDBAGGED)
    sandbagging_correction(raw, 0.4)
    assert raw == SANDBAGGED


def test_an_unknown_metric_or_an_out_of_range_rate_is_an_error():
    with pytest.raises(ValueError):
        sandbagging_correction({"vibes": 1.0}, 0.2)
    with pytest.raises(ValueError):
        sandbagging_correction({"cyber_uplift": 0.4}, 1.0)
    with pytest.raises(ValueError):
        sandbagging_correction({"cyber_uplift": 0.4}, -0.1)
