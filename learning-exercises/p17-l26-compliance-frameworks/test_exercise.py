"""Тесты к уроку «Compliance: матрица контролей, полей и сроков». Правь exercise.py."""

import pytest

from exercise import (
    coverage_gaps,
    frameworks_for,
    record_is_complete,
    records_to_purge,
    required_controls,
    required_frameworks,
    required_log_fields,
    retention_days,
)

FULL_RECORD = {
    "ts": "2026-08-07T10:00:00Z",
    "user": "u1",
    "tenant": "t1",
    "action": "call_model",
    "model": "claude",
    "model_version": "2026-05",
    "risk_tier": "limited",
    "prompt_hash": "abc123",
    "response_hash": "def456",
    "pii_redacted": True,
    "phi_redacted": True,
    "legal_basis": "contract",
    "decision_outcome": "approved",
    "appeal_channel": "support@example.com",
    "human_review": False,
}


# --------------------------------------------------------- frameworks_for
def test_one_control_satisfies_four_frameworks_at_once():
    """Ради этого матрицу и строят: один контроль закрывает много пунктов."""
    assert frameworks_for("access logging") == ["GDPR", "HIPAA", "ISO 27001", "SOC 2"]


def test_a_narrow_control_satisfies_exactly_one_framework():
    assert frameworks_for("data subject rights") == ["GDPR"]


def test_a_typo_in_a_control_name_is_an_error_not_an_empty_answer():
    """Тихий [] превращает опечатку в ложное «покрытия нет»."""
    with pytest.raises(ValueError):
        frameworks_for("acess logging")


# ----------------------------------------------------- required_frameworks
def test_us_healthcare_pulls_in_hipaa():
    assert required_frameworks("US", "healthcare") == ["HIPAA", "ISO 27001", "SOC 2"]


def test_eu_customers_pull_in_gdpr_and_the_eu_ai_act():
    assert required_frameworks("EU", "B2B SaaS") == [
        "EU AI Act",
        "GDPR",
        "ISO 27001",
        "SOC 2",
    ]


def test_an_unknown_profile_is_an_error():
    with pytest.raises(ValueError):
        required_frameworks("Mars", "B2B SaaS")


# ------------------------------------------------------- required_controls
def test_gdpr_requires_inference_time_redaction():
    """Пост-обработка не спасает — редактирование обязано быть контролем."""
    assert "PII redaction (inference-time)" in required_controls(["GDPR"])


def test_no_frameworks_means_no_controls():
    assert required_controls([]) == []


def test_adding_a_framework_never_shrinks_the_control_list():
    one = set(required_controls(["SOC 2"]))
    two = set(required_controls(["SOC 2", "HIPAA"]))
    assert one <= two and "BAA signed" in two - one


def test_soc2_alone_does_not_require_a_baa():
    assert "BAA signed" not in required_controls(["SOC 2"])


# ---------------------------------------------------------- coverage_gaps
def test_a_fully_implemented_matrix_has_no_gaps():
    frameworks = required_frameworks("US", "B2B SaaS")
    assert coverage_gaps(required_controls(frameworks), frameworks) == {}


def test_missing_baa_is_reported_against_hipaa_only():
    frameworks = ["SOC 2", "HIPAA"]
    implemented = [c for c in required_controls(frameworks) if c != "BAA signed"]
    assert coverage_gaps(implemented, frameworks) == {"HIPAA": ["BAA signed"]}


def test_one_missing_control_is_reported_against_every_framework_it_covered():
    """«access logging» не сделан — краснеют сразу четыре фреймворка."""
    frameworks = ["SOC 2", "HIPAA", "GDPR", "ISO 27001"]
    implemented = [c for c in required_controls(frameworks) if c != "access logging"]
    gaps = coverage_gaps(implemented, frameworks)
    assert sorted(gaps) == ["GDPR", "HIPAA", "ISO 27001", "SOC 2"]


def test_implementing_extra_controls_does_not_create_gaps():
    assert coverage_gaps(list(required_controls(["GDPR", "HIPAA"])), ["GDPR"]) == {}


# ------------------------------------------------------ required_log_fields
def test_iso_27001_needs_only_the_basics():
    assert required_log_fields(["ISO 27001"]) == ["action", "ts", "user"]


def test_fields_are_unioned_not_intersected():
    """Журнал один на всех, значит он обязан удовлетворить самого строгого."""
    assert required_log_fields(["ISO 27001", "GDPR"]) == [
        "action",
        "legal_basis",
        "pii_redacted",
        "prompt_hash",
        "ts",
        "user",
    ]


def test_the_eu_ai_act_adds_model_version_and_risk_tier():
    extra = set(required_log_fields(["EU AI Act"])) - set(required_log_fields(["SOC 2"]))
    assert extra == {"model_version", "risk_tier"}


def test_an_unknown_framework_in_the_field_matrix_is_an_error():
    with pytest.raises(ValueError):
        required_log_fields(["ISO 9001"])


# ----------------------------------------------------- record_is_complete
def test_a_full_record_satisfies_every_framework_in_the_matrix():
    frameworks = required_frameworks("Global", "enterprise")
    assert record_is_complete(FULL_RECORD, frameworks) == (True, [])


def test_a_missing_field_is_named_in_the_answer():
    assert record_is_complete({"ts": 1, "user": "u"}, ["ISO 27001"]) == (False, ["action"])


def test_a_null_value_counts_as_a_missing_field():
    """«legal_basis: null» — это отсутствие основания, а не его наличие."""
    record = dict(FULL_RECORD, legal_basis=None)
    assert record_is_complete(record, ["GDPR"]) == (False, ["legal_basis"])


def test_a_false_value_is_a_value_and_not_a_gap():
    """human_review=False значит «человек не смотрел», и это записано честно."""
    record = dict(FULL_RECORD, human_review=False)
    assert record_is_complete(record, ["ISO 42001"])[0] is True


def test_the_same_record_can_pass_one_framework_and_fail_another():
    record = {"ts": "2026-08-07", "user": "u", "action": "call_model"}
    assert record_is_complete(record, ["ISO 27001"])[0] is True
    assert record_is_complete(record, ["GDPR"])[0] is False


# --------------------------------------------------------- retention_days
def test_soc2_keeps_the_log_for_a_year():
    assert retention_days(["SOC 2"]) == 365


def test_hipaa_stretches_retention_to_six_years():
    assert retention_days(["HIPAA"]) == 2190


def test_a_mixed_profile_takes_the_maximum_not_the_minimum():
    """Один HIPAA в списке тянет хранение всего журнала с года до шести лет."""
    assert retention_days(["SOC 2", "GDPR", "HIPAA"]) == 2190


def test_no_frameworks_means_no_retention_requirement():
    assert retention_days([]) == 0


# ------------------------------------------------------- records_to_purge
def test_old_records_are_purged_and_fresh_ones_are_kept():
    recs = [{"ts": "2024-01-01"}, {"ts": "2026-08-01"}]
    assert records_to_purge(recs, ["SOC 2"], "2026-08-07") == [{"ts": "2024-01-01"}]


def test_a_record_exactly_at_the_retention_edge_is_still_kept():
    """Ошибка на единицу здесь — это провал аудита."""
    recs = [{"ts": "2025-08-07"}]
    assert records_to_purge(recs, ["SOC 2"], "2026-08-07") == []


def test_a_record_one_day_past_the_edge_is_purged():
    recs = [{"ts": "2025-08-06"}]
    assert records_to_purge(recs, ["SOC 2"], "2026-08-07") == recs


def test_hipaa_keeps_what_soc2_would_have_deleted():
    recs = [{"ts": "2023-01-01T12:00:00Z"}]
    assert records_to_purge(recs, ["SOC 2"], "2026-08-07") == recs
    assert records_to_purge(recs, ["HIPAA"], "2026-08-07") == []


def test_purging_reads_the_date_from_the_now_argument():
    """Никакого time.time(): срок хранения считается на переданную дату."""
    recs = [{"ts": "2025-01-01"}]
    assert records_to_purge(recs, ["SOC 2"], "2025-06-01") == []
    assert records_to_purge(recs, ["SOC 2"], "2027-06-01") == recs
