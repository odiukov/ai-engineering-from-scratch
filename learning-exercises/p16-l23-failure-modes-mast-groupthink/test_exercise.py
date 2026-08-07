"""Тесты к уроку «Режимы отказов: MAST, groupthink, каскады». Правь exercise.py."""

import pytest

from exercise import (
    MAST_CATEGORIES,
    UnknownIncident,
    audit,
    cascade_load,
    category_rates,
    circuit_state,
    classify_incident,
    detect_groupthink,
    opinion_spread,
    rank_mitigations,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

MITIGATIONS = [
    ("role-contracts", ["spec"]),
    ("state-versioning", ["coord"]),
    ("verifier-agent", ["verify"]),
    ("acceptance-tests", ["spec", "verify"]),
]


# ------------------------------------------------------- classify_incident
def test_classify_incident_maps_role_ambiguity_to_specification():
    assert classify_incident(["role_ambiguity"]) == "spec"


def test_classify_incident_maps_state_drift_to_coordination():
    assert classify_incident(["state_drift", "lost_message"]) == "coord"


def test_classify_incident_maps_memory_poisoning_to_verification():
    """Отравление общей памяти — это дыра в проверке, а не в коммуникации."""
    assert classify_incident(["memory_poisoning"]) == "verify"


def test_classify_incident_takes_the_majority_category():
    assert classify_incident(["state_drift", "lost_message", "role_ambiguity"]) == "coord"


def test_classify_incident_breaks_ties_toward_the_more_frequent_category():
    assert classify_incident(["role_ambiguity", "state_drift"]) == "spec"


def test_classify_incident_ignores_symptoms_it_does_not_know():
    assert classify_incident(["agents_went_quiet", "role_ambiguity"]) == "spec"


def test_unclassifiable_incident_raises_its_own_exception_type():
    """Инцидент вне таксономии — сигнал дополнить таксономию, а не «всё ок»."""
    with pytest.raises(UnknownIncident):
        classify_incident(["agents_went_quiet"])


# ---------------------------------------------------------- category_rates
def test_category_rates_sum_to_one():
    rates = category_rates([["role_ambiguity"], ["state_drift"], ["unchecked_output"]])
    assert sum(rates.values()) == APPROX(1.0)


def test_category_rates_report_every_category_including_empty_ones():
    rates = category_rates([["role_ambiguity"]])
    assert sorted(rates) == sorted(MAST_CATEGORIES)
    assert rates["verify"] == APPROX(0.0)


def test_category_rates_count_repeats():
    rates = category_rates([["role_ambiguity"], ["state_drift"], ["role_ambiguity"]])
    assert rates["spec"] == APPROX(2 / 3)
    assert rates["coord"] == APPROX(1 / 3)


def test_category_rates_of_an_empty_sample_do_not_divide_by_zero():
    assert category_rates([]) == {c: APPROX(0.0) for c in MAST_CATEGORIES}


# -------------------------------------------------------- rank_mitigations
def test_rank_mitigations_orders_by_covered_failure_share():
    ranked = rank_mitigations({"spec": 0.5, "coord": 0.3, "verify": 0.2},
                              [("verifier-agent", ["verify"]), ("role-contracts", ["spec"])])
    assert ranked == [("role-contracts", APPROX(0.5)), ("verifier-agent", APPROX(0.2))]


def test_rank_mitigations_adds_up_multi_category_coverage():
    ranked = rank_mitigations({"spec": 0.5, "coord": 0.3, "verify": 0.2}, MITIGATIONS)
    assert ranked[0] == ("acceptance-tests", APPROX(0.7))


def test_rank_mitigations_breaks_ties_alphabetically():
    ranked = rank_mitigations({"spec": 0.5, "coord": 0.5},
                              [("z-fix", ["coord"]), ("a-fix", ["spec"])])
    assert [name for name, _ in ranked] == ["a-fix", "z-fix"]


def test_rank_mitigations_scores_an_irrelevant_measure_at_zero():
    """Мера, не закрывающая ни одной наблюдаемой категории, не приоритет."""
    ranked = rank_mitigations({"spec": 1.0}, [("gpu-upgrade", ["verify"])])
    assert ranked == [("gpu-upgrade", APPROX(0.0))]


# ----------------------------------------------------------- opinion_spread
def test_opinion_spread_of_identical_opinions_is_zero():
    """Нулевой разброс — прямой признак монокультуры."""
    assert opinion_spread([1.0, 1.0, 1.0]) == APPROX(0.0)


def test_opinion_spread_of_two_extremes():
    assert opinion_spread([0.0, 1.0]) == APPROX(0.5)


def test_opinion_spread_of_a_short_sample_is_zero():
    assert opinion_spread([]) == APPROX(0.0)
    assert opinion_spread([0.7]) == APPROX(0.0)


def test_opinion_spread_is_shift_invariant():
    """Разброс не зависит от того, вокруг чего согласились."""
    assert opinion_spread([0.1, 0.2, 0.3]) == APPROX(opinion_spread([5.1, 5.2, 5.3]))


def test_opinion_spread_grows_when_agents_diverge():
    assert opinion_spread([0.4, 0.6]) < opinion_spread([0.0, 1.0])


# -------------------------------------------------------- detect_groupthink
def test_groupthink_is_flagged_when_spread_collapses_as_confidence_rises():
    rounds = [([0.0, 0.5, 1.0], [0.5, 0.5, 0.5]),
              ([0.4, 0.5, 0.6], [0.7, 0.7, 0.7])]
    assert detect_groupthink(rounds) == 1


def test_healthy_convergence_without_confidence_inflation_is_not_groupthink():
    """Мнения сошлись, но уверенность не выросла — это просто согласие."""
    rounds = [([0.0, 0.5, 1.0], [0.5, 0.5, 0.5]),
              ([0.4, 0.5, 0.6], [0.5, 0.5, 0.5])]
    assert detect_groupthink(rounds) is None


def test_rising_confidence_with_intact_disagreement_is_not_groupthink():
    rounds = [([0.0, 0.5, 1.0], [0.4, 0.4, 0.4]),
              ([0.0, 0.5, 1.0], [0.9, 0.9, 0.9])]
    assert detect_groupthink(rounds) is None


def test_groupthink_is_compared_against_the_first_round_not_the_previous_one():
    """Медленное сползание по чуть-чуть иначе никогда не превысит порог."""
    rounds = [([0.0, 0.5, 1.0], [0.40, 0.40, 0.40]),
              ([0.1, 0.5, 0.9], [0.45, 0.45, 0.45]),
              ([0.2, 0.5, 0.8], [0.50, 0.50, 0.50]),
              ([0.45, 0.5, 0.55], [0.55, 0.55, 0.55])]
    assert detect_groupthink(rounds) == 3


def test_detect_groupthink_returns_the_first_offending_round():
    rounds = [([0.0, 0.5, 1.0], [0.5, 0.5, 0.5]),
              ([0.45, 0.5, 0.55], [0.8, 0.8, 0.8]),
              ([0.49, 0.5, 0.51], [0.9, 0.9, 0.9])]
    assert detect_groupthink(rounds) == 1


def test_detect_groupthink_on_an_empty_history_is_none():
    assert detect_groupthink([]) is None


# ------------------------------------------------------------ circuit_state
def test_circuit_opens_above_the_error_threshold():
    assert circuit_state([True] * 9 + [False], 0.05, 5) == "open"


def test_circuit_stays_closed_on_a_healthy_service():
    assert circuit_state([True] * 20, 0.05, 5) == "closed"


def test_circuit_needs_a_minimum_sample_before_opening():
    """Без min_calls первый же сбой вырубает исправный сервис на старте."""
    assert circuit_state([False, False], 0.05, 5) == "closed"


def test_circuit_threshold_is_strict():
    """Ровно на пороге предохранитель ещё закрыт."""
    assert circuit_state([True] * 9 + [False], 0.1, 5) == "closed"


# ------------------------------------------------------------- cascade_load
def test_cascade_load_starts_at_the_base_load():
    assert cascade_load(100.0, 0.5, 2, 3)[0] == APPROX(100.0)


def test_cascade_load_amplifies_geometrically():
    assert cascade_load(100.0, 0.5, 2, 2) == APPROX([100.0, 200.0, 400.0])


def test_cascade_without_failures_stays_flat():
    assert cascade_load(100.0, 0.0, 5, 3) == APPROX([100.0] * 4)


def test_circuit_breaker_caps_the_retry_storm():
    assert cascade_load(100.0, 0.5, 2, 2, breaker_cap=150.0) == APPROX(
        [100.0, 150.0, 150.0]
    )


def test_breaker_keeps_deep_cascades_bounded_while_the_raw_one_explodes():
    """Пять уровней — разница между «пережили» и «кластер лёг»."""
    raw = cascade_load(100.0, 0.5, 2, 5)
    capped = cascade_load(100.0, 0.5, 2, 5, breaker_cap=200.0)
    assert raw[-1] > 10 * capped[-1]
    assert max(capped) <= 200.0


def test_cascade_load_returns_one_entry_per_level():
    assert len(cascade_load(10.0, 0.1, 3, 7)) == 8


# ------------------------------------------------------------------- audit
def test_audit_ranks_the_measure_that_covers_the_dominant_category():
    ranked = audit([["role_ambiguity"], ["role_ambiguity"], ["unchecked_output"]],
                   MITIGATIONS, top_k=1)
    assert ranked == [("acceptance-tests", APPROX(1.0))]


def test_audit_respects_top_k():
    assert len(audit([["role_ambiguity"], ["state_drift"]], MITIGATIONS, top_k=2)) == 2


def test_audit_propagates_an_unclassifiable_trace():
    """Аудит не имеет права тихо выбросить непонятную трассу из статистики."""
    with pytest.raises(UnknownIncident):
        audit([["role_ambiguity"], ["mystery"]], MITIGATIONS)
