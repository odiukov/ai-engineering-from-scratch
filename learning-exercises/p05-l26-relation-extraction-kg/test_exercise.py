"""Тесты к уроку «Relation extraction и сборка knowledge graph». Правь exercise.py."""

import pytest

from exercise import (
    build_graph,
    canonicalize,
    canonicalize_triples,
    extract_triples,
    filter_verified,
    hallucination_rate,
    neighbors,
    verify_span,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

PATTERNS = [
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
    (r"(?P<s>[A-Z]\w+) was born in (?P<o>[A-Z]\w+)", "was born in"),
    (r"(?P<s>[A-Z]\w+) works at (?P<o>[A-Z]\w+)", "works at"),
]

RELATION_MAP = {
    "was born in": "P19",
    "is a native of": "P19",
    "came from": "P19",
    "works at": "P108",
    "founded": "P112",
}

TEXT = "Tim Cook works at Apple. Steve was born in California."


def extraction(subject, s_span, relation, obj, o_span):
    """Формат, в котором LLM возвращает триплет со спанами."""
    return {
        "subject": subject,
        "subject_span": s_span,
        "relation": relation,
        "object": obj,
        "object_span": o_span,
    }


# ----------------------------------------------------------- extract_triples
def test_extract_triples_finds_a_single_pattern():
    got = extract_triples("Steve founded Apple", PATTERNS)
    assert got == [("Steve", "founded", "Apple")]


def test_extract_triples_finds_every_occurrence_of_one_pattern():
    text = "Steve founded Apple and Bill founded Microsoft"
    assert extract_triples(text, PATTERNS) == [
        ("Steve", "founded", "Apple"),
        ("Bill", "founded", "Microsoft"),
    ]


def test_extract_triples_groups_results_by_pattern_order():
    """Сначала все `founded`, потом все `was born in` — так задан PATTERNS."""
    text = "Steve was born in California. Bill founded Microsoft"
    assert extract_triples(text, PATTERNS) == [
        ("Bill", "founded", "Microsoft"),
        ("Steve", "was born in", "California"),
    ]


def test_extract_triples_of_unmatched_text_is_empty():
    assert extract_triples("nothing interesting happened here", PATTERNS) == []


def test_extract_triples_with_no_patterns_is_empty():
    assert extract_triples("Steve founded Apple", []) == []


# --------------------------------------------------------------- verify_span
def test_verify_span_accepts_an_exact_match():
    assert verify_span("Tim Cook works", "Tim Cook", (0, 8)) is True


def test_verify_span_rejects_a_hallucinated_surface():
    assert verify_span("Tim Cook works", "Steve Jobs", (0, 8)) is False


def test_verify_span_rejects_an_out_of_range_end():
    """Срез в Python молча обрезается — границы надо проверять руками."""
    assert verify_span("Tim Cook works", "Tim Cook works", (0, 999)) is False


def test_verify_span_rejects_a_negative_start():
    assert verify_span("Tim Cook works", "works", (-5, 14)) is False


def test_verify_span_rejects_a_reversed_span():
    assert verify_span("Tim Cook works", "Tim", (8, 3)) is False


def test_verify_span_rejects_an_empty_surface():
    assert verify_span("Tim Cook works", "", (0, 0)) is False


# ------------------------------------------------------------ filter_verified
def test_filter_verified_keeps_a_fully_grounded_triple():
    items = [extraction("Tim Cook", (0, 8), "works at", "Apple", (18, 23))]
    assert filter_verified(TEXT, items) == [("Tim Cook", "works at", "Apple")]


def test_filter_verified_drops_a_triple_with_a_hallucinated_object():
    """Правильный субъект не спасает: факт целиком не подтверждён."""
    items = [extraction("Tim Cook", (0, 8), "works at", "Google", (18, 23))]
    assert filter_verified(TEXT, items) == []


def test_filter_verified_drops_a_triple_with_a_hallucinated_subject():
    items = [extraction("Steve Jobs", (0, 8), "works at", "Apple", (18, 23))]
    assert filter_verified(TEXT, items) == []


def test_filter_verified_keeps_only_the_grounded_ones():
    items = [
        extraction("Tim Cook", (0, 8), "works at", "Apple", (18, 23)),
        extraction("Tim Cook", (0, 8), "founded", "Google", (18, 23)),
        extraction("Steve", (25, 30), "was born in", "California", (43, 53)),
    ]
    assert filter_verified(TEXT, items) == [
        ("Tim Cook", "works at", "Apple"),
        ("Steve", "was born in", "California"),
    ]


def test_filter_verified_of_nothing_is_empty():
    assert filter_verified(TEXT, []) == []


# -------------------------------------------------------- hallucination_rate
def test_hallucination_rate_is_zero_when_everything_is_grounded():
    items = [extraction("Tim Cook", (0, 8), "works at", "Apple", (18, 23))]
    assert hallucination_rate(TEXT, items) == APPROX(0.0)


def test_hallucination_rate_is_one_when_nothing_is_grounded():
    items = [extraction("Elon Musk", (0, 8), "works at", "Tesla", (18, 23))]
    assert hallucination_rate(TEXT, items) == APPROX(1.0)


def test_verify_step_removes_exactly_the_hallucinated_triples():
    """Смысл AEVS: доля отвергнутых + доля выживших = 1, ничего не теряется."""
    items = [
        extraction("Tim Cook", (0, 8), "works at", "Apple", (18, 23)),
        extraction("Tim Cook", (0, 8), "founded", "Google", (18, 23)),
        extraction("Elon", (0, 4), "works at", "Apple", (18, 23)),
        extraction("Steve", (25, 30), "was born in", "California", (43, 53)),
    ]
    rate = hallucination_rate(TEXT, items)
    survived = len(filter_verified(TEXT, items))
    assert rate == APPROX(0.5)
    assert survived / len(items) == APPROX(1.0 - rate)


def test_hallucination_rate_of_nothing_is_zero_not_an_error():
    assert hallucination_rate(TEXT, []) == APPROX(0.0)


# ------------------------------------------------------------- canonicalize
def test_canonicalize_maps_a_known_relation():
    assert canonicalize("was born in", RELATION_MAP) == "P19"


def test_canonicalize_ignores_case_and_padding():
    assert canonicalize("  Was Born In ", RELATION_MAP) == "P19"


def test_canonicalize_collapses_synonymous_phrasings():
    """Ровно ради этого шаг и нужен: три формулировки — одно ребро графа."""
    ids = {canonicalize(r, RELATION_MAP) for r in ("was born in", "came from", "is a native of")}
    assert ids == {"P19"}


def test_canonicalize_of_an_unmapped_relation_is_none():
    assert canonicalize("hangs out with", RELATION_MAP) is None


# ------------------------------------------------------ canonicalize_triples
def test_canonicalize_triples_rewrites_the_relation_only():
    triples = [("Tim", "works at", "Apple")]
    assert canonicalize_triples(triples, RELATION_MAP) == [("Tim", "P108", "Apple")]


def test_canonicalize_triples_drops_open_ie_leftovers():
    triples = [("Tim", "works at", "Apple"), ("Tim", "likes", "jazz")]
    assert canonicalize_triples(triples, RELATION_MAP) == [("Tim", "P108", "Apple")]


def test_canonicalize_triples_makes_the_graph_queryable():
    """До канонизации два синонима дают два разных ребра, после — одно."""
    triples = [("Tim", "was born in", "Alabama"), ("Tim", "came from", "Alabama")]
    raw = build_graph(triples)
    canonical = build_graph(canonicalize_triples(triples, RELATION_MAP))
    assert len(raw["Tim"]) == 2
    assert len(canonical["Tim"]) == 1


def test_canonicalize_triples_of_nothing_is_empty():
    assert canonicalize_triples([], RELATION_MAP) == []


# --------------------------------------------------------------- build_graph
def test_build_graph_groups_edges_by_subject():
    graph = build_graph([("Tim", "P108", "Apple"), ("Tim", "P19", "Alabama")])
    assert graph == {"Tim": [("P108", "Apple"), ("P19", "Alabama")]}


def test_build_graph_keeps_subjects_separate():
    graph = build_graph([("Tim", "P108", "Apple"), ("Steve", "P112", "Apple")])
    assert set(graph) == {"Tim", "Steve"}


def test_build_graph_deduplicates_the_same_fact():
    """Один факт из десяти документов — одно ребро, а не десять."""
    graph = build_graph([("Tim", "P108", "Apple")] * 10)
    assert graph["Tim"] == [("P108", "Apple")]


def test_build_graph_of_nothing_is_empty():
    assert build_graph([]) == {}


# ----------------------------------------------------------------- neighbors
def test_neighbors_returns_every_edge_by_default():
    graph = build_graph([("Tim", "P108", "Apple"), ("Tim", "P19", "Alabama")])
    assert neighbors(graph, "Tim") == [("P108", "Apple"), ("P19", "Alabama")]


def test_neighbors_filters_by_relation():
    graph = build_graph([("Tim", "P108", "Apple"), ("Tim", "P19", "Alabama")])
    assert neighbors(graph, "Tim", "P108") == [("P108", "Apple")]


def test_neighbors_of_an_unknown_node_is_empty():
    assert neighbors(build_graph([("Tim", "P108", "Apple")]), "Nobody") == []


def test_neighbors_does_not_expose_the_graph_itself():
    graph = build_graph([("Tim", "P108", "Apple")])
    neighbors(graph, "Tim").append(("P999", "Fake"))
    assert graph["Tim"] == [("P108", "Apple")]


def test_end_to_end_from_text_to_a_queryable_graph():
    """Весь урок в одном тесте: текст -> триплеты -> онтология -> граф -> запрос."""
    triples = extract_triples(TEXT, PATTERNS)
    graph = build_graph(canonicalize_triples(triples, RELATION_MAP))
    assert neighbors(graph, "Cook", "P108") == [("P108", "Apple")]
    assert neighbors(graph, "Steve", "P19") == [("P19", "California")]
