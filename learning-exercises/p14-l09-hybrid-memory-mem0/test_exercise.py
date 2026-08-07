"""Тесты к уроку «Гибридная память: вектор + граф + KV». Правь exercise.py."""

import pytest

from exercise import (
    DEFAULT_WEIGHTS,
    HALFLIFE_SECONDS,
    cosine,
    embed,
    fuse_score,
    graph_add_edge,
    graph_neighbors,
    hybrid_search,
    kv_lookup,
    vector_search,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def rec(rid, text, user_id="ava", ts=0.0, importance=0.5, kv=None,
        scope="user", session_id="s001"):
    """Запись гибридной памяти для теста."""
    return {"rid": rid, "text": text, "user_id": user_id, "session_id": session_id,
            "scope": scope, "importance": importance, "ts": ts, "kv": dict(kv or {})}


# --------------------------------------------------------------------- embed
def test_embed_is_deterministic():
    """Случайная соль встроенного hash() сломала бы индекс на следующем запуске."""
    assert embed("ava lives in Berlin") == embed("ava lives in Berlin")


def test_embed_has_the_requested_dimension():
    assert len(embed("ava lives in Berlin", dim=8)) == 8


def test_embed_ignores_word_order():
    """Мешок слов: порядок в эмбеддинг не попадает."""
    assert embed("ava lives") == embed("lives ava")


def test_embed_separates_different_texts():
    assert embed("ava lives in Berlin") != embed("bob requested a refund")


def test_embed_of_empty_text_is_all_zeros():
    assert embed("", dim=8) == [0.0] * 8


# -------------------------------------------------------------------- cosine
def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 0.0, 2.0], [1.0, 0.0, 2.0]) == APPROX(1.0)


def test_cosine_ignores_length():
    """Важен угол, а не длина: удвоенный вектор — тот же смысл."""
    assert cosine([1.0, 2.0], [2.0, 4.0]) == APPROX(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_with_a_zero_vector_answers_zero_instead_of_dividing():
    """Текст без слов не должен ронять поиск ZeroDivisionError."""
    assert cosine([0.0, 0.0], [1.0, 0.0]) == APPROX(0.0)


# ------------------------------------------------------------- vector_search
def test_vector_search_puts_the_exact_text_first():
    records = [rec("m001", "bob requested a refund for invoice 4711"),
               rec("m002", "ava lives in Berlin")]
    hits = vector_search(records, "ava lives in Berlin")
    assert hits[0][1]["rid"] == "m002"
    assert hits[0][0] == APPROX(1.0)


def test_vector_search_drops_records_with_zero_similarity():
    records = [rec("m001", "bob requested a refund for invoice 4711")]
    assert vector_search(records, "") == []


def test_vector_search_respects_top_k():
    records = [rec(f"m{i:03d}", f"ava lives in city number {i}") for i in range(5)]
    assert len(vector_search(records, "ava lives in city", top_k=2)) == 2


def test_vector_search_breaks_ties_by_record_id():
    """Одинаковый запрос обязан давать одинаковый контекст."""
    records = [rec("m009", "ava lives in Berlin"), rec("m002", "ava lives in Berlin")]
    hits = vector_search(records, "ava lives in Berlin")
    assert [h[1]["rid"] for h in hits] == ["m002", "m009"]


# ---------------------------------------------------------------- kv_lookup
def test_kv_lookup_returns_the_freshest_value():
    records = [rec("m001", "ava lives in Berlin", ts=1.0, kv={"city": "Berlin"}),
               rec("m002", "ava moved to Lisbon", ts=2.0, kv={"city": "Lisbon"})]
    assert kv_lookup(records, "ava", "city")["rid"] == "m002"


def test_kv_lookup_never_crosses_users():
    """Так и случается «ассистент рассказал Алисе про проект Боба»."""
    records = [rec("m001", "bob lives in Porto", user_id="bob", kv={"city": "Porto"})]
    assert kv_lookup(records, "ava", "city") is None


def test_kv_lookup_returns_none_for_an_unknown_fact_type():
    records = [rec("m001", "ava lives in Berlin", kv={"city": "Berlin"})]
    assert kv_lookup(records, "ava", "phone") is None


# ----------------------------------------------------------- graph_add_edge
def test_graph_add_edge_marks_the_contradicted_edge_instead_of_deleting_it():
    """Soft delete: удалить факт — значит потерять ответ «а что было в марте»."""
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    edges = graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert len(edges) == 2
    assert (edges[0]["valid"], edges[0]["invalid_from"]) == (False, 200.0)
    assert (edges[1]["valid"], edges[1]["obj"]) == (True, "Lisbon")


def test_graph_add_edge_leaves_other_relations_valid():
    """Противоречие — конфликт по одному отношению, а не по всему субъекту."""
    edges = graph_add_edge([], "ava", "owns_project", "curriculum", now=100.0)
    edges = graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert [e["valid"] for e in edges] == [True, True]


def test_graph_add_edge_leaves_the_input_list_alone():
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert edges[0]["valid"] is True


# ---------------------------------------------------------- graph_neighbors
def test_graph_neighbors_returns_only_the_current_truth():
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    edges = graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert [e["obj"] for e in graph_neighbors(edges, "ava")] == ["Lisbon"]


def test_graph_neighbors_answers_what_was_true_back_then():
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    edges = graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert [e["obj"] for e in graph_neighbors(edges, "ava", as_of=150.0)] == ["Berlin"]


def test_graph_neighbors_never_reports_two_cities_at_the_same_instant():
    """Правая граница строгая, иначе в момент смены город раздваивается."""
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    edges = graph_add_edge(edges, "ava", "lives_in", "Lisbon", now=200.0)
    assert len(graph_neighbors(edges, "ava", as_of=200.0)) == 1


def test_graph_neighbors_of_an_unknown_subject_is_empty():
    edges = graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
    assert graph_neighbors(edges, "bob") == []


# ---------------------------------------------------------------- fuse_score
def test_fuse_score_is_the_weighted_sum():
    record = rec("m001", "ava lives in Berlin", ts=0.0, importance=0.5)
    assert fuse_score(record, 1.0, now=0.0) == APPROX(
        DEFAULT_WEIGHTS["relevance"] * 1.0
        + DEFAULT_WEIGHTS["importance"] * 0.5
        + DEFAULT_WEIGHTS["recency"] * 1.0)


def test_fuse_score_halves_recency_after_one_halflife():
    record = rec("m001", "ava lives in Berlin", ts=0.0, importance=0.5)
    fresh = fuse_score(record, 1.0, now=0.0)
    aged = fuse_score(record, 1.0, now=HALFLIFE_SECONDS)
    assert fresh - aged == APPROX(DEFAULT_WEIGHTS["recency"] * 0.5)


def test_fuse_score_prefers_the_more_important_record():
    boring = rec("m001", "ava lives in Berlin", importance=0.1)
    critical = rec("m002", "ava lives in Berlin", importance=0.9)
    assert fuse_score(critical, 1.0, now=0.0) > fuse_score(boring, 1.0, now=0.0)


def test_fuse_score_does_not_reward_a_record_from_the_future():
    """Разъехавшиеся часы не должны превращать распад в рост."""
    record = rec("m001", "ava lives in Berlin", ts=1000.0)
    assert fuse_score(record, 1.0, now=0.0) == APPROX(
        fuse_score(record, 1.0, now=1000.0))


# -------------------------------------------------------------- hybrid_search
def test_hybrid_search_never_leaks_another_users_memory():
    records = [rec("m001", "bob requested a refund for invoice 4711",
                   user_id="bob", importance=0.9)]
    assert hybrid_search(records, "refund invoice 4711", "ava", now=0.0) == []


def test_hybrid_search_lets_an_exact_kv_hit_beat_a_similar_text():
    """Совпадение ключа — это не «похоже», это «то самое»."""
    records = [rec("m001", "ava writes terse citation heavy prose"),
               rec("m002", "ava relocated recently", kv={"city": "Lisbon"})]
    hits = hybrid_search(records, "what city does ava live in", "ava", now=0.0)
    assert hits[0][1]["rid"] == "m002"


def test_hybrid_search_fires_the_kv_path_only_on_the_fact_type_word():
    records = [rec("m002", "ava relocated recently", kv={"city": "Lisbon"})]
    with_key = hybrid_search(records, "what city does ava", "ava", now=0.0)
    without_key = hybrid_search(records, "what does ava", "ava", now=0.0)
    assert with_key[0][0] > without_key[0][0]


def test_hybrid_search_ranking_follows_the_weights():
    """Перекос весов в свежесть переставляет выдачу — за этим их и крутят."""
    old_important = rec("m001", "ava lives in Berlin", ts=0.0, importance=1.0)
    fresh_trivial = rec("m002", "ava lives in Berlin", ts=10 * HALFLIFE_SECONDS,
                        importance=0.0)
    records = [old_important, fresh_trivial]
    now = 10 * HALFLIFE_SECONDS
    by_importance = hybrid_search(
        records, "ava lives in Berlin", "ava", now=now,
        weights={"relevance": 0.0, "importance": 1.0, "recency": 0.0})
    by_recency = hybrid_search(
        records, "ava lives in Berlin", "ava", now=now,
        weights={"relevance": 0.0, "importance": 0.0, "recency": 1.0})
    assert by_importance[0][1]["rid"] == "m001"
    assert by_recency[0][1]["rid"] == "m002"


def test_hybrid_search_isolates_scope_before_taking_top_k():
    """Чужая запись не должна съедать место в выдаче."""
    records = [rec("m001", "ava lives in Berlin", scope="session"),
               rec("m002", "ava lives in Berlin", scope="user")]
    hits = hybrid_search(records, "ava lives in Berlin", "ava", now=0.0,
                         top_k=1, scope="user")
    assert [h[1]["rid"] for h in hits] == ["m002"]
