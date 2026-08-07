"""Тесты к уроку «Мультимодальный RAG и кросс-модальный поиск». Правь exercise.py."""

import pytest

from exercise import (
    MODALITY_KEYWORDS,
    agentic_retrieve,
    grounded_answer,
    min_max_normalize,
    moe_gate,
    needs_another_hop,
    recall_at_k,
    score_fusion,
    top_k,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def spy_retriever(per_hop):
    """Ретривер, отдающий заранее заданный результат на каждый заход.

    Возвращает пару (функция, список полученных запросов).
    """
    seen = []

    def fn(query):
        seen.append(query)
        return per_hop[min(len(seen) - 1, len(per_hop) - 1)]

    return fn, seen


def spy_reformulator():
    seen = []

    def fn(query):
        seen.append(query)
        return query + "!"

    return fn, seen


# ------------------------------------------------------- min_max_normalize
def test_min_max_normalize_stretches_scores_to_the_unit_range():
    assert min_max_normalize({"a": 2.0, "b": 4.0, "c": 3.0}) == pytest.approx(
        {"a": 0.0, "b": 1.0, "c": 0.5}, abs=1e-9
    )


def test_min_max_normalize_when_the_retriever_cannot_tell_candidates_apart():
    """Знаменатель ноль: договорённость — все получают 1.0, а не ZeroDivisionError."""
    assert min_max_normalize({"a": 7.0, "b": 7.0}) == {"a": 1.0, "b": 1.0}


def test_min_max_normalize_of_an_empty_result_is_empty():
    assert min_max_normalize({}) == {}


def test_min_max_normalize_erases_the_original_scale():
    """BM25 в десятках и косинус в [-1, 1] после нормировки становятся сравнимыми."""
    small = min_max_normalize({"a": 0.1, "b": 0.3, "c": 0.2})
    large = min_max_normalize({"a": 10.0, "b": 30.0, "c": 20.0})
    assert small == pytest.approx(large, abs=1e-9)


# --------------------------------------------------------------- score_fusion
def test_score_fusion_worked_example():
    fused = score_fusion([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], [0.7, 0.3])
    assert fused == pytest.approx({"a": 0.7, "b": 0.3}, abs=1e-9)


def test_score_fusion_weights_decide_the_winner():
    maps = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    assert top_k(score_fusion(maps, [0.7, 0.3]), 1)[0][0] == "a"
    assert top_k(score_fusion(maps, [0.3, 0.7]), 1)[0][0] == "b"


def test_a_document_missing_from_one_retriever_still_scores():
    """У каждой модальности свой охват, требовать полного пересечения нельзя."""
    fused = score_fusion([{"a": 1.0, "b": 0.0}, {"a": 0.0, "c": 1.0}], [0.5, 0.5])
    assert set(fused) == {"a", "b", "c"}
    assert fused["b"] == APPROX(0.0)
    assert fused["c"] == APPROX(0.5)


def test_score_fusion_is_immune_to_a_retrievers_score_scale():
    """Смысловое требование урока: без нормировки ретривер с крупной шкалой съедает веса."""
    text = {"a": 1.0, "b": 0.5, "c": 0.0}
    image = {"a": 0.0, "b": 0.5, "c": 1.0}
    blown_up = {k: v * 100.0 for k, v in image.items()}
    assert score_fusion([text, image], [0.5, 0.5]) == pytest.approx(
        score_fusion([text, blown_up], [0.5, 0.5]), abs=1e-9
    )


def test_score_fusion_does_not_mutate_the_retriever_results():
    text = {"a": 12.0, "b": 3.0}
    score_fusion([text], [1.0])
    assert text == {"a": 12.0, "b": 3.0}


# --------------------------------------------------------------------- top_k
def test_top_k_returns_the_best_first():
    assert top_k({"a": 0.2, "b": 0.9, "c": 0.5}, 2) == [("b", 0.9), ("c", 0.5)]


def test_top_k_does_not_choke_on_a_short_corpus():
    assert len(top_k({"a": 0.2}, 5)) == 1


def test_ties_break_by_doc_id_so_ab_tests_are_reproducible():
    assert [d for d, _ in top_k({"c": 0.5, "a": 0.5, "b": 0.5}, 3)] == ["a", "b", "c"]


# ------------------------------------------------------------------ moe_gate
def test_moe_gate_weights_always_sum_to_one():
    for query in ("", "quiet vegan brunch with natural light", "menu price"):
        assert sum(moe_gate(query).values()) == pytest.approx(1.0, abs=1e-9)


def test_moe_gate_without_signal_is_uniform():
    assert moe_gate("") == pytest.approx(
        {"text": 1 / 3, "image": 1 / 3, "audio": 1 / 3}, abs=1e-9
    )


def test_an_acoustic_query_routes_to_the_audio_expert():
    weights = moe_gate("quiet noisy music")
    assert max(weights, key=weights.get) == "audio"


def test_a_price_query_routes_to_the_text_expert():
    """Ровно то, чего фиксированные веса score fusion не умеют."""
    weights = moe_gate("menu price")
    assert max(weights, key=weights.get) == "text"


def test_every_modality_is_reachable_through_its_own_keywords():
    """Слова из MODALITY_KEYWORDS обязаны реально доводить до своего эксперта."""
    assert set(MODALITY_KEYWORDS) == {"text", "image", "audio"}
    for modality, keywords in MODALITY_KEYWORDS.items():
        weights = moe_gate(" ".join(keywords))
        assert max(weights, key=weights.get) == modality


# --------------------------------------------------------------- recall_at_k
def test_recall_at_k_counts_hits_inside_the_cutoff():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == APPROX(0.5)


def test_recall_reaches_one_when_everything_relevant_is_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == APPROX(1.0)


def test_recall_divides_by_the_relevant_set_not_by_k():
    """Иначе recall@10 при двух релевантных документах никогда не превысит 0.2."""
    assert recall_at_k(["a", "c", "x", "y", "z"], {"a", "c"}, 10) == APPROX(1.0)


def test_recall_without_relevant_documents_is_zero_not_a_crash():
    assert recall_at_k(["a"], set(), 1) == APPROX(0.0)


def test_recall_never_decreases_as_k_grows():
    ranked = ["x", "a", "y", "c"]
    values = [recall_at_k(ranked, {"a", "c"}, k) for k in range(1, 5)]
    assert values == sorted(values)


# ----------------------------------------------------------- grounded_answer
def test_grounded_answer_tags_an_image_source():
    assert grounded_answer([("r1", 0.9)], {"r1": ("image", "airy hall")}) == (
        "airy hall [img 1]"
    )


def test_grounded_answer_numbers_sources_from_one_in_ranking_order():
    ranked = [("r2", 0.9), ("r1", 0.4)]
    evidence = {"r1": ("text", "great reviews"), "r2": ("audio", "38 dB")}
    assert grounded_answer(ranked, evidence) == "38 dB [audio 1]\ngreat reviews [text 2]"


def test_grounded_answer_refuses_to_invent_an_answer_without_evidence():
    """Это и есть смысл grounding: без источников — не сочинять."""
    assert grounded_answer([], {"r1": ("text", "whatever")}) == "no evidence"


def test_an_untaggable_modality_is_an_error_not_a_bare_quote():
    with pytest.raises(KeyError):
        grounded_answer([("r1", 0.9)], {"r1": ("video", "a clip")})


# --------------------------------------------------------- needs_another_hop
def test_a_confident_leader_ends_the_loop():
    assert needs_another_hop({"a": 0.9, "b": 0.2}) is False


def test_a_weak_leader_triggers_another_hop():
    assert needs_another_hop({"a": 0.5, "b": 0.2}) is True


def test_two_indistinguishable_candidates_trigger_another_hop():
    """Переформулировка тут добавляет больше, чем ещё десять кандидатов."""
    assert needs_another_hop({"a": 0.9, "b": 0.88}) is True


def test_an_empty_result_always_triggers_another_hop():
    assert needs_another_hop({}) is True


def test_a_single_confident_candidate_needs_no_margin():
    assert needs_another_hop({"a": 0.95}) is False


# --------------------------------------------------------- agentic_retrieve
def test_a_confident_first_hop_skips_reformulation():
    retrieve, asked = spy_retriever([[{"a": 1.0, "b": 0.0}]])
    reformulate, rewritten = spy_reformulator()
    fused, hops = agentic_retrieve("quiet brunch", retrieve, reformulate, [1.0])
    assert hops == 1
    assert asked == ["quiet brunch"]
    assert rewritten == []
    assert fused["a"] == APPROX(1.0)


def test_a_hopeless_corpus_stops_at_max_hops():
    """Без потолка плохой корпус загоняет агента в бесконечный цикл."""
    retrieve, asked = spy_retriever([[{"a": 1.0, "b": 0.0}]])
    reformulate, _ = spy_reformulator()
    _, hops = agentic_retrieve("q", retrieve, reformulate, [0.5], max_hops=3)
    assert hops == 3
    assert len(asked) == 3


def test_each_extra_hop_uses_a_reformulated_query():
    retrieve, asked = spy_retriever([[{"a": 1.0, "b": 0.0}]])
    reformulate, rewritten = spy_reformulator()
    agentic_retrieve("q", retrieve, reformulate, [0.5], max_hops=3)
    assert asked == ["q", "q!", "q!!"]
    assert len(rewritten) == 2


def test_the_result_comes_from_the_last_hop():
    retrieve, _ = spy_retriever([
        [{"a": 1.0, "b": 1.0}],          # заход 1: ретривер не выбрал лидера
        [{"z": 1.0, "b": 0.0}],          # заход 2: уверенный лидер
    ])
    reformulate, _ = spy_reformulator()
    fused, hops = agentic_retrieve("q", retrieve, reformulate, [1.0], max_hops=4)
    assert hops == 2
    assert set(fused) == {"z", "b"}
