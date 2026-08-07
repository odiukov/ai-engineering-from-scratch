"""Тесты к уроку «Оценка LLM: RAGAS, DeepEval, G-Eval». Правь exercise.py."""

import re

import pytest

from exercise import (
    aggregate_scores,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    parse_judge_score,
    spearman_rho,
    split_claims,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CONTEXT = "Apple released the first iPhone on June 29, 2007."


def substring_judge(claim, context):
    """Детерминированная заглушка судьи: доля слов claim, встреченных в контексте."""
    words = re.findall(r"[a-z0-9]+", claim.lower())
    if not words:
        return 0.0
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    return sum(1 for w in words if w in ctx) / len(words)


def jaccard(a, b):
    """Заглушка вместо encoder-а: пересечение токенов делить на объединение."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ------------------------------------------------------------- split_claims
def test_split_claims_splits_on_sentence_end():
    text = "The iPhone launched in 2007. Apple is in Cupertino."
    assert split_claims(text) == [
        "The iPhone launched in 2007.",
        "Apple is in Cupertino.",
    ]


def test_split_claims_of_blank_text_is_empty():
    assert split_claims("   ") == []


def test_split_claims_does_not_break_on_a_comma_inside_a_date():
    """June 29, 2007 — одно утверждение, а не два."""
    assert split_claims("It shipped on June 29, 2007.") == [
        "It shipped on June 29, 2007."
    ]


# ------------------------------------------------------------- faithfulness
def test_faithfulness_is_one_when_every_claim_is_supported():
    assert faithfulness("A. B.", CONTEXT, lambda c, x: 1.0) == APPROX(1.0)


def test_faithfulness_is_the_fraction_of_supported_claims():
    """Один claim из двух подтверждён — ровно половина."""
    calls = iter([1.0, 0.0])
    assert faithfulness("A. B.", CONTEXT, lambda c, x: next(calls)) == APPROX(0.5)


def test_faithfulness_of_empty_answer_is_zero_not_a_crash():
    assert faithfulness("", CONTEXT, substring_judge) == APPROX(0.0)


def test_faithfulness_drops_when_a_claim_is_hallucinated():
    """Второе предложение про контекст не сказано — оценка обязана упасть."""
    grounded = "Apple released the first iPhone on June 29, 2007."
    with_hallucination = grounded + " The moon landing happened in 1969."
    assert faithfulness(with_hallucination, CONTEXT, substring_judge) < faithfulness(
        grounded, CONTEXT, substring_judge
    )


def test_faithfulness_passes_claim_and_context_to_the_judge_in_that_order():
    seen = []

    def spy(claim, context):
        seen.append((claim, context))
        return 1.0

    faithfulness("A. B.", CONTEXT, spy)
    assert seen == [("A.", CONTEXT), ("B.", CONTEXT)]


# ---------------------------------------------------------- answer_relevance
def test_answer_relevance_is_high_when_generated_questions_match_the_real_one():
    q = "When was the first iPhone released?"
    assert answer_relevance(q, "June 29, 2007.", lambda a: [q], jaccard) == APPROX(1.0)


def test_answer_relevance_collapses_for_an_off_topic_answer():
    q = "When was the first iPhone released?"
    on_topic = answer_relevance(
        q, "2007", lambda a: ["When did the iPhone come out?"], jaccard
    )
    off_topic = answer_relevance(
        q, "Cupertino", lambda a: ["Where is Apple headquartered?"], jaccard
    )
    assert off_topic < on_topic


def test_answer_relevance_without_generated_questions_is_zero():
    assert answer_relevance("q", "a", lambda a: [], jaccard) == APPROX(0.0)


def test_answer_relevance_ignores_blank_lines_from_the_generator():
    """Лишние переводы строки от LLM не должны тянуть среднее вниз."""
    q = "When was the first iPhone released?"
    clean = answer_relevance(q, "a", lambda a: [q], jaccard)
    noisy = answer_relevance(q, "a", lambda a: [q, "", "   "], jaccard)
    assert noisy == APPROX(clean)


# --------------------------------------------------------- context_precision
def test_context_precision_counts_only_retrieved_chunks():
    assert context_precision(["a", "b"], ["a"]) == APPROX(0.5)


def test_context_precision_of_empty_retrieval_is_zero():
    assert context_precision([], ["a"]) == APPROX(0.0)


def test_context_precision_falls_when_top_k_is_padded_with_junk():
    """Те же попадания, но выдали больше мусора — precision обязана упасть."""
    assert context_precision(["a", "junk", "junk2"], ["a"]) < context_precision(
        ["a"], ["a"]
    )


# ------------------------------------------------------------ context_recall
def test_context_recall_is_the_fraction_of_gold_claims_covered():
    gold = ["iPhone released 2007", "Apple in Cupertino"]
    assert context_recall(gold, [CONTEXT], substring_judge) == APPROX(0.5)


def test_context_recall_without_gold_claims_is_zero():
    assert context_recall([], [CONTEXT], substring_judge) == APPROX(0.0)


def test_context_recall_grows_when_more_chunks_are_retrieved():
    gold = ["iPhone released 2007", "Android launched 2008"]
    narrow = context_recall(gold, [CONTEXT], substring_judge)
    wide = context_recall(gold, [CONTEXT, "Android launched in 2008."], substring_judge)
    assert wide > narrow


def test_recall_and_precision_pull_in_opposite_directions():
    """Выдать всё подряд: recall вырастет, precision просядет — это и есть trade-off."""
    gold = ["iPhone released 2007"]
    junk = "Bananas are yellow."
    assert context_recall(gold, [CONTEXT, junk], substring_judge) == APPROX(
        context_recall(gold, [CONTEXT], substring_judge)
    )
    assert context_precision([CONTEXT, junk], [CONTEXT]) < context_precision(
        [CONTEXT], [CONTEXT]
    )


# --------------------------------------------------------- parse_judge_score
def test_parse_judge_score_reads_plain_json():
    assert parse_judge_score('{"score": 0.8}') == APPROX(0.8)


def test_parse_judge_score_survives_text_and_code_fences_around_json():
    raw = 'Sure, here you go:\n```json\n{"score": 1, "reason": "ok"}\n```'
    assert parse_judge_score(raw) == APPROX(1.0)


def test_parse_judge_score_returns_none_on_broken_json_not_zero():
    """Ноль означал бы «плохой ответ». Провал парсинга — это None."""
    assert parse_judge_score("score is high") is None
    assert parse_judge_score('{"score": ') is None


def test_parse_judge_score_rejects_scores_outside_the_unit_range():
    assert parse_judge_score('{"score": 1.5}') is None
    assert parse_judge_score('{"score": -0.2}') is None


def test_parse_judge_score_rejects_a_boolean_score():
    """True — подкласс int, без явной проверки он стал бы 1.0."""
    assert parse_judge_score('{"score": true}') is None


# --------------------------------------------------------- aggregate_scores
def test_aggregate_scores_averages_valid_scores():
    report = aggregate_scores([1.0, 0.0])
    assert report["mean"] == APPROX(0.5)
    assert report["valid"] == 2
    assert report["failed"] == 0


def test_aggregate_scores_excludes_none_from_the_mean_and_counts_it():
    report = aggregate_scores([1.0, None])
    assert report["mean"] == APPROX(1.0)
    assert report["valid"] == 1
    assert report["failed"] == 1


def test_aggregate_scores_of_all_failures_reports_zero_valid():
    report = aggregate_scores([None, None])
    assert report == {"mean": 0.0, "bottom_mean": 0.0, "valid": 0, "failed": 2}


def test_bottom_mean_exposes_catastrophes_that_the_mean_hides():
    """Девять отличных ответов и один провальный: среднее высокое, хвост нулевой."""
    report = aggregate_scores([1.0] * 9 + [0.0], q=0.1)
    assert report["mean"] > 0.85
    assert report["bottom_mean"] == APPROX(0.0)


def test_bottom_mean_takes_at_least_one_example():
    """На трёх примерах 10% округляются вверх до одного, а не до нуля."""
    assert aggregate_scores([0.2, 0.9, 1.0], q=0.1)["bottom_mean"] == APPROX(0.2)


# ------------------------------------------------------------- spearman_rho
def test_spearman_is_one_for_a_monotone_but_non_linear_pair():
    assert spearman_rho([1, 2, 3, 4], [1, 4, 9, 100]) == pytest.approx(1.0, abs=1e-9)


def test_spearman_is_minus_one_for_a_reversed_ranking():
    assert spearman_rho([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_averages_ranks_of_tied_values():
    """У связок общий средний ранг, поэтому порядок равных элементов не влияет."""
    a = spearman_rho([1, 1, 2, 3], [10, 20, 30, 40])
    b = spearman_rho([1, 1, 2, 3], [20, 10, 30, 40])
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(0.9486832980505138, abs=1e-9)


def test_spearman_of_a_constant_judge_is_zero():
    """Судья, который всем ставит одно и то же, не коррелирует ни с чем."""
    assert spearman_rho([1, 1, 1], [1, 2, 3]) == APPROX(0.0)


def test_spearman_rejects_lists_of_different_length():
    with pytest.raises(ValueError):
        spearman_rho([1, 2, 3], [1, 2])


def test_a_noisy_judge_scores_below_the_calibration_threshold():
    """Судья без калибровки: rho ниже 0.7 — числу верить нельзя."""
    human = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noisy = [0.6, 0.0, 1.0, 0.2, 0.4, 0.8]
    assert spearman_rho(noisy, human) < 0.7
