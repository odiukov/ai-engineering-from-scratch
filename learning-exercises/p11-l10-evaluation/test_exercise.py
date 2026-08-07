"""Тесты к уроку «Evaluation и тестирование LLM-приложений». Правь exercise.py."""

import pytest

from exercise import (
    PASSING_SCORE,
    bootstrap_interval,
    compare_runs,
    jaccard_overlap,
    lcs_length,
    normalize_tokens,
    rouge_l,
    wilson_interval,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------- normalize_tokens
def test_normalize_drops_case_and_punctuation():
    assert normalize_tokens("The capital of France is Paris.") == [
        "the", "capital", "of", "france", "is", "paris",
    ]


def test_normalize_keeps_contractions_in_one_piece():
    assert normalize_tokens("I don't know") == ["i", "don't", "know"]


def test_normalize_of_punctuation_only_is_empty():
    assert normalize_tokens("!!! ??? ...") == []


def test_normalize_makes_punctuation_invisible_to_metrics():
    """"Paris." и "Paris" обязаны быть одним токеном, иначе счёт занижен."""
    assert normalize_tokens("Paris.") == normalize_tokens("paris")


# ------------------------------------------------------------- lcs_length
def test_lcs_of_a_subsequence_is_its_length():
    assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2


def test_lcs_allows_gaps_unlike_a_substring():
    assert lcs_length(["a", "x", "b"], ["a", "b"]) == 2


def test_lcs_respects_order():
    assert lcs_length(["a", "b"], ["b", "a"]) == 1


def test_lcs_with_an_empty_side_is_zero():
    assert lcs_length([], ["a", "b"]) == 0


def test_lcs_of_identical_lists_is_the_full_length():
    tokens = ["one", "two", "three", "four"]
    assert lcs_length(tokens, tokens) == 4


# ----------------------------------------------------------------- rouge_l
def test_rouge_l_of_an_identical_answer_is_one():
    assert rouge_l("the cat sat", "the cat sat") == APPROX(1.0)


def test_rouge_l_of_disjoint_answers_is_zero():
    assert rouge_l("the cat sat", "a dog ran") == APPROX(0.0)


def test_rouge_l_is_symmetric():
    """F1 не зависит от того, что назвали эталоном: p и r просто меняются местами."""
    a, b = "the capital of france is paris", "paris is the french capital"
    assert rouge_l(a, b) == APPROX(rouge_l(b, a))


def test_rouge_l_punishes_reordering_while_jaccard_does_not():
    """Вот вся разница между ROUGE-L и мешком слов, в одном тесте."""
    ref, shuffled = "the cat sat", "the sat cat"
    assert jaccard_overlap(ref, shuffled) == APPROX(1.0)
    assert rouge_l(ref, shuffled) < 1.0


def test_rouge_l_ignores_case_and_punctuation():
    assert rouge_l("The capital is Paris.", "the capital is paris") == APPROX(1.0)


def test_rouge_l_of_an_empty_answer_is_zero():
    assert rouge_l("the capital is paris", "") == APPROX(0.0)


def test_rouge_l_can_be_zero_for_two_correct_answers():
    """Честная слабость метрики: оба ответа верны, общих слов нет."""
    assert rouge_l("Paris", "the French capital city") == APPROX(0.0)


# --------------------------------------------------------- jaccard_overlap
def test_jaccard_of_identical_texts_is_one():
    assert jaccard_overlap("machine learning", "learning machine") == APPROX(1.0)


def test_jaccard_counts_the_union_not_the_reference():
    assert jaccard_overlap("the cat", "the dog") == APPROX(1 / 3)


def test_jaccard_ignores_repetitions():
    assert jaccard_overlap("cat cat cat", "cat") == APPROX(1.0)


def test_jaccard_of_two_empty_texts_is_zero():
    assert jaccard_overlap("", "") == APPROX(0.0)


# ---------------------------------------------------------- wilson_interval
def test_wilson_matches_the_worked_example():
    low, high = wilson_interval(45, 50)
    assert (low, high) == pytest.approx((0.7864, 0.9565), abs=1e-4)


def test_wilson_interval_contains_the_observed_rate():
    low, high = wilson_interval(180, 200)
    assert low < 0.9 < high


def test_wilson_interval_narrows_as_the_suite_grows():
    """Ровно та таблица из урока: на 50 кейсах решение принимать нельзя."""
    width = lambda n: wilson_interval(int(n * 0.9), n)[1] - wilson_interval(int(n * 0.9), n)[0]
    assert width(50) > width(200) > width(1000)


def test_wilson_does_not_claim_certainty_on_a_perfect_score():
    """Наивная формула при 20 из 20 даёт ширину ноль — Вильсон так не врёт."""
    low, high = wilson_interval(20, 20)
    assert high == APPROX(1.0)
    assert low < 0.9


def test_wilson_never_leaves_the_zero_one_range():
    low, high = wilson_interval(0, 5)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_of_an_empty_suite_is_a_point_at_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# ------------------------------------------------------- bootstrap_interval
SCORES = [4, 5, 3, 4, 4, 5, 3, 4, 5, 4, 3, 4, 4, 5, 4]


def test_bootstrap_returns_the_exact_mean_in_the_middle():
    low, mean, high = bootstrap_interval(SCORES, seed=0)
    assert mean == APPROX(sum(SCORES) / len(SCORES))
    assert low <= mean <= high


def test_bootstrap_is_reproducible_for_the_same_seed():
    assert bootstrap_interval(SCORES, seed=7) == bootstrap_interval(SCORES, seed=7)


def test_bootstrap_is_wider_for_noisier_scores():
    """Разброс оценок судьи виден именно в ширине интервала, а не в среднем."""
    tight = bootstrap_interval([3, 3, 3, 3, 3, 3, 3, 3, 3, 3], seed=0)
    noisy = bootstrap_interval([1, 5, 1, 5, 1, 5, 1, 5, 1, 5], seed=0)
    assert tight[1] == APPROX(noisy[1])
    assert (noisy[2] - noisy[0]) > (tight[2] - tight[0])


def test_bootstrap_of_constant_scores_has_zero_width():
    """Разброса нет — интервалу неоткуда взяться."""
    low, mean, high = bootstrap_interval([4, 4, 4, 4, 4], seed=0)
    assert (low, mean, high) == pytest.approx((4.0, 4.0, 4.0))


def test_bootstrap_narrows_as_the_suite_grows():
    small = bootstrap_interval(SCORES, seed=3)
    large = bootstrap_interval(SCORES * 20, seed=3)
    assert (large[2] - large[0]) < (small[2] - small[0])


def test_bootstrap_of_an_empty_suite_is_all_zeros():
    assert bootstrap_interval([], seed=0) == (0.0, 0.0, 0.0)


def test_bootstrap_of_a_single_score_is_that_score():
    assert bootstrap_interval([5], seed=0) == pytest.approx((5.0, 5.0, 5.0))


# ------------------------------------------------------------- compare_runs
def test_compare_marks_a_clear_improvement():
    report = compare_runs({"relevance": [4, 4, 4]}, {"relevance": [5, 5, 5]})
    assert report["criteria"]["relevance"]["status"] == "IMPROVED"
    assert report["improvements"] == ["relevance"]


def test_compare_marks_a_clear_regression():
    report = compare_runs({"safety": [5, 5, 5]}, {"safety": [3, 3, 3]})
    assert report["criteria"]["safety"]["status"] == "REGRESSION"
    assert report["overall"]["ship_decision"] == "BLOCK"


def test_noise_under_the_threshold_stays_stable():
    """4.2 против 4.0 — это шум судьи, а не прогресс."""
    report = compare_runs({"c": [4, 4, 4, 4, 4]}, {"c": [4, 4, 4, 4, 5]})
    assert report["criteria"]["c"]["status"] == "STABLE"
    assert report["overall"]["ship_decision"] == "SHIP"


def test_one_regressed_criterion_blocks_an_overall_improvement():
    """Средним по всем критериям провал по safety не увидеть — разбивкой видно."""
    baseline = {"helpfulness": [2, 2, 2, 2], "safety": [5, 5, 5, 5]}
    new = {"helpfulness": [5, 5, 5, 5], "safety": [3, 3, 3, 3]}
    report = compare_runs(baseline, new)
    assert report["overall"]["diff"] > 0
    assert report["regressions"] == ["safety"]
    assert report["overall"]["ship_decision"] == "BLOCK"


def test_pass_rate_counts_scores_of_four_and_above():
    report = compare_runs({"c": [1, 1, 1, 1]}, {"c": [3, 4, 5, 2]})
    assert PASSING_SCORE == 4
    assert report["criteria"]["c"]["new_passing"] == 2


def test_pass_rate_comes_with_a_confidence_interval():
    report = compare_runs({"c": [4] * 10}, {"c": [4] * 8 + [1, 1]})
    low, high = report["criteria"]["c"]["new_pass_ci"]
    assert low < 0.8 < high


def test_criteria_missing_from_one_run_are_skipped():
    report = compare_runs({"a": [4], "b": [4]}, {"a": [4]})
    assert list(report["criteria"]) == ["a"]
    assert report["overall"]["n_criteria"] == 1


def test_comparing_nothing_ships_nothing():
    report = compare_runs({}, {})
    assert report["criteria"] == {}
    assert report["overall"]["n_criteria"] == 0
