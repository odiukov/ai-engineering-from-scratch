"""Тесты к уроку «Оценка LLM: бенчмарки, метрики, ELO». Правь exercise.py."""

import math

import pytest

from exercise import (
    INITIAL_RATING,
    elo_tournament,
    elo_update,
    exact_match,
    expected_score,
    perplexity,
    run_suite,
    summarize,
    token_f1,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CASES = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Hamlet?", "William Shakespeare"),
]

CONCISE = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "Who wrote Hamlet?": "William Shakespeare",
}

VERBOSE = {
    "What is the capital of France?": "Paris is the capital city of France",
    "What is 2 + 2?": "The answer is 4",
    "Who wrote Hamlet?": "Hamlet was written by William Shakespeare",
}


# ------------------------------------------------------------ exact_match
def test_exact_match_on_an_identical_answer():
    assert exact_match("Paris", "Paris") == APPROX(1.0)


def test_exact_match_normalizes_case_and_whitespace():
    """Лишний пробел — не ошибка модели, метрика не должна за него штрафовать."""
    assert exact_match("  paris ", "Paris") == APPROX(1.0)


def test_exact_match_punishes_a_correct_but_verbose_answer():
    """Ключевой урок: правильный по сути ответ получает ноль."""
    assert exact_match("Paris is the capital city of France", "Paris") == APPROX(0.0)


# --------------------------------------------------------------- token_f1
def test_token_f1_on_an_identical_answer():
    assert token_f1("Paris", "Paris") == APPROX(1.0)


def test_token_f1_forgives_verbosity_where_exact_match_does_not():
    """Та же пара строк: exact_match даёт 0.0, token_f1 — заметно больше."""
    verbose = "Paris is the capital city of France"
    assert exact_match(verbose, "Paris") == APPROX(0.0)
    assert token_f1(verbose, "Paris") > 0.0


def test_token_f1_of_a_total_miss_is_zero():
    assert token_f1("London", "Paris") == APPROX(0.0)


def test_token_f1_of_an_empty_prediction_is_zero():
    """Пустая строка — ноль, а не ZeroDivisionError."""
    assert token_f1("", "Paris") == APPROX(0.0)


def test_token_f1_ignores_repeated_words():
    """Работаем с множествами: повтор слова не добавляет баллов."""
    assert token_f1("paris paris paris", "Paris") == APPROX(token_f1("paris", "Paris"))


def test_token_f1_drops_as_padding_grows():
    tight = token_f1("Paris France", "Paris")
    loose = token_f1("Paris is a very large city in France", "Paris")
    assert tight > loose > 0


# ------------------------------------------------------------- perplexity
def test_perplexity_of_a_certain_model_is_one():
    """Вероятность 1 на каждом токене — колебаться не между чем."""
    assert perplexity([0.0, 0.0, 0.0]) == APPROX(1.0)


def test_perplexity_of_a_coin_flip_is_two():
    assert perplexity([math.log(0.5)] * 4) == APPROX(2.0)


def test_perplexity_of_an_empty_sequence_is_infinite():
    assert perplexity([]) == float("inf")


def test_perplexity_falls_as_the_model_gets_more_confident():
    """Меньше — лучше: это главное, что нужно помнить про перплексию."""
    weak = perplexity([math.log(0.1)] * 5)
    strong = perplexity([math.log(0.9)] * 5)
    assert weak > strong > 1.0


# ---------------------------------------------------------- expected_score
def test_expected_score_of_equal_ratings_is_a_coin_flip():
    assert expected_score(1500, 1500) == APPROX(0.5)


def test_expected_score_of_a_four_hundred_point_gap_is_ten_to_one():
    """400 пунктов — это и есть определение шкалы ELO."""
    assert expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-9)


def test_expected_scores_of_both_sides_sum_to_one():
    assert expected_score(1720, 1480) + expected_score(1480, 1720) == APPROX(1.0)


def test_expected_score_sign_is_not_flipped():
    """Ловушка знака: сильнее рейтинг — выше шанс, а не наоборот."""
    assert expected_score(1900, 1500) > 0.5 > expected_score(1500, 1900)


# ------------------------------------------------------------- elo_update
def test_elo_update_moves_the_winner_up_by_half_a_k():
    assert elo_update(1500, 1500, "a") == pytest.approx((1516.0, 1484.0))


def test_elo_update_on_a_tie_between_equals_changes_nothing():
    assert elo_update(1500, 1500, "tie") == pytest.approx((1500.0, 1500.0))


def test_elo_is_zero_sum():
    """Сколько выиграл один — столько проиграл другой, при любом исходе."""
    for outcome in ("a", "b", "tie"):
        new_a, new_b = elo_update(1700, 1420, outcome)
        assert new_a + new_b == pytest.approx(1700 + 1420)


def test_an_upset_moves_the_ratings_more_than_an_expected_win():
    """Неожиданная победа слабого стоит дороже — в этом весь смысл ELO."""
    expected_win = elo_update(1900, 1500, "a")[0] - 1900
    upset_win = elo_update(1500, 1900, "a")[0] - 1500
    assert upset_win > expected_win


def test_a_tie_still_moves_the_favourite_down():
    """Ничья с явным аутсайдером — это потеря рейтинга."""
    new_a, _ = elo_update(1900, 1500, "tie")
    assert new_a < 1900


def test_elo_update_rejects_an_unknown_outcome():
    """Опечатка в исходе обязана падать, а не тихо считаться ничьёй."""
    with pytest.raises(ValueError):
        elo_update(1500, 1500, "draw")


# --------------------------------------------------------- elo_tournament
def test_elo_tournament_seeds_newcomers_at_the_initial_rating():
    ratings = elo_tournament([("gpt", "llama", "tie")])
    assert ratings == pytest.approx({"gpt": INITIAL_RATING, "llama": INITIAL_RATING})


def test_elo_tournament_of_no_matches_is_empty():
    assert elo_tournament([]) == {}


def test_a_model_that_always_wins_climbs_above_its_rival():
    matches = [("gpt", "llama", "a")] * 10
    ratings = elo_tournament(matches)
    assert ratings["gpt"] > INITIAL_RATING > ratings["llama"]


def test_elo_tournament_conserves_the_total_rating():
    matches = [("a", "b", "a"), ("b", "c", "tie"), ("a", "c", "b"), ("c", "a", "a")]
    ratings = elo_tournament(matches)
    assert sum(ratings.values()) == pytest.approx(3 * INITIAL_RATING)


# --------------------------------------------------------------- run_suite
def test_run_suite_records_prediction_and_scores_per_case():
    results = run_suite([("2+2?", "4")], lambda q: "4", {"em": exact_match})
    assert results == [
        {"input": "2+2?", "expected": "4", "prediction": "4", "scores": {"em": 1.0}}
    ]


def test_run_suite_calls_the_model_once_per_case():
    """Метрики обязаны мерить ОДИН ответ, иначе их нельзя сравнивать между собой."""
    calls = []

    def model(q):
        calls.append(q)
        return "4"

    run_suite(CASES, model, {"em": exact_match, "f1": token_f1})
    assert calls == [c[0] for c in CASES]


# --------------------------------------------------------------- summarize
def test_summarize_computes_mean_and_spread():
    stats = summarize([{"scores": {"em": 1.0}}, {"scores": {"em": 0.0}}])["em"]
    assert (stats["mean"], stats["std"], stats["min"], stats["max"], stats["n"]) == (
        pytest.approx(0.5), pytest.approx(0.5), 0.0, 1.0, 2
    )


def test_summarize_takes_the_average_of_two_middles_for_an_even_count():
    """Ловушка медианы: при чётном n элемента посередине нет."""
    rows = [{"scores": {"s": v}} for v in (0.0, 0.2, 0.8, 1.0)]
    assert summarize(rows)["s"]["median"] == APPROX(0.5)


def test_summarize_of_an_empty_run_is_empty():
    assert summarize([]) == {}


def test_pass_rate_counts_scores_at_the_threshold():
    """Порог включительный: ровно 0.8 — это зачёт."""
    rows = [{"scores": {"s": v}} for v in (0.79, 0.8, 0.81)]
    assert summarize(rows)["s"]["pass_rate"] == pytest.approx(2 / 3)


def test_pass_rate_sees_a_split_that_the_mean_hides():
    """Две выборки с одним средним, но разной надёжностью."""
    steady = [{"scores": {"s": 0.5}} for _ in range(4)]
    split = [{"scores": {"s": v}} for v in (1.0, 1.0, 0.0, 0.0)]
    assert summarize(steady)["s"]["mean"] == APPROX(summarize(split)["s"]["mean"])
    assert summarize(steady)["s"]["pass_rate"] < summarize(split)["s"]["pass_rate"]


# ------------------------------------------------------------- всё вместе
def test_the_metric_decides_which_model_looks_better():
    """Сквозной вывод урока: одна и та же пара моделей меняется местами.

    Лаконичная модель выигрывает по exact_match вчистую, многословная
    догоняет по token_f1 — потому что метрики меряют разное.
    """
    scorers = {"em": exact_match, "f1": token_f1}
    concise = summarize(run_suite(CASES, CONCISE.get, scorers))
    verbose = summarize(run_suite(CASES, VERBOSE.get, scorers))

    assert concise["em"]["mean"] == APPROX(1.0)
    assert verbose["em"]["mean"] == APPROX(0.0)
    assert verbose["f1"]["mean"] > verbose["em"]["mean"]
