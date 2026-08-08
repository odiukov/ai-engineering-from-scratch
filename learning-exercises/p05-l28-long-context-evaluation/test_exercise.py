"""Тесты к уроку «Оценка длинного контекста: NIAH, RULER, LongBench». Правь exercise.py."""

import re

import pytest

from exercise import (
    build_haystack,
    effective_length,
    insert_needles,
    niah_grid,
    pass_rates,
    score_multi_needle,
    score_needle,
    trace_variables,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

FILLER = "the quick brown fox jumps over the lazy dog near a quiet river bank"
NEEDLES = [
    "the magic word is pineapple",
    "the magic word is compass",
    "the magic word is whisper",
]


def first_needle_model(context, question):
    """Заглушка модели с насыщением внимания: помнит только первую иголку."""
    found = re.findall(r"magic word is (\w+)", context)
    return f"The magic word is {found[0]}." if found else "no answer"


def capacity_trial(depth, length):
    """Заглушка прогона: модель держит контекст только до 4000 токенов."""
    return 1 if length <= 4000 else 0


def lost_in_the_middle_trial(depth, length):
    """Заглушка прогона: начало и конец видны, середина проваливается."""
    return 1 if depth <= 0.25 or depth >= 0.75 else 0


# ------------------------------------------------------------ build_haystack
def test_build_haystack_puts_the_needle_first_at_depth_zero():
    assert build_haystack("a b c d", "N", 0.0, 3) == "N a b"


def test_build_haystack_puts_the_needle_last_at_depth_one():
    assert build_haystack("a b c d", "N", 1.0, 3) == "a b N"


def test_build_haystack_length_counts_the_needle_in():
    """Иголка занимает место filler-а, а не добавляется сверху."""
    text = build_haystack(FILLER, "the magic word is pineapple", 0.5, 100)
    assert len(text.split()) == 100


def test_build_haystack_repeats_a_short_filler():
    text = build_haystack("a b", "N", 0.5, 9)
    assert len(text.split()) == 9
    assert "N" in text.split()


def test_build_haystack_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        build_haystack("a b", "N", 1.5, 10)
    with pytest.raises(ValueError):
        build_haystack("a b", "N", 0.5, 0)
    with pytest.raises(ValueError):
        build_haystack("   ", "N", 0.5, 10)


def test_build_haystack_rejects_a_needle_larger_than_the_exact_budget():
    with pytest.raises(ValueError, match="needle does not fit"):
        build_haystack("a b", "too many needle tokens", 0.5, 3)


# ------------------------------------------------------------ insert_needles
def test_insert_needles_places_each_needle_at_its_depth():
    assert insert_needles("a b c d", ["X", "Y"], [0.0, 0.5]) == "X a b Y c d"


def test_insert_needles_starts_from_the_deepest_so_positions_do_not_shift():
    """Мелкая иголка удлиняет текст — глубокая должна встать до неё."""
    words = insert_needles("a b c d e f g h", ["X", "Y"], [0.25, 0.75]).split()
    assert words.index("X") == 2
    assert words.index("Y") == 7


def test_insert_needles_keeps_every_filler_token():
    text = insert_needles("a b c d", ["X"], [0.5])
    assert [w for w in text.split() if w != "X"] == ["a", "b", "c", "d"]


def test_insert_needles_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        insert_needles("a b c", ["X", "Y"], [0.5])


# --------------------------------------------------------------- score_needle
def test_score_needle_matches_regardless_of_case():
    assert score_needle("ctx", "q", "pineapple", lambda c, q: "It is Pineapple.") == 1


def test_score_needle_is_zero_when_the_model_misses():
    assert score_needle("ctx", "q", "pineapple", lambda c, q: "no answer") == 0


def test_score_needle_hands_context_and_question_to_the_model():
    seen = []

    def spy(context, question):
        seen.append((context, question))
        return "pineapple"

    score_needle("haystack text", "What is the magic word?", "pineapple", spy)
    assert seen == [("haystack text", "What is the magic word?")]


# --------------------------------------------------------- score_multi_needle
def test_score_multi_needle_is_the_fraction_recalled():
    assert score_multi_needle("c", "q", ["a", "b"], lambda c, q: "a") == APPROX(0.5)


def test_score_multi_needle_without_expectations_is_zero():
    assert score_multi_needle("c", "q", [], lambda c, q: "anything") == APPROX(0.0)


def test_score_multi_needle_matches_regardless_of_case():
    assert score_multi_needle("c", "q", ["Pineapple"], lambda c, q: "pineapple") == APPROX(1.0)


def test_single_needle_success_does_not_predict_multi_needle_success():
    """Одна иголка найдена — а из трёх та же модель достаёт только одну."""
    haystack = insert_needles(FILLER, NEEDLES, [0.2, 0.5, 0.8])
    single = score_needle(haystack, "q", "pineapple", first_needle_model)
    multi = score_multi_needle(
        haystack, "q", ["pineapple", "compass", "whisper"], first_needle_model
    )
    assert single == 1
    assert multi == APPROX(1 / 3)


# ------------------------------------------------------------------ niah_grid
def test_niah_grid_covers_every_length_and_depth_pair():
    grid = niah_grid([0.0, 0.5], [100, 200], lambda d, n: 1)
    assert set(grid) == {(100, 0.0), (100, 0.5), (200, 0.0), (200, 0.5)}


def test_niah_grid_passes_depth_first_and_length_second():
    grid = niah_grid([0.25], [1000], lambda d, n: (d, n))
    assert grid[(1000, 0.25)] == (0.25, 1000)


def test_niah_grid_of_empty_axes_is_empty():
    assert niah_grid([], [100], lambda d, n: 1) == {}


# ------------------------------------------------------------------ pass_rates
def test_pass_rates_by_length_average_over_depths():
    grid = {(100, 0.5): 1, (100, 0.9): 0, (200, 0.5): 1, (200, 0.9): 1}
    assert pass_rates(grid, "length") == pytest.approx({100: 0.5, 200: 1.0})


def test_pass_rates_by_depth_average_over_lengths():
    grid = {(100, 0.5): 1, (200, 0.5): 0, (100, 0.9): 1, (200, 0.9): 1}
    assert pass_rates(grid, "depth") == pytest.approx({0.5: 0.5, 0.9: 1.0})


def test_pass_rates_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        pass_rates({(100, 0.5): 1}, "temperature")


def test_retrieval_degrades_in_the_middle_of_the_context():
    """Lost in the middle: по краям находит, в середине теряет."""
    grid = niah_grid([0.0, 0.5, 1.0], [1000, 4000], lost_in_the_middle_trial)
    by_depth = pass_rates(grid, "depth")
    assert by_depth[0.5] < by_depth[0.0]
    assert by_depth[0.5] < by_depth[1.0]


# ------------------------------------------------------------ effective_length
def test_effective_length_stops_at_the_first_drop():
    assert effective_length({1000: 1.0, 4000: 1.0, 16000: 0.4}) == 4000


def test_effective_length_is_zero_when_even_the_shortest_fails():
    assert effective_length({1000: 0.5}) == 0


def test_effective_length_ignores_recovery_after_a_drop():
    """Провал на 4k не отменяется случайной удачей на 16k."""
    assert effective_length({1000: 1.0, 4000: 0.2, 16000: 1.0}) == 1000


def test_effective_length_is_smaller_than_the_advertised_window():
    """Заявленное окно и реально рабочее — разные числа."""
    lengths = [1000, 4000, 16000]
    grid = niah_grid([0.25, 0.5, 0.75], lengths, capacity_trial)
    assert effective_length(pass_rates(grid, "length")) < max(lengths)


# --------------------------------------------------------------- trace_variables
def test_trace_variables_resolves_a_three_hop_chain():
    text = "X1 = 42. X2 = X1 + 10. X3 = X2 * 2."
    assert trace_variables(text) == {"X1": 42, "X2": 52, "X3": 104}


def test_trace_variables_ignores_filler_between_assignments():
    """Хопы раскиданы по haystack-у — filler не должен ничего сломать."""
    text = build_haystack(FILLER, "X1 = 7", 0.2, 60)
    text = text + " " + FILLER + " X2 = X1 - 3 " + FILLER
    assert trace_variables(text) == {"X1": 7, "X2": 4}


def test_trace_variables_keeps_the_last_assignment():
    assert trace_variables("X1 = 1. X1 = 9. X2 = X1 * 2.") == {"X1": 9, "X2": 18}


def test_trace_variables_rejects_a_forward_reference():
    """Ссылка на ещё не заданную переменную — сломанная цепочка, а не ноль."""
    with pytest.raises(ValueError):
        trace_variables("X2 = X1 + 10.")


def test_trace_variables_without_assignments_is_empty():
    assert trace_variables("the quick brown fox jumps over the lazy dog") == {}
