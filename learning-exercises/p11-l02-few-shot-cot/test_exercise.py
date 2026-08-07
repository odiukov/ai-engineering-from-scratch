"""Тесты к уроку «Few-shot, chain-of-thought и голосование». Правь exercise.py."""

from collections import Counter

import pytest

from exercise import (
    COT_SYSTEM,
    build_cot_prompt,
    extract_answer,
    format_example,
    majority_vote,
    select_diverse_examples,
    select_examples,
    self_consistency,
    tree_of_thought,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

EX_SUM = {"question": "2+2?", "reasoning": "2 and 2 make 4.", "answer": "4"}
EX_APPLES = {
    "question": "How much do apples cost in total?",
    "reasoning": "Three apples at 2 each is 6.",
    "answer": "6",
}
EX_TRAIN = {
    "question": "A train travels 120 km in 2 hours, what is its speed?",
    "reasoning": "120 divided by 2 is 60.",
    "answer": "60",
}


# ----------------------------------------------------------- format_example
def test_format_example_puts_reasoning_before_the_answer():
    assert format_example(EX_SUM) == "Q: 2+2?\nA: 2 and 2 make 4. The answer is 4."


def test_format_example_without_reasoning_drops_the_chain():
    plain = format_example(EX_SUM, with_reasoning=False)
    assert plain == "Q: 2+2?\nA: The answer is 4."
    assert "2 and 2 make 4" not in plain


def test_format_example_output_is_parseable_by_extract_answer():
    """Формат примера и парсер ответа обязаны сходиться, иначе голосование пусто."""
    assert extract_answer(format_example(EX_SUM)) == APPROX(4.0)


# --------------------------------------------------------- build_cot_prompt
def test_build_cot_prompt_uses_the_shared_system_message():
    system, _ = build_cot_prompt("5+5?", [EX_SUM])
    assert system == COT_SYSTEM


def test_build_cot_prompt_ends_with_an_open_answer_line():
    """Оборванная 'A:' — приглашение модели продолжить, а не начать заново."""
    _, user = build_cot_prompt("5+5?", [EX_SUM])
    assert user.endswith("Q: 5+5?\nA:")


def test_build_cot_prompt_without_examples_is_zero_shot():
    _, user = build_cot_prompt("5+5?", [])
    assert user == "Q: 5+5?\nA:"


def test_build_cot_prompt_respects_num_examples():
    _, user = build_cot_prompt("5+5?", [EX_SUM, EX_APPLES, EX_TRAIN], num_examples=2)
    assert user.count("Q: ") == 3  # два примера плюс сам вопрос
    assert "120 km" not in user


# ------------------------------------------------------------ extract_answer
def test_extract_answer_reads_a_plain_number():
    assert extract_answer("Janet makes 9 * 2 = 18. The answer is 18.") == APPROX(18.0)


def test_extract_answer_reads_a_negative_number():
    assert extract_answer("The answer is -3.5") == APPROX(-3.5)


def test_extract_answer_returns_none_when_there_is_no_answer_line():
    assert extract_answer("I have no idea what to do here") is None


def test_extract_answer_takes_the_last_occurrence():
    """Модель рассуждает сверху вниз: вывод внизу, черновик выше."""
    text = "First I thought the answer is 10. Rechecking: the answer is 12."
    assert extract_answer(text) == APPROX(12.0)


def test_extract_answer_strips_thousand_separators():
    """Без этого '1,200' превращается в 1.0 и тихо ломает голосование."""
    assert extract_answer("The answer is 1,200") == APPROX(1200.0)


# ----------------------------------------------------------- select_examples
def test_select_examples_prefers_the_lexically_closest_question():
    picked = select_examples("How much do apples cost?", [EX_SUM, EX_TRAIN, EX_APPLES], 1)
    assert picked == [EX_APPLES]


def test_select_examples_returns_exactly_k_items():
    assert len(select_examples("anything", [EX_SUM, EX_TRAIN, EX_APPLES], 2)) == 2


def test_select_examples_keeps_input_order_on_ties():
    """Запрос не пересекается ни с чем — порядок обязан остаться исходным."""
    picked = select_examples("zzz qqq", [EX_SUM, EX_TRAIN, EX_APPLES], 3)
    assert picked == [EX_SUM, EX_TRAIN, EX_APPLES]


# --------------------------------------------------- select_diverse_examples
def test_select_diverse_examples_covers_distinct_answers_first():
    a = {"question": "q1", "reasoning": "r", "answer": "1"}
    b = {"question": "q2", "reasoning": "r", "answer": "1"}
    c = {"question": "q3", "reasoning": "r", "answer": "2"}
    assert select_diverse_examples([a, b, c], 2) == [a, c]


def test_select_diverse_examples_falls_back_to_duplicates_when_labels_run_out():
    a = {"question": "q1", "reasoning": "r", "answer": "1"}
    b = {"question": "q2", "reasoning": "r", "answer": "1"}
    assert select_diverse_examples([a, b], 2) == [a, b]


def test_select_diverse_examples_never_exceeds_k():
    a = {"question": "q1", "reasoning": "r", "answer": "1"}
    b = {"question": "q2", "reasoning": "r", "answer": "2"}
    c = {"question": "q3", "reasoning": "r", "answer": "3"}
    assert len(select_diverse_examples([a, b, c], 2)) == 2


# ------------------------------------------------------------- majority_vote
def test_majority_vote_picks_the_most_common_answer():
    answer, confidence = majority_vote([24.0, 24.0, 27.0])
    assert answer == APPROX(24.0)
    assert confidence == APPROX(2 / 3)


def test_majority_vote_of_nothing_is_none():
    assert majority_vote([]) == (None, 0.0)


def test_majority_vote_breaks_ties_by_first_appearance():
    """Без этого правила два прогона одного набора дадут разные ответы."""
    assert majority_vote([7.0, 9.0])[0] == APPROX(7.0)


def test_majority_vote_confidence_is_one_when_everyone_agrees():
    assert majority_vote([5.0, 5.0, 5.0])[1] == APPROX(1.0)


# ---------------------------------------------------------- self_consistency
def test_self_consistency_outvotes_a_wrong_reasoning_path():
    samples = [
        "48 - 16 = 32, 32 - 8 = 24. The answer is 24.",
        "1/3 of 48 is 16, then 1/4 of 32 is 8. The answer is 24.",
        "Sell 1/3: 48 - 12 = 36, then 36 - 9 = 27. The answer is 27.",
    ]
    answer, confidence, votes = self_consistency(samples)
    assert answer == APPROX(24.0)
    assert confidence == APPROX(2 / 3)
    assert votes == Counter({24.0: 2, 27.0: 1})


def test_self_consistency_ignores_samples_without_an_answer():
    """Мусорный прогон не должен раздувать знаменатель уверенности."""
    samples = ["The answer is 5.", "The answer is 5.", "I got lost halfway."]
    _, confidence, votes = self_consistency(samples)
    assert confidence == APPROX(1.0)
    assert sum(votes.values()) == 2


def test_self_consistency_of_nothing_is_none():
    assert self_consistency([])[0] is None


def test_self_consistency_confidence_drops_when_paths_disagree():
    """Именно по падению уверенности принимают решение об эскалации к ToT."""
    agree = self_consistency(["The answer is 3.", "The answer is 3."])[1]
    disagree = self_consistency(["The answer is 3.", "The answer is 4."])[1]
    assert disagree < agree


# ---------------------------------------------------------- tree_of_thought
def _expand_arith(path):
    last = path[-1]
    return [last + 1, last + 3, last - 1]


def _score_last(path):
    return float(path[-1])


def test_tree_of_thought_returns_the_root_when_depth_is_zero():
    assert tree_of_thought(0, _expand_arith, _score_last, depth=0) == ([0], APPROX(0.0))


def test_tree_of_thought_path_grows_by_one_node_per_level():
    path, _ = tree_of_thought(0, _expand_arith, _score_last, depth=2)
    assert len(path) == 3


def test_tree_of_thought_finds_the_highest_scoring_path():
    assert tree_of_thought(0, _expand_arith, _score_last, depth=2)[0] == [0, 3, 6]


def test_tree_of_thought_keeps_the_best_path_when_the_last_level_is_worse():
    """Оценка ветки может рухнуть на следующем шаге — лучший путь терять нельзя."""
    expand = lambda path: {0: [5], 5: [-100]}.get(path[-1], [])
    path, score = tree_of_thought(0, expand, _score_last, depth=2)
    assert (path, score) == ([0, 5], APPROX(5.0))


def test_tree_of_thought_wider_beam_beats_a_greedy_one():
    """Смысл поиска: не отбрасывать ветку, которая выстрелит через шаг."""
    expand = lambda path: {0: [10, 1], 10: [10.5], 1: [100]}.get(path[-1], [])
    greedy = tree_of_thought(0, expand, _score_last, depth=2, beam=1)[1]
    wide = tree_of_thought(0, expand, _score_last, depth=2, beam=2)[1]
    assert wide > greedy
