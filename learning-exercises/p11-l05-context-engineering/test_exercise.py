"""Тесты к уроку «Context engineering: бюджет окна, порядок, память». Правь exercise.py."""

import pytest

from exercise import (
    DEFAULT_INTENT,
    TOOL_REGISTRY,
    allocate_budget,
    classify_intent,
    compress_history,
    count_tokens,
    reorder_lost_in_middle,
    score_relevance,
    select_tools,
    truncate_to_tokens,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

KB = [
    "The project uses PostgreSQL 16 with pgvector for embedding storage",
    "Authentication is handled by Supabase Auth with JWT tokens",
    "Test coverage must be above 80 percent for all new modules",
]


# ---------------------------------------------------------------- count_tokens
def test_count_tokens_scales_word_count():
    assert count_tokens("one two three") == 3


def test_count_tokens_of_empty_text_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_grows_with_the_text():
    assert count_tokens("a b c d e f g h i j") > count_tokens("a b c")


# ---------------------------------------------------------- truncate_to_tokens
def test_truncate_to_tokens_cuts_to_the_limit():
    assert truncate_to_tokens("a b c d e", 3) == "a b"


def test_truncate_to_tokens_leaves_short_text_alone():
    assert truncate_to_tokens("a b c", 100) == "a b c"


def test_truncate_to_tokens_with_zero_budget_returns_nothing():
    assert truncate_to_tokens("a b c", 0) == ""


def test_truncate_to_tokens_never_overshoots_the_budget():
    """Ловушка: деление на 1.3 округляется вниз и легко даёт лишний токен."""
    text = " ".join(str(i) for i in range(50))
    for limit in range(1, 20):
        assert count_tokens(truncate_to_tokens(text, limit)) <= limit


# --------------------------------------------------------------- score_relevance
def test_score_relevance_counts_the_share_of_query_words():
    assert score_relevance("vector search", ["vector db", "cooking"]) == APPROX([0.5, 0.0])


def test_score_relevance_ignores_letter_case():
    assert score_relevance("JWT", ["Authentication uses jwt tokens"]) == APPROX([1.0])


def test_score_relevance_of_an_empty_query_is_all_zeros():
    assert score_relevance("", KB) == APPROX([0.0, 0.0, 0.0])


# ------------------------------------------------------ reorder_lost_in_middle
def test_reorder_puts_the_best_document_first():
    order = reorder_lost_in_middle(["a", "b", "c", "d"], [0.9, 0.1, 0.5, 0.7])
    assert order[0] == "a"


def test_reorder_puts_the_runner_up_last():
    """Конец окна модель читает почти так же хорошо, как начало."""
    order = reorder_lost_in_middle(["a", "b", "c", "d"], [0.9, 0.1, 0.5, 0.7])
    assert order[-1] == "d"


def test_reorder_buries_the_weakest_document_in_the_middle():
    order = reorder_lost_in_middle(["a", "b", "c", "d"], [0.9, 0.1, 0.5, 0.7])
    assert order.index("b") not in (0, len(order) - 1)


def test_reorder_survives_uncomparable_items_with_equal_scores():
    """Ловушка: сортировка пар (score, item) уронит TypeError на двух dict."""
    docs = [{"id": 1}, {"id": 2}]
    assert len(reorder_lost_in_middle(docs, [0.5, 0.5])) == 2


def test_reorder_keeps_every_document():
    docs = ["a", "b", "c", "d", "e"]
    assert sorted(reorder_lost_in_middle(docs, [0.95, 0.6, 0.2, 0.8, 0.5])) == docs


# ---------------------------------------------------------------- allocate_budget
def test_allocate_budget_truncates_to_the_window():
    assert allocate_budget([("sys", "a b c d", None)], 3) == [("sys", "a b", 2)]


def test_allocate_budget_respects_a_per_component_limit():
    report = allocate_budget([("sys", "a b c d e f", 2)], 1000)
    assert report[0][2] <= 2


def test_allocate_budget_reserves_room_for_the_answer():
    """Забить окно под завязку — значит не оставить модели места на ответ."""
    text = " ".join(str(i) for i in range(100))
    report = allocate_budget([("docs", text, None)], 100, generation_reserve=90)
    assert report[0][2] <= 10


def test_allocate_budget_starves_later_components_first():
    """Порядок компонентов — это и есть приоритет: кто раньше, тот и получил."""
    text = " ".join(str(i) for i in range(100))
    report = allocate_budget([("first", text, None), ("second", text, None)], 20)
    assert report[0][2] > 0
    assert report[1][2] == 0


def test_allocate_budget_reports_every_component_even_the_starved_one():
    text = " ".join(str(i) for i in range(100))
    report = allocate_budget([("first", text, None), ("second", text, None)], 20)
    assert [name for name, _, _ in report] == ["first", "second"]


# --------------------------------------------------------------- compress_history
def test_compress_history_leaves_a_short_conversation_untouched():
    turns = [("user", "a b c"), ("assistant", "d e f")]
    assert compress_history(turns, 100) == ("", turns)


def test_compress_history_folds_the_oldest_turns_into_a_summary():
    turns = [("user", "w " * 20), ("assistant", "x " * 20), ("user", "y " * 3)]
    summary, kept = compress_history(turns, 10)
    assert summary.startswith("Previous:")
    assert len(kept) < len(turns)


def test_compress_history_keeps_the_most_recent_turns():
    turns = [("user", "old " * 20), ("assistant", "mid " * 20), ("user", "fresh here")]
    _, kept = compress_history(turns, 5)
    assert kept[-1] == ("user", "fresh here")


def test_compress_history_never_folds_below_keep_last():
    """Даже нулевой бюджет не имеет права стереть текущий вопрос пользователя."""
    turns = [("user", "a " * 50), ("assistant", "b " * 50), ("user", "c " * 50)]
    _, kept = compress_history(turns, 0, keep_last=2)
    assert len(kept) == 2


def test_compress_history_does_not_mutate_the_input():
    turns = [("user", "a " * 30), ("assistant", "b " * 30), ("user", "c")]
    compress_history(turns, 1)
    assert len(turns) == 3


# --------------------------------------------------------------- classify_intent
def test_classify_intent_recognizes_a_code_task():
    assert classify_intent("Fix the bug in auth.py") == ["code"]


def test_classify_intent_recognizes_a_calendar_task():
    assert classify_intent("Schedule a meeting with the team") == ["calendar"]


def test_classify_intent_ignores_letter_case():
    assert classify_intent("SCHEDULE A MEETING") == ["calendar"]


def test_classify_intent_falls_back_to_the_default():
    """Пустой список намерений оставил бы модель вообще без инструментов."""
    assert classify_intent("hello there") == [DEFAULT_INTENT]


# ------------------------------------------------------------------ select_tools
def test_select_tools_picks_only_matching_categories():
    picked, _ = select_tools("Fix the bug in the code", 1000)
    assert "create_calendar_event" not in picked
    assert "search_code" in picked


def test_select_tools_stays_within_the_budget():
    picked, total = select_tools("Fix the bug in the code", 300)
    assert total <= 300
    assert total == sum(TOOL_REGISTRY[name]["tokens"] for name in picked)


def test_select_tools_fits_more_tools_by_taking_cheap_ones_first():
    """Ради этого tool pruning и делают: те же 300 токенов, но больше пользы."""
    picked, _ = select_tools("Fix the bug in the code", 300)
    assert len(picked) >= 2


def test_select_tools_with_no_budget_picks_nothing():
    assert select_tools("Fix the bug in the code", 0) == ([], 0)
