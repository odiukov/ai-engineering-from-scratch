"""Тесты к уроку «Чат-боты: от правил к нейросетям и агентам». Правь exercise.py."""

import pytest

from exercise import (
    agent_loop,
    faq_respond,
    hybrid_chat,
    is_destructive_action,
    jaccard_similarity,
    reflect,
    rule_based_respond,
)

SWAPS = {
    "i": "you",
    "me": "you",
    "my": "your",
    "am": "are",
    "you": "i",
    "your": "my",
    "are": "am",
}

PATTERNS = [
    (r"my name is (\w+)", "Nice to meet you, {0}."),
    (r"i (need|want) (.+)", "Why do you {0} {1}?"),
    (r"i feel (.+)", "Why do you feel {0}?"),
    (r"(.*)", "Tell me more about that."),
]

FAQ = [
    ("how do i reset my password", "Settings > Security > Reset Password."),
    ("how do i cancel my order", "Orders > Cancel."),
]

FLOW = lambda text: f"confirm: {text}"


def boom_llm(history, tools):
    """LLM, который обязан остаться невызванным. Сработает — тест покраснеет."""
    raise AssertionError("до LLM дойти было нельзя")


# ------------------------------------------------------------------- reflect
def test_reflect_swaps_pronouns_of_the_user_utterance():
    assert reflect("I am sad about my job", SWAPS) == "you are sad about your job"


def test_reflect_makes_exactly_one_pass():
    """Ловушка: два прохода схлопнут "i you" в "i i"."""
    assert reflect("i you", SWAPS) == "you i"


def test_reflect_leaves_unknown_words_alone():
    assert reflect("the sky is blue", SWAPS) == "the sky is blue"


def test_reflect_misses_a_word_glued_to_punctuation():
    """Ограничение подхода: split по пробелам, "my," в словаре не найдётся."""
    assert reflect("about my, job", SWAPS) == "about my, job"


# ------------------------------------------------------- rule_based_respond
def test_rule_based_respond_uses_the_first_matching_pattern():
    assert rule_based_respond("My name is Alex", PATTERNS, SWAPS) == (
        "Nice to meet you, Alex."
    )


def test_rule_based_respond_reflects_the_captured_group():
    """Тот самый фокус ELIZA: "my job" возвращается пользователю как "your job"."""
    assert rule_based_respond("I feel bad about my job", PATTERNS, SWAPS) == (
        "Why do you feel bad about your job?"
    )


def test_rule_based_respond_falls_through_to_the_catch_all():
    assert rule_based_respond("The sky is blue", PATTERNS, SWAPS) == (
        "Tell me more about that."
    )


def test_rule_based_respond_treats_list_order_as_priority():
    """Ловушка: r"(.*)" впереди — и все остальные правила мертвы."""
    inverted = [(r"(.*)", "Tell me more."), (r"i feel (.+)", "Why do you feel {0}?")]
    assert rule_based_respond("i feel bad", inverted, SWAPS) == "Tell me more."


def test_rule_based_respond_ignores_case_when_matching():
    assert rule_based_respond("I Feel Sad", PATTERNS, SWAPS) == "Why do you feel Sad?"


def test_rule_based_respond_returns_the_fallback_when_nothing_matches():
    assert rule_based_respond("hello", [], SWAPS) == "I don't understand."


# -------------------------------------------------------- jaccard_similarity
def test_jaccard_similarity_of_half_overlapping_texts():
    assert jaccard_similarity("cancel my order", "cancel the order") == pytest.approx(
        0.5
    )


def test_jaccard_similarity_ignores_case_and_word_order():
    assert jaccard_similarity("Reset Password", "password reset") == pytest.approx(1.0)


def test_jaccard_similarity_ignores_repeats():
    """Множество, а не мешок: три "cat" и один "cat" — одно и то же."""
    assert jaccard_similarity("cat cat cat", "cat") == pytest.approx(1.0)


def test_jaccard_similarity_of_disjoint_texts_is_zero():
    assert jaccard_similarity("reset password", "track shipment") == pytest.approx(0.0)


def test_jaccard_similarity_of_two_empty_texts_is_zero_not_a_crash():
    assert jaccard_similarity("", "!!!") == pytest.approx(0.0)


# ---------------------------------------------------------------- faq_respond
def test_faq_respond_returns_the_answer_to_the_exact_question():
    assert faq_respond("how do i cancel my order", FAQ) == "Orders > Cancel."


def test_faq_respond_survives_a_paraphrase():
    assert faq_respond("how do i reset my password please", FAQ) == (
        "Settings > Security > Reset Password."
    )


def test_faq_respond_refuses_instead_of_inventing_an_answer():
    """Отказ — не сбой, а проектное решение: генерации нет, врать нечем."""
    assert faq_respond("what is the weather in berlin", FAQ) is None


def test_faq_respond_with_a_zero_threshold_never_refuses():
    assert faq_respond("what is the weather in berlin", FAQ, threshold=0.0) is not None


def test_faq_respond_on_an_empty_faq_returns_none():
    assert faq_respond("anything at all", [], threshold=0.0) is None


# -------------------------------------------------------- is_destructive_action
def test_is_destructive_action_finds_the_danger_word():
    assert is_destructive_action("please cancel my order") is True


def test_is_destructive_action_ignores_case():
    assert is_destructive_action("REFUND me now") is True


def test_is_destructive_action_is_false_for_a_harmless_question():
    assert is_destructive_action("how do i reset my password") is False


def test_is_destructive_action_over_triggers_on_substrings_by_design():
    """"cancellation" ловится целиком — для guardrail это осознанный перекос."""
    assert is_destructive_action("what is the cancellation policy") is True


# ----------------------------------------------------------------- agent_loop
def test_agent_loop_starts_the_history_with_the_user_message():
    seen = []

    def llm(history, tools):
        seen.append([dict(m) for m in history])
        return {"content": "ok"}

    assert agent_loop("hello", {}, llm) == "ok"
    assert seen[0] == [{"role": "user", "content": "hello"}]


def test_agent_loop_feeds_the_tool_result_back_into_the_history():
    seen = []

    def llm(history, tools):
        seen.append([dict(m) for m in history])
        if len(history) == 1:
            return {"tool_call": {"name": "add", "arguments": {"a": 2, "b": 3}}}
        return {"content": history[-1]["content"]}

    assert agent_loop("2+3", {"add": lambda a, b: str(a + b)}, llm) == "5"
    assert seen[1][1]["role"] == "assistant"
    assert seen[1][2] == {"role": "tool", "name": "add", "content": "5"}


def test_agent_loop_survives_an_unknown_tool_name():
    """Ловушка: ошибка инструмента уходит в history, а не роняет цикл."""

    def llm(history, tools):
        if len(history) == 1:
            return {"tool_call": {"name": "nope", "arguments": {}}}
        return {"content": history[-1]["content"]}

    assert agent_loop("hi", {"add": lambda: "x"}, llm) == "error: unknown tool 'nope'"


def test_agent_loop_rejects_arguments_that_are_not_a_dict():
    def llm(history, tools):
        if len(history) == 1:
            return {"tool_call": {"name": "add", "arguments": [1, 2]}}
        return {"content": history[-1]["content"]}

    assert agent_loop("hi", {"add": lambda a, b: "x"}, llm) == (
        "error: arguments must be a dict, got list"
    )


def test_agent_loop_returns_a_runtime_error_to_the_llm_as_an_observation():
    def fail():
        raise RuntimeError("service unavailable")

    def llm(history, tools):
        if len(history) == 1:
            return {"tool_call": {"name": "lookup", "arguments": {}}}
        return {"content": history[-1]["content"]}

    assert agent_loop("hi", {"lookup": fail}, llm) == (
        "error: tool 'lookup' failed: RuntimeError: service unavailable"
    )


def test_agent_loop_stops_at_the_step_budget():
    """Ловушка: без бюджета зациклившийся агент крутится вечно."""
    calls = []

    def llm(history, tools):
        calls.append(1)
        return {"tool_call": {"name": "ping", "arguments": {}}}

    out = agent_loop("hi", {"ping": lambda: "pong"}, llm, max_steps=3)
    assert out == "I could not complete the task in the step budget."
    assert len(calls) == 3


def test_agent_loop_hands_the_tools_to_the_llm():
    tools = {"ping": lambda: "pong"}
    seen = []

    def llm(history, given_tools):
        seen.append(given_tools)
        return {"content": "ok"}

    agent_loop("hi", tools, llm)
    assert seen[0] is tools


# ---------------------------------------------------------------- hybrid_chat
def test_hybrid_chat_sends_a_destructive_request_to_the_structured_flow():
    assert hybrid_chat("please delete my account", FAQ, {}, boom_llm, FLOW) == (
        "confirm: please delete my account"
    )


def test_hybrid_chat_answers_a_known_question_from_the_faq():
    assert hybrid_chat("how do i reset my password", FAQ, {}, boom_llm, FLOW) == (
        "Settings > Security > Reset Password."
    )


def test_hybrid_chat_falls_back_to_the_agent_for_anything_else():
    llm = lambda history, tools: {"content": "agent says hi"}
    assert hybrid_chat("tell me a joke", FAQ, {}, llm, FLOW) == "agent says hi"


def test_hybrid_chat_checks_destruction_before_retrieval():
    """Ловушка порядка: FAQ ответил бы, но отмена заказа обязана идти в сценарий."""
    assert hybrid_chat(
        "how do i cancel my order", FAQ, {}, boom_llm, FLOW, threshold=0.0
    ) == "confirm: how do i cancel my order"
