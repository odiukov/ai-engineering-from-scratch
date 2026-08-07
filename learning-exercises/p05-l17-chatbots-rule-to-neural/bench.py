"""Входные данные для замера скорости."""

import random

random.seed(0)

_SWAPS = {"i": "you", "me": "you", "my": "your", "am": "are", "you": "i"}

_PATTERNS = [
    (r"my name is (\w+)", "Nice to meet you, {0}."),
    (r"i (need|want) (.+)", "Why do you {0} {1}?"),
    (r"i feel (.+)", "Why do you feel {0}?"),
    (r"(hi|hello|hey)\b.*", "Hello. How can I help?"),
    (r"(.*)", "Tell me more about that."),
]

_WORDS = [
    "order", "password", "reset", "cancel", "refund", "shipping", "account",
    "invoice", "delivery", "payment", "email", "address", "policy", "return",
]

_faq = [
    (" ".join(random.sample(_WORDS, 5)), f"answer {i}") for i in range(300)
]
_query = "how do i reset my account password"
_long_text = " ".join(random.choice(_WORDS) for _ in range(200))


def _llm(history, tools):
    """Заглушка модели: два вызова инструмента, потом финальный ответ."""
    if len(history) < 5:
        return {"tool_call": {"name": "echo", "arguments": {"text": "x"}}}
    return {"content": "done"}


BENCH = {
    "reflect": (_long_text, _SWAPS),
    "rule_based_respond": ("i feel tired after a very long day at work", _PATTERNS, _SWAPS),
    "jaccard_similarity": (_long_text, _query),
    "faq_respond": (_query, _faq, 0.3),
    "is_destructive_action": (_long_text,),
    "agent_loop": ("do the thing", {"echo": lambda text: text}, _llm, 5),
    "hybrid_chat": (_query, _faq, {"echo": lambda text: text}, _llm, lambda t: t, 0.6),
}
