"""Тесты к уроку «Групповой чат и выбор говорящего». Правь exercise.py."""

import pytest

from exercise import (
    DEFAULT_MAX_ROUNDS,
    TERMINATION_TOKEN,
    auto_selector,
    dominance,
    is_terminated,
    keyword_score,
    relevance_selector,
    round_robin_selector,
    run_groupchat,
    speaker_counts,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SPECIALTIES = {
    "coder": ["code", "implement"],
    "reviewer": ["review", "bug", "fix"],
    "manager": ["deadline", "scope"],
}
NAMES = list(SPECIALTIES)

# Все трое произносят слова кодера — без запрета повтора ход не уйдёт от него.
SHOUTS_CODE = {name: (lambda pool: "code code code") for name in SPECIALTIES}


def scripted_team():
    """Кодер пишет баг, ревьюер его ловит, менеджер закрывает чат."""

    def coder(pool):
        if any(speaker == "reviewer" for speaker, _ in pool):
            return "implement: fixed code, a + b"
        return "implement: code returns a - b"

    def reviewer(pool):
        last_code = [c for s, c in pool if s == "coder"]
        if last_code and "a + b" in last_code[-1]:
            return "review: approved"
        return "review: bug found, please fix"

    def manager(pool):
        if any("approved" in c for _, c in pool):
            return f"scope closed {TERMINATION_TOKEN}"
        return "deadline is friday"

    return {"coder": coder, "reviewer": reviewer, "manager": manager}


# ------------------------------------------------------------ keyword_score
def test_keyword_score_ignores_case():
    assert keyword_score("Please REVIEW the code", ["review"]) == 1


def test_keyword_score_counts_distinct_keywords_not_repetitions():
    """Слово, повторённое трижды, даёт 1 — иначе побеждает самый болтливый."""
    assert keyword_score("code code code", ["code"]) == 1


def test_keyword_score_is_zero_when_nothing_matches():
    assert keyword_score("nothing relevant here", ["review", "deadline"]) == 0


def test_keyword_score_sums_over_different_keywords():
    assert keyword_score("fix the bug in review", SPECIALTIES["reviewer"]) == 3


# ------------------------------------------------------ round_robin_selector
def test_round_robin_starts_with_the_first_name():
    assert round_robin_selector([], NAMES) == "coder"


def test_round_robin_advances_one_step():
    assert round_robin_selector([("coder", "hi")], NAMES) == "reviewer"


def test_round_robin_wraps_around():
    assert round_robin_selector([("manager", "hi")], NAMES) == "coder"


def test_round_robin_ignores_the_content_entirely():
    """Очередь дойдёт до менеджера, о чём бы ни шла речь."""
    about_code = round_robin_selector([("reviewer", "please write code")], NAMES)
    about_law = round_robin_selector([("reviewer", "check the contract")], NAMES)
    assert about_code == about_law == "manager"


def test_round_robin_rejects_a_speaker_outside_the_team():
    with pytest.raises(ValueError):
        round_robin_selector([("ghost", "hi")], NAMES)


# ------------------------------------------------------ relevance_selector
def test_relevance_picks_the_agent_whose_keywords_match():
    assert relevance_selector([("manager", "fix the bug")], SPECIALTIES) == "reviewer"


def test_relevance_starts_with_the_first_agent_on_an_empty_pool():
    assert relevance_selector([], SPECIALTIES) == "coder"


def test_relevance_breaks_ties_by_declaration_order():
    assert relevance_selector([("manager", "nothing matches")], SPECIALTIES) == "coder"


def test_relevance_respects_the_candidate_list():
    picked = relevance_selector([("manager", "fix the bug")], SPECIALTIES, ["coder", "manager"])
    assert picked == "coder"


# ---------------------------------------------------------- auto_selector
def test_auto_selector_never_gives_the_turn_back_to_the_last_speaker():
    assert auto_selector([("coder", "code code")], SPECIALTIES) != "coder"


def test_auto_selector_with_repeat_allowed_keeps_the_hot_speaker():
    assert auto_selector([("coder", "code code")], SPECIALTIES, allow_repeat=True) == "coder"


def test_auto_selector_still_answers_when_the_team_is_a_single_agent():
    """Запрет повтора не должен оставить селектор без кандидатов."""
    solo = {"coder": ["code"]}
    assert auto_selector([("coder", "code")], solo) == "coder"


# ------------------------------------------------------------ is_terminated
def test_termination_token_stops_the_chat():
    assert is_terminated([("manager", f"done {TERMINATION_TOKEN}")]) is True


def test_ordinary_message_does_not_stop_the_chat():
    assert is_terminated([("coder", "still working")]) is False


def test_empty_pool_is_not_terminated():
    assert is_terminated([]) is False


def test_round_cap_stops_the_chat_without_any_token():
    assert is_terminated([("coder", "x")] * DEFAULT_MAX_ROUNDS) is True


def test_token_must_be_at_the_end_not_merely_mentioned():
    assert is_terminated([("coder", f"do not say {TERMINATION_TOKEN} yet please")]) is False


# ------------------------------------------------------------ run_groupchat
def test_groupchat_under_round_robin_cycles_through_everyone():
    pool = run_groupchat(
        scripted_team(),
        lambda p: round_robin_selector(p, NAMES),
        max_rounds=6,
    )
    assert [name for name, _ in pool[:3]] == NAMES


def test_groupchat_stops_on_the_termination_token_before_the_cap():
    pool = run_groupchat(
        scripted_team(),
        lambda p: round_robin_selector(p, NAMES),
        max_rounds=DEFAULT_MAX_ROUNDS,
    )
    assert len(pool) < DEFAULT_MAX_ROUNDS
    assert pool[-1][1].endswith(TERMINATION_TOKEN)


def test_groupchat_stops_at_the_cap_when_nobody_says_the_token():
    endless = {name: (lambda pool: "still working") for name in SPECIALTIES}
    pool = run_groupchat(endless, lambda p: round_robin_selector(p, NAMES), max_rounds=7)
    assert len(pool) == 7


def test_groupchat_rejects_a_selector_that_invents_an_agent():
    with pytest.raises(ValueError):
        run_groupchat(scripted_team(), lambda p: "archivist", max_rounds=3)


def test_selector_without_a_repeat_ban_degenerates_into_a_monologue():
    """Все говорят словами кодера — и ход не уходит от кодера ни разу."""
    pool = run_groupchat(
        SHOUTS_CODE,
        lambda p: auto_selector(p, SPECIALTIES, allow_repeat=True),
        max_rounds=8,
    )
    assert speaker_counts(pool) == {"coder": 8}
    assert dominance(pool) == APPROX(1.0)


def test_the_repeat_ban_breaks_the_monologue():
    """Тот же чат с запретом повтора: монолог рассыпается на диалог."""
    pool = run_groupchat(
        SHOUTS_CODE,
        lambda p: auto_selector(p, SPECIALTIES, allow_repeat=False),
        max_rounds=8,
    )
    assert dominance(pool) < 1.0
    assert all(a != b for (a, _), (b, _) in zip(pool, pool[1:]))


# ----------------------------------------------------------- speaker_counts
def test_speaker_counts_tallies_the_pool():
    assert speaker_counts([("a", "x"), ("b", "y"), ("a", "z")]) == {"a": 2, "b": 1}


def test_speaker_counts_of_an_empty_pool_is_empty():
    assert speaker_counts([]) == {}


# ---------------------------------------------------------------- dominance
def test_dominance_of_a_balanced_chat():
    assert dominance([("a", "x"), ("b", "y")]) == APPROX(0.5)


def test_dominance_of_a_monologue_is_one():
    assert dominance([("a", "x"), ("a", "y")]) == APPROX(1.0)


def test_round_robin_keeps_dominance_at_its_floor():
    """Круг раздаёт ходы поровну — доля лидера не превышает 1/N с округлением."""
    endless = {name: (lambda pool: "still working") for name in SPECIALTIES}
    pool = run_groupchat(endless, lambda p: round_robin_selector(p, NAMES), max_rounds=9)
    assert dominance(pool) == APPROX(1 / 3)


def test_dominance_of_an_empty_pool_is_an_error():
    with pytest.raises(ValueError):
        dominance([])
