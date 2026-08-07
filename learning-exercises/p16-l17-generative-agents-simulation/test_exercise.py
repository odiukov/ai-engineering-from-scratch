"""Тесты к уроку «Генеративные агенты и эмерджентная симуляция». Правь exercise.py."""

import random

import pytest

from exercise import (
    HALF_LIFE,
    INVITATION,
    PARTY_QUERY,
    WEIGHTS,
    keywords,
    make_memory,
    make_plan,
    reflect,
    relevance,
    retrieval_score,
    retrieve,
    simulate,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def planners(agents):
    return [a["name"] for a in agents if a["plan"] is not None]


# ---------------------------------------------------------------- keywords
def test_keywords_drop_the_stopwords():
    assert keywords(INVITATION) == {"вечеринка", "кафе", "пять"}


def test_keywords_are_case_insensitive():
    assert keywords("Вечеринка КАФЕ") == keywords("вечеринка кафе")


def test_a_text_made_of_stopwords_has_no_keywords():
    assert keywords("в и на о с") == set()


# ------------------------------------------------------------- make_memory
def test_make_memory_records_text_kind_time_and_importance():
    m = make_memory(INVITATION, "observation", 3, 9)
    assert (m["text"], m["kind"], m["ts"], m["importance"]) == (INVITATION, "observation", 3, 9)


def test_a_fresh_memory_is_not_yet_reflected():
    assert make_memory("что-то", "observation", 0, 5)["reflected"] is False


def test_make_memory_rejects_importance_outside_the_scale():
    with pytest.raises(ValueError):
        make_memory("что-то", "observation", 0, 11)


def test_make_memory_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        make_memory("что-то", "dream", 0, 5)


# --------------------------------------------------------------- relevance
def test_relevance_of_an_exact_match_is_one():
    assert relevance(make_memory(PARTY_QUERY, "observation", 0, 5), PARTY_QUERY) == APPROX(1.0)


def test_relevance_of_an_unrelated_memory_is_zero():
    memory = make_memory("кот прошёл мимо окна", "observation", 0, 5)
    assert relevance(memory, PARTY_QUERY) == APPROX(0.0)


def test_partial_overlap_lands_strictly_between_zero_and_one():
    assert 0.0 < relevance(make_memory(INVITATION, "observation", 0, 5), PARTY_QUERY) < 1.0


# --------------------------------------------------------- retrieval_score
def test_the_score_of_a_memory_falls_as_it_ages():
    memory = make_memory(INVITATION, "observation", 0, 5)
    assert retrieval_score(memory, PARTY_QUERY, 6) < retrieval_score(memory, PARTY_QUERY, 0)


def test_one_half_life_halves_exactly_the_recency_term():
    memory = make_memory(PARTY_QUERY, "observation", 0, 6)
    floor = WEIGHTS["importance"] * 0.6 + WEIGHTS["relevance"] * relevance(memory, PARTY_QUERY)
    fresh = retrieval_score(memory, PARTY_QUERY, 0) - floor
    aged = retrieval_score(memory, PARTY_QUERY, HALF_LIFE) - floor
    assert aged == pytest.approx(fresh / 2, abs=1e-9)


def test_an_old_important_memory_outscores_a_fresh_triviality():
    """Ради этого в оценке три слагаемых, а не одна свежесть."""
    important = make_memory(PARTY_QUERY, "observation", 0, 10)
    trivial = make_memory("кот прошёл мимо окна", "observation", 10, 1)
    assert retrieval_score(important, PARTY_QUERY, 10) > retrieval_score(trivial, PARTY_QUERY, 10)


def test_relevance_separates_two_otherwise_identical_memories():
    on_topic = make_memory(PARTY_QUERY, "observation", 4, 5)
    off_topic = make_memory("кот прошёл мимо окна", "observation", 4, 5)
    assert retrieval_score(on_topic, PARTY_QUERY, 4) > retrieval_score(off_topic, PARTY_QUERY, 4)


# ---------------------------------------------------------------- retrieve
def test_retrieve_returns_the_top_k_in_score_order():
    stream = [
        make_memory("кот прошёл мимо окна", "observation", 4, 1),
        make_memory(PARTY_QUERY, "observation", 4, 10),
        make_memory("дождь", "observation", 4, 2),
    ]
    top = retrieve(stream, PARTY_QUERY, 4, k=2)
    assert [m["text"] for m in top] == [PARTY_QUERY, "дождь"]


def test_retrieve_never_returns_more_than_the_stream_holds():
    stream = [make_memory(INVITATION, "observation", 0, 5)]
    assert len(retrieve(stream, PARTY_QUERY, 0, k=10)) == 1


def test_retrieve_of_an_empty_stream_is_empty():
    assert retrieve([], PARTY_QUERY, 0) == []


def test_retrieval_is_ranked_not_filtered():
    """Слабая запись всё равно вернётся, если сильнее ничего нет."""
    stream = [make_memory("кот прошёл мимо окна", "observation", 0, 1)]
    assert len(retrieve(stream, PARTY_QUERY, 100)) == 1


# ----------------------------------------------------------------- reflect
def test_reflection_waits_until_the_unprocessed_importance_adds_up():
    stream = [make_memory(INVITATION, "observation", 0, 3)]
    assert reflect(stream, 1) is None


def test_reflection_is_appended_to_the_stream_as_a_new_memory():
    stream = [make_memory(INVITATION, "observation", 0, 9)]
    entry = reflect(stream, 1)
    assert entry["kind"] == "reflection" and stream[-1] is entry


def test_reflection_names_the_most_frequent_topic():
    stream = [make_memory(INVITATION, "observation", 0, 9)]
    assert "вечеринка" in reflect(stream, 1)["text"]


def test_reflection_does_not_chew_the_same_memories_twice():
    stream = [make_memory(INVITATION, "observation", 0, 9)]
    reflect(stream, 1)
    assert reflect(stream, 2) is None
    assert all(m["reflected"] for m in stream[:1])


def test_a_conclusion_weighs_more_than_the_observations_behind_it():
    """Поэтому вывод и переживает затухание дольше сырого наблюдения."""
    stream = [make_memory(INVITATION, "observation", 0, 5), make_memory(INVITATION, "observation", 0, 5)]
    entry = reflect(stream, 1)
    assert entry["importance"] > max(m["importance"] for m in stream[:2])


# --------------------------------------------------------------- make_plan
def test_a_strong_fresh_memory_produces_a_plan():
    stream = [make_memory(INVITATION, "observation", 0, 9)]
    assert make_plan(stream, PARTY_QUERY, 0) == f"план: {PARTY_QUERY}"


def test_an_empty_stream_produces_no_plan():
    assert make_plan([], PARTY_QUERY, 0) is None


def test_the_plan_dissipates_as_the_memory_decays():
    stream = [make_memory(INVITATION, "observation", 0, 5)]
    assert make_plan(stream, PARTY_QUERY, 0) is not None
    assert make_plan(stream, PARTY_QUERY, 20) is None


# ---------------------------------------------------------------- simulate
def test_only_one_agent_starts_out_knowing_about_the_party():
    assert planners(simulate(5, 0, random.Random(0))) == ["a0"]


def test_a_single_seed_reaches_everyone_without_an_orchestrator():
    """Эмерджентность: вечеринка собралась из двусторонних встреч, без центра."""
    assert len(planners(simulate(5, 24, random.Random(0)))) == 5


def test_without_reflection_the_goal_dissipates():
    with_reflection = planners(simulate(5, 24, random.Random(0)))
    without = planners(simulate(5, 24, random.Random(0), use_reflection=False))
    assert len(without) < len(with_reflection)


def test_reflection_helps_across_many_seeds_not_just_a_lucky_one():
    on = sum(len(planners(simulate(5, 24, random.Random(s)))) for s in range(8))
    off = sum(
        len(planners(simulate(5, 24, random.Random(s), use_reflection=False)))
        for s in range(8)
    )
    assert off < on


def test_the_simulation_is_reproducible_for_a_given_seed():
    assert planners(simulate(5, 12, random.Random(3))) == planners(
        simulate(5, 12, random.Random(3))
    )
