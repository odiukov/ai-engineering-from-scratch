"""Тесты к уроку «Общая память и доска объявлений». Правь exercise.py."""

import pytest

from exercise import (
    active_entries,
    append_entry,
    correct,
    make_entry,
    provenance_chain,
    spread,
    subscribe,
    verify,
)

TRUTH = {"page-1": 4.2}


def poisoned_pool():
    """Сценарий из урока: A прочитал 4.2, написал 42.0, B и C растащили число."""
    pool = []
    append_entry(pool, make_entry("A", "research", 42.0, 1, source="page-1"))
    spread(pool, ["B", "C"], "research", 10)
    return pool


# -------------------------------------------------------------- make_entry
def test_make_entry_records_who_wrote_and_when():
    e = make_entry("A", "prices", 4.2, 10, source="page-1")
    assert (e["writer"], e["topic"], e["value"], e["ts"], e["source"]) == (
        "A",
        "prices",
        4.2,
        10,
        "page-1",
    )


def test_make_entry_has_no_seq_until_appended():
    assert make_entry("A", "prices", 4.2, 10)["seq"] is None


def test_make_entry_without_provenance_cites_nobody():
    e = make_entry("A", "prices", 4.2, 10)
    assert (e["source"], e["cites"], e["supersedes"]) == (None, (), None)


def test_make_entry_freezes_cites_so_provenance_cannot_be_edited_later():
    refs = [0, 1]
    e = make_entry("B", "prices", 4.2, 11, cites=refs)
    refs.append(99)
    assert e["cites"] == (0, 1)


# ------------------------------------------------------------ append_entry
def test_append_assigns_sequential_numbers():
    pool = []
    first = append_entry(pool, make_entry("A", "t", 1.0, 0))
    second = append_entry(pool, make_entry("B", "t", 2.0, 1))
    assert (first, second, len(pool)) == (0, 1, 2)


def test_append_refuses_an_entry_that_is_already_in_the_log():
    pool = []
    entry = make_entry("A", "t", 1.0, 0)
    append_entry(pool, entry)
    with pytest.raises(ValueError):
        append_entry(pool, entry)


def test_append_refuses_a_citation_that_points_nowhere():
    pool = []
    with pytest.raises(ValueError):
        append_entry(pool, make_entry("A", "t", 1.0, 0, cites=(7,)))


def test_append_refuses_superseding_a_missing_entry():
    pool = []
    with pytest.raises(ValueError):
        append_entry(pool, make_entry("A", "t", 1.0, 0, supersedes=3))


# --------------------------------------------------------------- subscribe
def test_subscribe_returns_only_requested_topics():
    pool = []
    append_entry(pool, make_entry("A", "prices", 1.0, 0))
    append_entry(pool, make_entry("B", "orders", 2.0, 1))
    assert [e["topic"] for e in subscribe(pool, ["prices"])] == ["prices"]


def test_subscribe_preserves_write_order():
    pool = []
    for i in range(4):
        append_entry(pool, make_entry("A", "prices", float(i), i))
    assert [e["seq"] for e in subscribe(pool, ["prices"])] == [0, 1, 2, 3]


def test_subscribe_to_nothing_returns_nothing():
    assert subscribe(poisoned_pool(), []) == []


def test_blackboard_view_is_smaller_than_the_full_pool():
    """Ради этого доска и нужна: контекст агента не забивается чужой темой."""
    pool = []
    append_entry(pool, make_entry("A", "prices", 1.0, 0))
    append_entry(pool, make_entry("B", "orders", 2.0, 1))
    append_entry(pool, make_entry("C", "alerts", 3.0, 2))
    assert len(subscribe(pool, ["prices"])) < len(pool)


# ---------------------------------------------------------- active_entries
def test_active_entries_hides_superseded_writes():
    pool = []
    append_entry(pool, make_entry("A", "t", 42.0, 0, source="page-1"))
    append_entry(pool, make_entry("V", "t", 4.2, 1, source="page-1", supersedes=0))
    assert [e["seq"] for e in active_entries(pool)] == [1]


def test_superseded_entry_stays_in_the_log():
    """Append-only: отменённая запись не исчезает, иначе аудит теряется."""
    pool = []
    append_entry(pool, make_entry("A", "t", 42.0, 0))
    append_entry(pool, make_entry("V", "t", 4.2, 1, supersedes=0))
    assert len(pool) == 2 and pool[0]["value"] == 42.0


def test_active_entries_of_an_empty_pool_is_empty():
    assert active_entries([]) == []


# -------------------------------------------------------- provenance_chain
def test_provenance_chain_of_a_root_entry_is_itself():
    pool = []
    append_entry(pool, make_entry("A", "t", 1.0, 0, source="page-1"))
    assert provenance_chain(pool, 0) == [0]


def test_provenance_chain_reaches_the_original_source():
    pool = poisoned_pool()
    chain = provenance_chain(pool, 2)
    assert pool[chain[-1]]["source"] == "page-1"


def test_provenance_chain_only_points_backwards():
    """Ссылка вперёд означала бы цикл — append_entry такого не пропускает."""
    chain = provenance_chain(poisoned_pool(), 2)
    assert chain == sorted(chain, reverse=True)


# ------------------------------------------------------------------ spread
def test_spread_copies_the_same_value_to_every_reader():
    pool = poisoned_pool()
    assert [e["value"] for e in pool] == [42.0, 42.0, 42.0]


def test_spread_adds_one_entry_per_reader():
    pool = []
    append_entry(pool, make_entry("A", "research", 42.0, 1, source="page-1"))
    assert spread(pool, ["B", "C", "D"], "research", 10) == [1, 2, 3]


def test_every_spread_entry_cites_the_one_written_before_it():
    pool = poisoned_pool()
    assert (pool[1]["cites"], pool[2]["cites"]) == ((0,), (1,))


def test_spread_on_a_topic_nobody_wrote_to_raises():
    pool = poisoned_pool()
    with pytest.raises(ValueError):
        spread(pool, ["B"], "prices", 10)


# ------------------------------------------------------------------ verify
def test_verifier_flags_the_hallucination_and_everything_derived_from_it():
    assert verify(poisoned_pool(), TRUTH) == [0, 1, 2]


def test_verifier_writes_nothing_to_the_pool():
    """Верификатор без права записи — единственный, кого пул не отравит."""
    pool = poisoned_pool()
    before = len(pool)
    verify(pool, TRUTH)
    assert len(pool) == before


def test_verifier_is_silent_on_an_honest_pool():
    pool = []
    append_entry(pool, make_entry("A", "research", 4.2, 1, source="page-1"))
    spread(pool, ["B", "C"], "research", 10)
    assert verify(pool, TRUTH) == []


def test_verifier_skips_entries_whose_root_cited_no_source():
    """Агент ничего не читал, а рассуждал — сверять его вывод не с чем."""
    pool = []
    append_entry(pool, make_entry("D", "research", 99.0, 1))
    assert verify(pool, TRUTH) == []


def test_verifier_ignores_entries_that_were_already_superseded():
    pool = []
    append_entry(pool, make_entry("A", "research", 42.0, 1, source="page-1"))
    append_entry(
        pool, make_entry("V", "research", 4.2, 2, source="page-1", supersedes=0)
    )
    assert verify(pool, TRUTH) == []


# ----------------------------------------------------------------- correct
def test_correction_clears_every_flag():
    pool = poisoned_pool()
    correct(pool, verify(pool, TRUTH), "verifier-out", TRUTH, 30)
    assert verify(pool, TRUTH) == []


def test_correction_puts_the_true_value_into_active_state():
    pool = poisoned_pool()
    correct(pool, verify(pool, TRUTH), "verifier-out", TRUTH, 30)
    assert {e["value"] for e in active_entries(pool)} == {4.2}


def test_correction_keeps_the_poisoned_entries_in_the_log():
    pool = poisoned_pool()
    correct(pool, verify(pool, TRUTH), "verifier-out", TRUTH, 30)
    assert len(pool) == 6 and pool[0]["value"] == 42.0


def test_correction_is_a_new_entry_not_an_edit_in_place():
    pool = poisoned_pool()
    new = correct(pool, [0], "verifier-out", TRUTH, 30)
    assert pool[new[0]]["supersedes"] == 0 and pool[new[0]]["writer"] == "verifier-out"
