"""Тесты к уроку «Автоматизированные исследования алаймента». Правь exercise.py."""

import random

import pytest

from exercise import (
    FIXED_SPREAD,
    GENESIS,
    HASH_LEN,
    allocate,
    append_record,
    record_hash,
    regime_summary,
    run_forum,
    solve_task,
    tamper_record,
    verify_forum,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TASKS = [
    ("weak-to-strong-distill", 0.40),
    ("reward-model-diagnosis", 0.30),
    ("in-context-safety-probe", 0.50),
    ("alignment-faking-detector", 0.25),
]
AGENTS = ["AAR-A", "AAR-B", "AAR-C"]


def finding(author="AAR-A", task="t", regime="free", result=0.5):
    """Находка агента до того, как она уехала в цепочку."""
    return {"author": author, "task": task, "regime": regime, "result": result}


# ---------------------------------------------------------------- record_hash
def test_hash_has_the_declared_length():
    assert len(record_hash(finding(), GENESIS)) == HASH_LEN


def test_hash_is_deterministic_with_no_salt_and_no_clock():
    """Проверить цепочку через год должно быть чем."""
    first = record_hash(finding(), GENESIS)
    second = record_hash(finding(), GENESIS)
    assert first == second


def test_changing_the_result_changes_the_hash():
    assert record_hash(finding(result=0.5), GENESIS) != record_hash(finding(result=0.9), GENESIS)


def test_the_previous_link_is_part_of_the_signature():
    """Иначе записи можно переставлять местами, и каждая по отдельности сойдётся."""
    same = finding()
    assert record_hash(same, GENESIS) != record_hash(same, "a" * HASH_LEN)


def test_keys_the_agent_invented_do_not_enter_the_signature():
    plain = finding()
    padded = {**plain, "self_assessment": "excellent"}
    assert record_hash(plain, GENESIS) == record_hash(padded, GENESIS)


# -------------------------------------------------------------- append_record
def test_the_first_record_links_to_genesis():
    forum = append_record([], finding())
    assert forum[0]["prev_hash"] == GENESIS


def test_each_record_links_to_the_previous_one():
    forum = append_record(append_record([], finding()), finding(result=0.7))
    assert forum[1]["prev_hash"] == forum[0]["my_hash"]


def test_appending_returns_a_new_list_and_leaves_the_old_forum_alone():
    """Append-only — это в первую очередь про то, что лежащее не меняется."""
    before = append_record([], finding())
    append_record(before, finding(result=0.9))
    assert len(before) == 1


def test_appending_does_not_write_hashes_back_into_the_caller_record():
    original = finding()
    append_record([], original)
    assert "my_hash" not in original


def test_a_freshly_built_chain_verifies_clean():
    forum = []
    for i in range(5):
        forum = append_record(forum, finding(result=i / 10))
    assert verify_forum(forum) == []


# --------------------------------------------------------------- verify_forum
def test_an_empty_forum_has_nothing_to_complain_about():
    assert verify_forum([]) == []


def test_a_silent_edit_is_caught():
    forum = append_record(append_record([], finding()), finding(result=0.7))
    assert verify_forum(tamper_record(forum, 0, 0.5)) == [0]


def test_only_the_edited_record_is_flagged_not_the_whole_tail():
    """Отчёт «испорчено 47 записей» перестаёт указывать на место."""
    forum = []
    for i in range(8):
        forum = append_record(forum, finding(result=i / 10))
    assert verify_forum(tamper_record(forum, 3, 0.5)) == [3]


def test_reordering_two_records_breaks_the_chain():
    forum = append_record(append_record([], finding(task="a")), finding(task="b"))
    swapped = [forum[1], forum[0]]
    assert verify_forum(swapped) != []


def test_dropping_a_record_from_the_middle_is_visible():
    forum = []
    for i in range(4):
        forum = append_record(forum, finding(result=i / 10))
    assert verify_forum(forum[:1] + forum[2:]) == [1]


# -------------------------------------------------------------- tamper_record
def test_tampering_raises_the_reported_result():
    forum = append_record([], finding(result=0.2))
    assert tamper_record(forum, 0, 0.5)[0]["result"] == APPROX(0.7)


def test_tampering_leaves_the_hashes_untouched_which_is_exactly_the_giveaway():
    forum = append_record([], finding(result=0.2))
    assert tamper_record(forum, 0, 0.5)[0]["my_hash"] == forum[0]["my_hash"]


def test_tampering_works_on_a_copy():
    forum = append_record([], finding(result=0.2))
    tamper_record(forum, 0, 0.5)
    assert forum[0]["result"] == APPROX(0.2)
    assert verify_forum(forum) == []


def test_an_index_outside_the_forum_is_rejected():
    forum = append_record([], finding())
    with pytest.raises(IndexError):
        tamper_record(forum, 5, 0.5)


# -------------------------------------------------------------------- allocate
def test_tasks_are_dealt_round_robin():
    assert allocate(["a", "b", "c"], ["X", "Y"]) == {"X": ["a", "c"], "Y": ["b"]}


def test_every_task_is_handed_out_exactly_once():
    plan = allocate([f"t{i}" for i in range(17)], AGENTS)
    handed = [task for tasks in plan.values() for task in tasks]
    assert sorted(handed) == sorted(f"t{i}" for i in range(17))


def test_load_differs_by_at_most_one_task():
    sizes = [len(v) for v in allocate([f"t{i}" for i in range(17)], AGENTS).values()]
    assert max(sizes) - min(sizes) <= 1


def test_allocation_does_not_depend_on_any_rng():
    """Два прогона «с одинаковыми настройками» должны быть одним экспериментом."""
    first = allocate(["a", "b", "c", "d"], AGENTS)
    second = allocate(["a", "b", "c", "d"], AGENTS)
    assert first == second


def test_an_agent_with_no_tasks_still_appears_in_the_plan():
    assert allocate(["a"], ["X", "Y"]) == {"X": ["a"], "Y": []}


def test_no_agents_means_the_tasks_would_vanish():
    with pytest.raises(ValueError):
        allocate(["a"], [])


def test_duplicate_agent_names_are_rejected():
    """Второй "X" затёр бы список первого, и часть задач потерялась бы молча."""
    with pytest.raises(ValueError):
        allocate(["a", "b"], ["X", "X"])


# ------------------------------------------------------------------ solve_task
def test_a_prescribed_workflow_never_falls_below_the_baseline():
    rng = random.Random(0)
    results = [solve_task(rng, 0.4, "fixed") for _ in range(2000)]
    assert min(results) >= 0.4


def test_a_prescribed_workflow_has_a_hard_ceiling():
    rng = random.Random(1)
    results = [solve_task(rng, 0.4, "fixed") for _ in range(2000)]
    assert max(results) < 0.4 + FIXED_SPREAD


def test_free_decomposition_reaches_higher_than_the_prescribed_ceiling():
    rng = random.Random(2)
    free = [solve_task(rng, 0.4, "free") for _ in range(2000)]
    assert max(free) > 0.4 + FIXED_SPREAD


def test_free_decomposition_also_undershoots_the_baseline():
    """Тот же простор, что даёт рекорд, даёт и провал."""
    rng = random.Random(3)
    free = [solve_task(rng, 0.4, "free") for _ in range(2000)]
    assert min(free) < 0.4


def test_free_decomposition_has_the_wider_spread():
    rng = random.Random(4)
    fixed = [solve_task(rng, 0.4, "fixed") for _ in range(3000)]
    free = [solve_task(rng, 0.4, "free") for _ in range(3000)]
    assert max(free) - min(free) > max(fixed) - min(fixed)


def test_a_misspelled_regime_does_not_silently_pick_a_branch():
    with pytest.raises(ValueError):
        solve_task(random.Random(0), 0.4, "freee")


# ------------------------------------------------------------------- run_forum
def test_the_forum_holds_one_record_per_task():
    forum = run_forum(random.Random(3), TASKS, AGENTS, "fixed")
    assert len(forum) == len(TASKS)


def test_every_task_is_reported_exactly_once():
    forum = run_forum(random.Random(3), TASKS, AGENTS, "free")
    assert sorted(r["task"] for r in forum) == sorted(name for name, _ in TASKS)


def test_the_forum_it_produces_verifies_clean():
    forum = run_forum(random.Random(3), TASKS, AGENTS, "free")
    assert verify_forum(forum) == []


def test_same_seed_reproduces_the_whole_forum():
    a = run_forum(random.Random(5), TASKS, AGENTS, "free")
    b = run_forum(random.Random(5), TASKS, AGENTS, "free")
    assert a == b


def test_the_regime_is_recorded_on_every_finding():
    forum = run_forum(random.Random(6), TASKS, AGENTS, "fixed")
    assert {r["regime"] for r in forum} == {"fixed"}


def test_a_tampered_run_is_still_caught_after_the_fact():
    forum = run_forum(random.Random(7), TASKS, AGENTS, "free")
    assert verify_forum(tamper_record(forum, 2, 0.5)) == [2]


# -------------------------------------------------------------- regime_summary
def test_summary_counts_every_record():
    forum = run_forum(random.Random(8), TASKS, AGENTS, "fixed")
    assert regime_summary(forum)["records"] == len(TASKS)


def test_per_task_statistics_bracket_the_mean():
    forum = run_forum(random.Random(9), TASKS, AGENTS, "free")
    for stats in regime_summary(forum)["per_task"].values():
        assert stats["min"] <= stats["mean"] <= stats["max"]


def test_overall_mean_averages_tasks_not_records():
    """Задача с тремя отчётами не должна перевешивать задачу с одним."""
    forum = []
    for result in (0.4, 0.6):
        forum = append_record(forum, finding(task="t1", result=result))
    forum = append_record(forum, finding(task="t2", result=0.5))
    assert regime_summary(forum)["overall_mean"] == APPROX(0.5)


def test_a_lopsided_allocation_does_not_move_the_overall_mean():
    forum = []
    for _ in range(9):
        forum = append_record(forum, finding(task="t1", result=1.0))
    forum = append_record(forum, finding(task="t2", result=0.0))
    assert regime_summary(forum)["overall_mean"] == APPROX(0.5)


def test_counts_per_task_are_reported():
    forum = append_record(append_record([], finding(task="t1")), finding(task="t1"))
    assert regime_summary(forum)["per_task"]["t1"]["count"] == 2


def test_an_empty_forum_has_no_summary():
    with pytest.raises(ValueError):
        regime_summary([])
