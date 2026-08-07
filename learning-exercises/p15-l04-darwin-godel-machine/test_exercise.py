"""Тесты к уроку «Darwin Godel Machine». Правь exercise.py."""

import random

import pytest

from exercise import (
    CASES,
    HACK_RATE,
    MAX_OPS,
    TOOLS,
    apply_ops,
    archive_accept,
    archive_key,
    mutate_agent,
    passes_gate,
    reported_score,
    run_dgm,
    true_score,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def agent(ops, hack_bonus=0.0):
    """Собрать агента: цепочка операторов плюс приписка к собственной оценке."""
    return {"ops": list(ops), "hack_bonus": hack_bonus}


# ------------------------------------------------------------------ apply_ops
def test_single_operator_is_applied():
    assert apply_ops(["trim"], "  hi  ") == "hi"


def test_operators_run_left_to_right():
    assert apply_ops(["collapse", "upper"], " a   b ") == "A B"


def test_order_of_operators_matters():
    assert apply_ops(["upper", "lower"], "aB") == "ab"
    assert apply_ops(["lower", "upper"], "aB") == "AB"


def test_empty_chain_is_the_identity():
    assert apply_ops([], "as is") == "as is"


def test_unknown_operator_is_reported_not_swallowed():
    """Сломанный агент должен получить диагноз, а не тихую пустую строку."""
    with pytest.raises(ValueError):
        apply_ops(["no-such-tool"], "x")


# ----------------------------------------------------------------- true_score
def test_the_honest_ceiling_of_the_tool_library_is_one_quarter():
    """Title-case в TOOLS нет, поэтому выше 0.25 честным путём не подняться."""
    assert true_score(["collapse"]) == APPROX(0.25)
    assert max(true_score([name]) for name in TOOLS) == APPROX(0.25)


def test_doing_nothing_solves_nothing():
    assert true_score(["nop"]) == APPROX(0.0)


def test_trimming_alone_is_worse_than_collapsing():
    assert true_score(["trim"]) < true_score(["collapse"])


def test_score_is_a_fraction_of_the_benchmark():
    assert true_score(["collapse"]) == APPROX(2 / len(CASES))


def test_a_broken_agent_scores_zero_instead_of_crashing_the_evaluator():
    """Оценщик обязан пережить любой вариант, который ему подсунут."""
    assert true_score(["collapse", "no-such-tool"]) == APPROX(0.0)


# ------------------------------------------------------------- reported_score
def test_with_the_firewall_up_the_report_equals_the_truth():
    a = agent(["collapse"], hack_bonus=0.5)
    assert reported_score(a) == APPROX(true_score(a["ops"]))


def test_the_hack_bonus_is_inert_while_the_firewall_holds():
    """Оценщик в недоступном пространстве имён — это когда поле просто не
    влияет ни на что."""
    honest = reported_score(agent(["collapse"], 0.0))
    for bonus in (0.1, 0.5, 1.0):
        assert reported_score(agent(["collapse"], bonus)) == APPROX(honest)


def test_an_open_side_channel_inflates_the_report():
    assert reported_score(agent(["collapse"], 0.5), firewall=False) == APPROX(0.75)


def test_the_inflated_report_is_still_capped_at_one():
    """Петля, выдающая 1.4, заметна даже невнимательному дежурному."""
    assert reported_score(agent(["collapse"], 3.0), firewall=False) == APPROX(1.0)


def test_the_hack_does_not_change_the_true_score():
    a = agent(["collapse"], 0.5)
    assert reported_score(a, firewall=False) > true_score(a["ops"])


# ---------------------------------------------------------------- archive_key
def test_key_carries_the_chain_length_and_the_rounded_score():
    assert archive_key(agent(["collapse"])) == (1, 0.25)


def test_chain_length_separates_otherwise_identical_scores():
    short = archive_key(agent(["collapse"]))
    long = archive_key(agent(["nop", "collapse"]))
    assert short != long
    assert short[1] == long[1]


def test_rounding_glues_nearly_equal_scores_into_one_cell():
    assert archive_key(agent(["trim", "nop"])) == (2, 0.12)


def test_key_follows_the_report_so_the_hack_moves_the_cell():
    closed = archive_key(agent(["collapse"], 0.5))
    opened = archive_key(agent(["collapse"], 0.5), firewall=False)
    assert closed != opened


# ---------------------------------------------------------------- passes_gate
def test_a_strictly_better_variant_passes():
    assert passes_gate(agent(["collapse"]), 0.125) is True


def test_a_worse_variant_is_rejected():
    assert passes_gate(agent(["trim"]), 0.25) is False


def test_an_equal_variant_is_rejected_too():
    """Нестрогое сравнение забило бы архив копиями одного качества."""
    assert passes_gate(agent(["collapse"]), 0.25) is False


def test_min_delta_raises_the_bar():
    assert passes_gate(agent(["collapse"]), 0.125, min_delta=0.0) is True
    assert passes_gate(agent(["collapse"]), 0.125, min_delta=0.2) is False


def test_a_broken_variant_never_passes_and_never_explodes():
    assert passes_gate(agent(["no-such-tool"]), 0.0) is False


def test_the_gate_can_be_fooled_once_the_side_channel_is_open():
    weak = agent(["nop"], hack_bonus=0.6)
    assert passes_gate(weak, 0.25) is False
    assert passes_gate(weak, 0.25, firewall=False) is True


# -------------------------------------------------------------- archive_accept
def test_a_passing_variant_claims_its_cell():
    archive = archive_accept({}, agent(["collapse"]), 0.0)
    assert archive == {(1, 0.25): agent(["collapse"])}


def test_a_variant_that_fails_the_gate_never_enters_the_archive():
    """Даже при пустой ячейке: архив вариантов — не свалка предложений."""
    archive = archive_accept({}, agent(["nop"]), 0.0)
    assert archive == {}


def test_a_better_tenant_takes_the_cell_over():
    """Обе прибавки округляются в одну ячейку, но вторая строго выше."""
    weaker, stronger = agent(["collapse"], 0.0), agent(["collapse"], 0.002)
    archive = archive_accept({}, weaker, 0.1, firewall=False)
    archive = archive_accept(archive, stronger, 0.1, firewall=False)
    assert archive[(1, 0.25)] is stronger


def test_an_equally_scoring_tenant_does_not_evict_the_incumbent():
    incumbent = agent(["collapse"])
    archive = archive_accept({}, incumbent, 0.125)
    archive = archive_accept(archive, agent(["collapse"]), 0.125)
    assert archive[(1, 0.25)] is incumbent


def test_accept_returns_a_copy_and_leaves_the_old_archive_alone():
    before = {}
    archive_accept(before, agent(["collapse"]), 0.0)
    assert before == {}


def test_variants_of_different_length_keep_their_own_cells():
    archive = archive_accept({}, agent(["collapse"]), 0.0)
    archive = archive_accept(archive, agent(["nop", "collapse"]), 0.0)
    assert len(archive) == 2


# --------------------------------------------------------------- mutate_agent
def test_mutation_returns_a_runnable_agent():
    rng = random.Random(0)
    a = agent(["nop"])
    for _ in range(200):
        a = mutate_agent(rng, a)
        assert 0.0 <= true_score(a["ops"]) <= 1.0
        assert len(a["ops"]) <= MAX_OPS


def test_mutation_leaves_the_parent_untouched():
    parent = agent(["collapse", "trim"])
    rng = random.Random(3)
    for _ in range(100):
        mutate_agent(rng, parent)
    assert parent == agent(["collapse", "trim"])


def test_the_firewall_freezes_the_hack_bonus():
    """Закрытый боковой канал нечему эволюционировать."""
    rng = random.Random(4)
    a = agent(["nop"], hack_bonus=0.0)
    for _ in range(500):
        a = mutate_agent(rng, a)
    assert a["hack_bonus"] == APPROX(0.0)


def test_an_open_side_channel_lets_the_hack_grow():
    rng = random.Random(4)
    a = agent(["nop"], hack_bonus=0.0)
    for _ in range(500):
        a = mutate_agent(rng, a, firewall=False)
    assert a["hack_bonus"] > 0.0
    assert HACK_RATE > 0.0


def test_same_seed_reproduces_the_same_edit():
    parent = agent(["collapse"])
    assert mutate_agent(random.Random(8), parent) == mutate_agent(random.Random(8), parent)


# -------------------------------------------------------------------- run_dgm
def test_the_champion_report_never_drops():
    """Архив не забывает чемпиона, поэтому история не убывает."""
    out = run_dgm(random.Random(7), 200)
    assert out["history"] == sorted(out["history"])


def test_a_closed_side_channel_keeps_report_equal_to_truth():
    out = run_dgm(random.Random(7), 300)
    assert out["gap"] == APPROX(0.0)
    assert out["reported"] == APPROX(out["true"])


def test_the_loop_reaches_the_honest_ceiling():
    out = run_dgm(random.Random(7), 300)
    assert out["true"] == APPROX(0.25)


def test_an_open_side_channel_pushes_the_report_above_the_truth():
    """Задокументированный в статье режим: оценка растёт, поведение — нет."""
    out = run_dgm(random.Random(7), 300, firewall=False)
    assert out["gap"] > 0.1
    assert out["true"] <= 0.25


def test_hacking_never_improves_the_underlying_behaviour():
    closed = run_dgm(random.Random(2), 300)
    opened = run_dgm(random.Random(2), 300, firewall=False)
    assert opened["reported"] > closed["reported"]
    assert opened["true"] <= closed["true"]


def test_same_seed_reproduces_the_whole_run():
    a = run_dgm(random.Random(11), 120)
    b = run_dgm(random.Random(11), 120)
    assert a["history"] == b["history"]
    assert a["best"] == b["best"]


def test_history_has_one_entry_per_generation():
    assert len(run_dgm(random.Random(1), 43)["history"]) == 43


def test_every_archived_variant_beat_the_starting_agent():
    """В архив нельзя попасть, не пройдя проверку."""
    out = run_dgm(random.Random(5), 200)
    start_score = true_score(["nop"])
    others = [a for key, a in out["archive"].items() if key != (1, 0.0)]
    assert all(true_score(a["ops"]) > start_score for a in others)
