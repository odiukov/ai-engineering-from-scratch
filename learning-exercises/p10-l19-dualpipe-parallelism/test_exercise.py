"""Тесты к уроку «DualPipe: расписание пайплайна без пузырей». Правь exercise.py."""

import pytest

from exercise import (
    bubble_fraction,
    bubble_slots,
    dualpipe_order,
    gpipe_order,
    makespan,
    one_f_one_b_order,
    peak_activation_memory,
    simulate_pipeline,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def kinds(queue):
    """Только вид операции: 'FFBB' читается глазами лучше, чем список кортежей."""
    return "".join(kind for kind, _, _ in queue)


# ------------------------------------------------------------- gpipe_order
def test_gpipe_runs_every_forward_before_any_backward():
    assert kinds(gpipe_order(3, 4)[0]) == "FFFFBBBB"


def test_gpipe_backward_order_is_reversed():
    """Последний вошёл — первый выходит: активации освобождаются с конца."""
    backward = [mb for kind, mb, _ in gpipe_order(2, 4)[0] if kind == "B"]
    assert backward == [3, 2, 1, 0]


def test_gpipe_gives_every_rank_the_same_queue():
    queues = gpipe_order(4, 3)
    assert len(queues) == 4
    assert all(q == queues[0] for q in queues)


# ------------------------------------------------------- one_f_one_b_order
def test_forwards_before_the_first_backward_shrink_towards_the_last_rank():
    """Ранг r успевает сделать P - r forward-ов, пока к нему идёт градиент."""
    queues = one_f_one_b_order(4, 6)
    lead = [len(kinds(q).split("B")[0]) for q in queues]
    assert lead == [4, 3, 2, 1]


def test_last_rank_alternates_forward_and_backward():
    assert kinds(one_f_one_b_order(4, 4)[3]) == "FBFBFBFB"


def test_one_f_one_b_does_the_same_amount_of_work_as_gpipe():
    """Расписание другое, работа та же: 2 * micro_batches операций на ранг."""
    for gp, ob in zip(gpipe_order(4, 6), one_f_one_b_order(4, 6)):
        assert len(gp) == len(ob) == 12
        assert sorted(gp) == sorted(ob)


# ---------------------------------------------------------- dualpipe_order
def test_dualpipe_uses_both_directions():
    directions = {d for queue in dualpipe_order(4, 8) for _, _, d in queue}
    assert directions == {1, -1}


def test_dualpipe_splits_micro_batches_in_half():
    """Первая половина номеров течёт вперёд, вторая — навстречу."""
    for _, mb, d in dualpipe_order(4, 8)[0]:
        assert d == (1 if mb < 4 else -1)


def test_dualpipe_rejects_an_odd_micro_batch_count():
    with pytest.raises(ValueError):
        dualpipe_order(4, 7)


def test_dualpipe_queue_holds_all_the_work():
    queue = dualpipe_order(4, 8)[0]
    assert len(queue) == 16
    assert {mb for _, mb, _ in queue} == set(range(8))


# -------------------------------------------------------- simulate_pipeline
def test_forward_walks_the_ranks_in_order():
    events = simulate_pipeline(one_f_one_b_order(4, 4))
    steps = {rank: step for rank, step, kind, mb, _ in events if kind == "F" and mb == 0}
    assert steps[0] < steps[1] < steps[2] < steps[3]


def test_a_rank_never_runs_two_operations_on_one_step():
    events = simulate_pipeline(dualpipe_order(4, 8))
    slots = [(rank, step) for rank, step, _, _, _ in events]
    assert len(slots) == len(set(slots))


def test_backward_waits_for_its_own_forward():
    events = simulate_pipeline(one_f_one_b_order(4, 4))
    when = {(rank, kind, mb): step for rank, step, kind, mb, _ in events}
    for rank, kind, mb in when:
        if kind == "B":
            assert when[(rank, "B", mb)] > when[(rank, "F", mb)]


def test_a_queue_with_a_missing_dependency_is_not_executable():
    """Backward без своего forward — это тупик, а не бесконечный цикл."""
    with pytest.raises(ValueError):
        simulate_pipeline([[("B", 0, 1)]])


# ----------------------------------------------------------------- makespan
def test_makespan_counts_steps_not_the_last_index():
    assert makespan([(0, 0, "F", 0, 1), (1, 1, "F", 0, 1)]) == 2


def test_makespan_of_an_empty_run_is_zero():
    assert makespan([]) == 0


# ------------------------------------------------------------- bubble_slots
def test_one_f_one_b_wastes_two_slots_per_extra_rank():
    """Классика: пузырь 1F1B равен 2 * (P - 1) и одинаков на всех рангах."""
    events = simulate_pipeline(one_f_one_b_order(4, 8))
    assert bubble_slots(events) == [6, 6, 6, 6]


def test_a_rank_that_finished_early_still_burns_gpu():
    """Последний ранг доделывает раньше всех, но простой считается до конца."""
    events = simulate_pipeline(one_f_one_b_order(4, 4))
    last_rank_finish = max(step for rank, step, _, _, _ in events if rank == 3)
    assert last_rank_finish < makespan(events) - 1
    assert bubble_slots(events)[3] == 6


# ---------------------------------------------------------- bubble_fraction
def test_bubble_fraction_falls_as_micro_batches_grow():
    small = bubble_fraction(simulate_pipeline(one_f_one_b_order(4, 4)))
    big = bubble_fraction(simulate_pipeline(one_f_one_b_order(4, 16)))
    assert 0 < big < small < 1


def test_gpipe_and_one_f_one_b_waste_exactly_the_same_time():
    """1F1B выигрывает память, а не время: пузырь у него ровно как у GPipe."""
    gp = bubble_fraction(simulate_pipeline(gpipe_order(4, 8)))
    ob = bubble_fraction(simulate_pipeline(one_f_one_b_order(4, 8)))
    assert gp == APPROX(ob)


def test_dualpipe_bubble_is_smaller_than_one_f_one_b():
    """Главное утверждение урока, на тех же P и micro_batches."""
    ob = simulate_pipeline(one_f_one_b_order(8, 16))
    dp = simulate_pipeline(dualpipe_order(8, 16))
    assert bubble_fraction(dp) < bubble_fraction(ob)
    assert makespan(dp) < makespan(ob)


def test_dualpipe_bubble_does_not_grow_with_micro_batches():
    """Пузырь не растёт от числа микробатчей — за это статью и хвалят."""
    few = bubble_slots(simulate_pipeline(dualpipe_order(8, 16)))
    many = bubble_slots(simulate_pipeline(dualpipe_order(8, 32)))
    assert few == many == [6] * 8


# --------------------------------------------------- peak_activation_memory
def test_gpipe_holds_every_micro_batch_at_once():
    assert peak_activation_memory(gpipe_order(4, 8)) == [8, 8, 8, 8]


def test_one_f_one_b_holds_at_most_pipeline_depth():
    """Ради этого 1F1B и живёт: пик не micro_batches, а глубина пайплайна."""
    assert peak_activation_memory(one_f_one_b_order(4, 8)) == [4, 3, 2, 1]


def test_the_last_rank_holds_a_single_micro_batch():
    assert peak_activation_memory(one_f_one_b_order(8, 32))[7] == 1


def test_dualpipe_pays_memory_for_its_smaller_bubble():
    """Два встречных потока — больше активаций в полёте, чем у 1F1B."""
    assert max(peak_activation_memory(dualpipe_order(4, 8))) > max(
        peak_activation_memory(one_f_one_b_order(4, 8))
    )
