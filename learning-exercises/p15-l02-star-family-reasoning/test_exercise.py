"""Тесты к уроку «STaR, V-STaR, Quiet-STaR». Правь exercise.py."""

import random

import pytest

from exercise import (
    STRATEGIES,
    expected_accuracy,
    finetune,
    pick_strategy,
    rationalize,
    sample_trace,
    star_filter,
    star_round,
    vstar_select,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def trace(strategy, correct, sound):
    """Собрать обоснование руками — тестам не нужен rng, чтобы задать случай."""
    return {"strategy": strategy, "answer_correct": correct, "rationale_sound": sound}


# --------------------------------------------------------------- pick_strategy
def test_a_single_nonzero_weight_wins_every_draw():
    rng = random.Random(0)
    assert {pick_strategy(rng, {"sound": 1.0, "shortcut": 0.0})
            for _ in range(50)} == {"sound"}


def test_unnormalised_weights_are_accepted():
    """Таблица весов не обязана суммироваться в единицу."""
    rng = random.Random(1)
    assert pick_strategy(rng, {"sound": 7, "random": 0}) == "sound"


def test_draw_frequencies_follow_the_weights():
    rng = random.Random(2)
    draws = [pick_strategy(rng, {"sound": 3.0, "random": 1.0}) for _ in range(4000)]
    share = draws.count("sound") / len(draws)
    assert 0.70 < share < 0.80


def test_same_seed_reproduces_the_same_sequence():
    a = [pick_strategy(random.Random(5), {"sound": 1.0, "random": 1.0})
         for _ in range(1)]
    b = [pick_strategy(random.Random(5), {"sound": 1.0, "random": 1.0})
         for _ in range(1)]
    assert a == b


def test_insertion_order_of_the_table_does_not_change_the_draw():
    """Одна и та же таблица, собранная в другом порядке, обязана дать то же."""
    forward = {"random": 1.0, "shortcut": 1.0, "sound": 1.0}
    backward = {"sound": 1.0, "shortcut": 1.0, "random": 1.0}
    a = [pick_strategy(random.Random(s), forward) for s in range(30)]
    b = [pick_strategy(random.Random(s), backward) for s in range(30)]
    assert a == b


def test_zero_total_weight_is_rejected():
    with pytest.raises(ValueError):
        pick_strategy(random.Random(0), {"sound": 0.0})


# ---------------------------------------------------------------- sample_trace
def test_sound_strategy_always_lands_the_answer():
    rng = random.Random(3)
    traces = [sample_trace(rng, {"sound": 1.0}) for _ in range(200)]
    assert all(t["answer_correct"] for t in traces)
    assert all(t["rationale_sound"] for t in traces)


def test_shortcut_rationale_is_never_marked_sound():
    rng = random.Random(4)
    traces = [sample_trace(rng, {"shortcut": 1.0}) for _ in range(200)]
    assert not any(t["rationale_sound"] for t in traces)


def test_shortcut_hits_far_less_often_out_of_distribution():
    """Именно этот разрыв обучающая выборка увидеть не может."""
    rng = random.Random(6)
    on_id = [sample_trace(rng, {"shortcut": 1.0}) for _ in range(2000)]
    off_id = [sample_trace(rng, {"shortcut": 1.0}, on_ood=True) for _ in range(2000)]
    id_rate = sum(t["answer_correct"] for t in on_id) / len(on_id)
    ood_rate = sum(t["answer_correct"] for t in off_id) / len(off_id)
    assert id_rate > 4 * ood_rate


def test_sound_strategy_is_immune_to_the_distribution_shift():
    rng = random.Random(8)
    traces = [sample_trace(rng, {"sound": 1.0}, on_ood=True) for _ in range(200)]
    assert all(t["answer_correct"] for t in traces)


# ----------------------------------------------------------- expected_accuracy
def test_expected_accuracy_of_a_purely_sound_model_is_one():
    assert expected_accuracy({"sound": 1.0}) == APPROX(1.0)


def test_expected_accuracy_is_a_weighted_average():
    assert expected_accuracy({"sound": 1.0, "random": 1.0}) == APPROX(0.55)


def test_expected_accuracy_ignores_the_scale_of_the_weights():
    small = expected_accuracy({"sound": 1.0, "shortcut": 3.0})
    large = expected_accuracy({"sound": 100.0, "shortcut": 300.0})
    assert small == APPROX(large)


def test_shortcut_heavy_model_collapses_out_of_distribution():
    weights = {"sound": 0.2, "shortcut": 0.8}
    assert expected_accuracy(weights) > 0.5
    assert expected_accuracy(weights, on_ood=True) < 0.3


# --------------------------------------------------------------- star_filter
def test_filter_keeps_correct_answers():
    kept = star_filter([trace("sound", True, True)])
    assert len(kept) == 1


def test_filter_discards_a_sound_rationale_that_missed_the_answer():
    """Честное рассуждение с неверным ответом STaR выбрасывает — это и есть
    answer-conditioned gradient."""
    assert star_filter([trace("sound", False, True)]) == []


def test_filter_keeps_an_unsound_rationale_that_hit_the_answer():
    """И наоборот: срез, случайно попавший в ответ, остаётся в обучении."""
    kept = star_filter([trace("shortcut", True, False)])
    assert kept == [trace("shortcut", True, False)]


def test_filter_does_not_look_at_rationale_soundness_at_all():
    mixed = [trace("sound", False, True), trace("shortcut", True, False)]
    assert [t["strategy"] for t in star_filter(mixed)] == ["shortcut"]


def test_filter_does_not_mutate_its_input():
    traces = [trace("sound", True, True), trace("random", False, False)]
    star_filter(traces)
    assert len(traces) == 2


# ----------------------------------------------------------------- rationalize
def test_rationalization_targets_only_the_failures():
    out = rationalize([trace("sound", True, True), trace("random", False, False)])
    assert len(out) == 1


def test_rationalized_answer_is_correct_by_construction():
    out = rationalize([trace("random", False, False)])
    assert out[0]["answer_correct"] is True
    assert out[0]["strategy"] == "rationalized"


def test_rationalized_rationale_is_never_claimed_sound():
    """Обоснование, дописанное под известный ответ, честным не считается."""
    out = rationalize([trace("random", False, False)])
    assert out[0]["rationale_sound"] is False


def test_nothing_to_rationalize_when_everything_was_already_correct():
    assert rationalize([trace("sound", True, True)]) == []


def test_rationalization_rescues_problems_the_filter_alone_would_drop():
    traces = [trace("random", False, False) for _ in range(5)]
    assert star_filter(traces) == []
    assert len(rationalize(traces)) == 5


# -------------------------------------------------------------------- finetune
def test_finetune_returns_a_normalised_table():
    out = finetune({"sound": 3.0, "random": 1.0}, [trace("sound", True, True)])
    assert sum(out.values()) == APPROX(1.0)


def test_alpha_one_copies_the_empirical_distribution():
    out = finetune({"sound": 1.0, "random": 1.0},
                   [trace("sound", True, True)], alpha=1.0)
    assert out == {"sound": APPROX(1.0), "random": APPROX(0.0)}


def test_alpha_zero_keeps_the_prior_and_ignores_the_data():
    out = finetune({"sound": 1.0, "random": 1.0},
                   [trace("sound", True, True)], alpha=0.0)
    assert out == {"sound": APPROX(0.5), "random": APPROX(0.5)}


def test_a_strategy_absent_from_the_data_loses_weight():
    before = {"sound": 1.0, "shortcut": 1.0}
    after = finetune(before, [trace("sound", True, True)], alpha=0.6)
    assert after["shortcut"] < 0.5


def test_a_strategy_new_to_the_table_still_gets_weight():
    """Рационализация вводит стратегию, которой в прежней таблице не было."""
    after = finetune({"sound": 1.0}, [trace("rationalized", True, False)], alpha=0.6)
    assert after["rationalized"] > 0.0


def test_empty_training_set_leaves_the_model_where_it_was():
    after = finetune({"sound": 1.0, "random": 3.0}, [])
    assert after == {"sound": APPROX(0.25), "random": APPROX(0.75)}


# ------------------------------------------------------------------ star_round
def test_solved_fraction_matches_the_expected_accuracy_of_the_model():
    out = star_round(random.Random(11), {"sound": 0.2, "shortcut": 0.8},
                     n_samples=4000)
    assert out["solved_fraction"] == pytest.approx(0.52, abs=0.03)


def test_every_kept_trace_landed_the_answer():
    out = star_round(random.Random(12), {"sound": 0.3, "random": 0.7},
                     n_samples=500)
    assert all(t["answer_correct"] for t in out["kept"])


def test_a_round_never_lowers_in_distribution_accuracy():
    """Свойство петли: на своём распределении итерация не ухудшает результат."""
    before = {"sound": 0.2, "shortcut": 0.5, "random": 0.3}
    after = star_round(random.Random(13), before, n_samples=4000)["weights"]
    assert expected_accuracy(after) >= expected_accuracy(before)


def test_the_same_round_can_wreck_out_of_distribution_accuracy():
    """Срез проходит фильтр чаще, чем угадывание, но на чужом распределении
    работает ХУЖЕ него. Петля растит точность на своём и роняет на чужом."""
    before = {"shortcut": 0.5, "random": 0.5}
    after = star_round(random.Random(14), before, n_samples=6000)["weights"]
    assert after["shortcut"] > 0.5                       # срез усилен
    assert expected_accuracy(after) > expected_accuracy(before)
    assert expected_accuracy(after, on_ood=True) < expected_accuracy(before, on_ood=True)


def test_rationalization_enlarges_the_training_set():
    plain = star_round(random.Random(15), {"random": 1.0}, n_samples=300)
    with_hints = star_round(random.Random(15), {"random": 1.0}, n_samples=300,
                            use_rationalization=True)
    assert len(with_hints["kept"]) > len(plain["kept"])


def test_solved_fraction_is_measured_before_rationalization():
    """Иначе метрика всегда 1.0 и ничего не измеряет."""
    out = star_round(random.Random(16), {"random": 1.0}, n_samples=300,
                     use_rationalization=True)
    assert out["solved_fraction"] < 0.5
    assert len(out["kept"]) == 300


# ---------------------------------------------------------------- vstar_select
def test_verifier_picks_the_highest_scoring_rationale():
    a, b = trace("random", False, False), trace("sound", True, True)
    chosen = vstar_select([a, b], lambda t: 1.0 if t["rationale_sound"] else 0.0)
    assert chosen is b


def test_ties_are_broken_deterministically_by_first_seen():
    """Best-of-N с плавающим разрешением ничьих невоспроизводим."""
    a, b, c = trace("s1", True, True), trace("s2", True, True), trace("s3", True, True)
    assert vstar_select([a, b, c], lambda t: 0.5) is a
    assert vstar_select([c, b, a], lambda t: 0.5) is c


def test_a_confident_verifier_can_prefer_a_polished_shortcut():
    """Верификатор обучен на тех же метках и умеет ошибаться уверенно."""
    honest_miss = trace("sound", False, True)
    lucky_shortcut = trace("shortcut", True, False)
    chosen = vstar_select([honest_miss, lucky_shortcut],
                          lambda t: 1.0 if t["answer_correct"] else 0.0)
    assert chosen["rationale_sound"] is False


def test_selection_over_one_candidate_returns_it():
    only = trace("sound", True, True)
    assert vstar_select([only], lambda t: 0.0) is only


def test_empty_candidate_list_is_rejected():
    with pytest.raises(ValueError):
        vstar_select([], lambda t: 1.0)


# ------------------------------------------------------------------ STRATEGIES
def test_shortcut_and_sound_are_indistinguishable_by_answer_alone_in_training():
    """Таблица стратегий обязана держать ловушку урока: срез бьёт по ответу
    достаточно часто, чтобы фильтр STaR его сохранял."""
    assert STRATEGIES["shortcut"]["id"] > STRATEGIES["random"]["id"]
    assert STRATEGIES["shortcut"]["ood"] < STRATEGIES["shortcut"]["id"]
    kept = star_filter([trace("shortcut", True, False)])
    assert kept and kept[0]["strategy"] == "shortcut"
