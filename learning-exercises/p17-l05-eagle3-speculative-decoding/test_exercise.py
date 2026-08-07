"""Тесты к уроку «EAGLE-3: спекулятивное декодирование». Правь exercise.py."""

import random

import pytest

from exercise import (
    ALPHA_GATE,
    BAD_DRAFT,
    GOOD_DRAFT,
    TARGET_PROBS,
    TRAFFIC_MIX,
    blended_alpha,
    breakeven_alpha,
    expected_speedup,
    normalize,
    residual_distribution,
    run_speculative,
    sample_index,
    speculative_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SEED = 7
DIST_STEPS = 20000       # столько шагов хватает, чтобы отличить 0.5 от 0.51
DIST_TOL = 0.015         # втрое больше наблюдаемого разброса на этом seed


def maxdev(a, b):
    """Максимальное покоординатное расхождение двух распределений."""
    return max(abs(x - y) for x, y in zip(a, b))


# --------------------------------------------------------- expected_speedup
def test_expected_speedup_reproduces_the_worked_example_from_the_lesson():
    assert expected_speedup(0.7, 5, 0.1) == pytest.approx(4.5 / 1.1)


def test_a_draft_that_is_never_accepted_is_pure_loss():
    """alpha = 0: за форвард всё тот же один токен, но накладные уже оплачены."""
    assert expected_speedup(0.0, 5, 0.1) < 1.0


def test_speedup_grows_with_the_draft_length_at_a_fixed_alpha():
    assert expected_speedup(0.7, 8, 0.1) > expected_speedup(0.7, 5, 0.1)


def test_verify_overhead_eats_the_speedup():
    """Тот же alpha на высокой конкурентности стоит дороже — растёт epsilon."""
    assert expected_speedup(0.7, 5, 0.5) < expected_speedup(0.7, 5, 0.1)


def test_expected_speedup_refuses_an_impossible_acceptance_rate():
    with pytest.raises(ValueError):
        expected_speedup(1.5, 5, 0.1)


# ---------------------------------------------------------- breakeven_alpha
def test_breakeven_alpha_matches_the_lesson_arithmetic():
    assert breakeven_alpha(5, 0.15) == pytest.approx(0.03)


def test_at_the_breakeven_alpha_the_speedup_is_exactly_one():
    """Определение точки безубыточности, проверенное через саму формулу."""
    for k, eps in ((5, 0.15), (8, 0.4), (4, 0.02)):
        assert expected_speedup(breakeven_alpha(k, eps), k, eps) == APPROX(1.0)


def test_a_breakeven_above_one_means_the_overhead_can_never_be_repaid():
    assert breakeven_alpha(5, 10.0) > 1.0
    assert expected_speedup(1.0, 5, 10.0) < 1.0


# ------------------------------------------------------------ blended_alpha
def test_blended_alpha_of_the_lesson_traffic_mix():
    assert blended_alpha(TRAFFIC_MIX) == pytest.approx(0.61)


def test_the_blend_passes_the_gate_while_a_segment_inside_it_loses():
    """Ровно поэтому alpha меряют по сегментам, а не одним числом на весь прод."""
    blend = blended_alpha(TRAFFIC_MIX)
    code_alpha = TRAFFIC_MIX[1][1]
    assert blend > ALPHA_GATE > code_alpha
    assert expected_speedup(code_alpha, 5, 0.15) < expected_speedup(blend, 5, 0.15)


def test_blended_alpha_refuses_shares_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        blended_alpha(((0.7, 0.7), (0.7, 0.4)))


# ----------------------------------------------------------------- normalize
def test_normalize_turns_weights_into_a_distribution():
    assert normalize([1, 3]) == APPROX([0.25, 0.75])


def test_normalize_keeps_a_zero_weight_at_exactly_zero():
    probs = normalize([0.4, 0.2, 0.0])
    assert probs[2] == 0.0
    assert sum(probs) == APPROX(1.0)


def test_normalize_refuses_all_zero_weights():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0])


def test_normalize_refuses_a_negative_weight():
    with pytest.raises(ValueError):
        normalize([1.0, -0.5])


# -------------------------------------------------------------- sample_index
def test_sample_index_always_returns_the_only_possible_token():
    rng = random.Random(SEED)
    assert {sample_index([0.0, 1.0, 0.0], rng) for _ in range(50)} == {1}


def test_sample_index_never_returns_a_zero_probability_token():
    """Ловушка `r <= acc`: с ней нулевой токен в начале списка иногда выпадает."""
    rng = random.Random(SEED)
    drawn = {sample_index([0.0, 0.5, 0.5], rng) for _ in range(5000)}
    assert 0 not in drawn


def test_sample_index_follows_the_distribution_it_is_given():
    rng = random.Random(SEED)
    counts = [0, 0, 0]
    for _ in range(20000):
        counts[sample_index([0.5, 0.3, 0.2], rng)] += 1
    assert maxdev(normalize(counts), [0.5, 0.3, 0.2]) < DIST_TOL


# ------------------------------------------------------ residual_distribution
def test_residual_subtracts_per_token_not_by_a_scalar_shift():
    assert residual_distribution([0.5, 0.3, 0.2], [0.1, 0.1, 0.8]) == APPROX(
        [2 / 3, 1 / 3, 0.0]
    )


def test_residual_gives_exactly_zero_mass_where_the_draft_overproposes():
    """Токен, который черновик тянул чаще цели, уже оплачен на принятии."""
    residual = residual_distribution(list(TARGET_PROBS), list(BAD_DRAFT))
    assert residual[3] == 0.0
    assert sum(residual) == APPROX(1.0)


def test_residual_leans_towards_what_the_draft_underproposes():
    residual = residual_distribution(list(TARGET_PROBS), list(BAD_DRAFT))
    assert residual[0] > TARGET_PROBS[0]


def test_residual_of_a_perfect_draft_is_empty():
    """Черновик равен цели — отказов не бывает, и остатку взяться неоткуда."""
    with pytest.raises(ValueError):
        residual_distribution(list(TARGET_PROBS), list(TARGET_PROBS))


# ----------------------------------------------------------- speculative_step
def test_a_perfect_draft_accepts_every_proposal_and_emits_the_bonus_token():
    step = speculative_step(list(TARGET_PROBS), list(TARGET_PROBS), 5, random.Random(SEED))
    assert step["accepted"] == 5
    assert len(step["tokens"]) == 6


def test_every_step_emits_exactly_one_more_token_than_it_accepted():
    """Инвариант, из которого и вырастает формула 1 + k * alpha."""
    rng = random.Random(SEED)
    for _ in range(300):
        step = speculative_step(list(TARGET_PROBS), list(BAD_DRAFT), 5, rng)
        assert len(step["tokens"]) == step["accepted"] + 1


def test_drafted_counts_the_requested_length_not_the_verified_positions():
    """Все k черновых токенов посчитаны одним проходом и оплачены даже при отказе."""
    rng = random.Random(SEED)
    steps = [speculative_step(list(TARGET_PROBS), list(BAD_DRAFT), 5, rng) for _ in range(50)]
    assert all(s["drafted"] == 5 for s in steps)
    assert any(s["accepted"] < 5 for s in steps)


def test_speculative_step_refuses_a_draft_over_a_different_vocabulary():
    with pytest.raises(ValueError):
        speculative_step([0.5, 0.5], [0.3, 0.3, 0.4], 3, random.Random(SEED))


# ----------------------------------------------------------- run_speculative
def test_a_bad_draft_still_reproduces_the_target_distribution():
    """Главная гарантия метода: плохой черновик стоит скорости, но не качества."""
    run = run_speculative(
        list(TARGET_PROBS), list(BAD_DRAFT), 5, DIST_STEPS, random.Random(SEED)
    )
    assert run["acceptance_rate"] < 0.15
    assert maxdev(run["distribution"], TARGET_PROBS) < DIST_TOL


def test_a_good_draft_gives_the_same_distribution_only_faster():
    good = run_speculative(
        list(TARGET_PROBS), list(GOOD_DRAFT), 5, DIST_STEPS, random.Random(SEED)
    )
    bad = run_speculative(
        list(TARGET_PROBS), list(BAD_DRAFT), 5, DIST_STEPS, random.Random(SEED)
    )
    assert maxdev(good["distribution"], TARGET_PROBS) < DIST_TOL
    assert good["acceptance_rate"] > bad["acceptance_rate"]


def test_tokens_per_forward_matches_the_closed_form_exactly():
    """Не приближённо, а тождественно — эмиссия равна accepted + 1 на каждом шаге."""
    for draft in (BAD_DRAFT, GOOD_DRAFT, TARGET_PROBS):
        run = run_speculative(list(TARGET_PROBS), list(draft), 5, 500, random.Random(SEED))
        assert run["tokens_per_forward"] == APPROX(1 + 5 * run["acceptance_rate"])


def test_a_perfect_draft_accepts_everything():
    run = run_speculative(
        list(TARGET_PROBS), list(TARGET_PROBS), 5, 500, random.Random(SEED)
    )
    assert run["acceptance_rate"] == APPROX(1.0)
    assert run["tokens_per_forward"] == APPROX(6.0)


def test_the_same_overhead_turns_a_bad_draft_negative_and_a_good_one_positive():
    """Тот же verify_overhead: решает только alpha, ничего кроме."""
    bad = run_speculative(
        list(TARGET_PROBS), list(BAD_DRAFT), 5, 2000, random.Random(SEED), 0.5
    )
    good = run_speculative(
        list(TARGET_PROBS), list(GOOD_DRAFT), 5, 2000, random.Random(SEED), 0.5
    )
    assert bad["speedup"] < 1.0 < good["speedup"]


def test_a_measured_alpha_below_breakeven_means_the_run_lost_time():
    bad = run_speculative(
        list(TARGET_PROBS), list(BAD_DRAFT), 5, 2000, random.Random(SEED), 0.5
    )
    assert bad["acceptance_rate"] < breakeven_alpha(5, 0.5)
    assert bad["speedup"] < 1.0


def test_the_same_seed_reproduces_the_whole_run():
    args = (list(TARGET_PROBS), list(GOOD_DRAFT), 5, 300)
    first = run_speculative(*args, random.Random(SEED))
    second = run_speculative(*args, random.Random(SEED))
    assert first == second


def test_run_speculative_refuses_zero_steps():
    with pytest.raises(ValueError):
        run_speculative(list(TARGET_PROBS), list(GOOD_DRAFT), 5, 0, random.Random(SEED))
