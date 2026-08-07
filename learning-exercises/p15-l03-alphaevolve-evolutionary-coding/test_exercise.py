"""Тесты к уроку «AlphaEvolve: эволюционный поиск программ». Правь exercise.py."""

import itertools
import math
import random

import pytest

from exercise import (
    MAX_CONST_BUCKET,
    MAX_DEPTH,
    archive_insert,
    best_of,
    cell_key,
    depth,
    evaluate_expr,
    evolve,
    mse,
    mutate,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

X = ("x",)
QUADRATIC = lambda x: 2.0 * x * x + 3.0 * x - 1.0
IDENTITY = lambda x: x


# -------------------------------------------------------------- evaluate_expr
def test_constant_node_ignores_the_argument():
    assert evaluate_expr(("num", 2.0), 5.0) == APPROX(2.0)


def test_variable_node_returns_the_argument():
    assert evaluate_expr(X, 5.0) == APPROX(5.0)


def test_nested_program_is_evaluated_bottom_up():
    program = ("add", ("mul", ("num", 2.0), X), ("num", 1.0))
    assert evaluate_expr(program, 3.0) == APPROX(7.0)


def test_squaring_program_matches_x_times_x():
    assert evaluate_expr(("mul", X, X), 3.0) == APPROX(9.0)


def test_unknown_node_raises_instead_of_returning_garbage():
    """Оценщик, молча возвращающий мусор, — дырка, через которую утекает поиск."""
    with pytest.raises(ValueError):
        evaluate_expr(("div", X, X), 1.0)


# ---------------------------------------------------------------------- depth
def test_leaf_has_depth_one():
    assert depth(X) == 1
    assert depth(("num", 7.0)) == 1


def test_one_operator_gives_depth_two():
    assert depth(("add", X, ("num", 1.0))) == 2


def test_depth_follows_the_deeper_branch():
    assert depth(("mul", ("add", X, X), X)) == 3


def test_depth_grows_by_one_per_wrapping():
    program = X
    for expected in (2, 3, 4):
        program = ("add", program, ("num", 1.0))
        assert depth(program) == expected


# ------------------------------------------------------------------------ mse
def test_perfect_program_scores_zero():
    assert mse(X, [1.0, 2.0], IDENTITY) == APPROX(0.0)


def test_mse_averages_the_squared_errors():
    assert mse(("num", 0.0), [1.0, 3.0], IDENTITY) == APPROX(5.0)


def test_mse_is_symmetric_in_the_sign_of_the_error():
    over = mse(("num", 2.0), [1.0], IDENTITY)
    under = mse(("num", 0.0), [1.0], IDENTITY)
    assert over == APPROX(under)


def test_empty_point_set_is_not_a_free_perfect_score():
    """mse=0 на нуле точек — самый дешёвый способ обмануть петлю."""
    with pytest.raises(ValueError):
        mse(X, [], IDENTITY)


def test_overflowing_program_is_worst_not_fatal():
    huge = X
    for _ in range(9):
        huge = ("mul", huge, huge)
    assert mse(huge, [10.0], IDENTITY) == math.inf


# ------------------------------------------------------------------- cell_key
def test_bare_variable_lands_in_the_shallow_zero_constant_cell():
    assert cell_key(X) == (1, 0)


def test_large_constant_moves_the_program_to_another_bucket():
    assert cell_key(("num", 5.0)) == (1, 2)


def test_wrapping_a_program_changes_its_cell():
    assert cell_key(("add", X, ("num", 3.0))) == (2, 1)


def test_both_coordinates_are_clipped():
    """Без подрезки у каждой программы была бы личная клетка."""
    deep = X
    for _ in range(20):
        deep = ("add", deep, ("num", 99.0))
    assert cell_key(deep) == (MAX_DEPTH, MAX_CONST_BUCKET)


def test_sign_of_a_constant_does_not_change_its_bucket():
    assert cell_key(("num", -5.0)) == cell_key(("num", 5.0))


# --------------------------------------------------------------------- mutate
def test_mutation_always_produces_a_runnable_program():
    rng = random.Random(0)
    program = X
    for _ in range(200):
        program = mutate(rng, program)
        assert isinstance(evaluate_expr(program, 1.5), float)


def test_same_seed_reproduces_the_same_edit():
    assert mutate(random.Random(4), X) == mutate(random.Random(4), X)


def test_mutation_does_not_rewrite_the_parent():
    parent = ("add", X, ("num", 1.0))
    rng = random.Random(9)
    for _ in range(50):
        mutate(rng, parent)
    assert parent == ("add", X, ("num", 1.0))


def test_mutation_explores_more_than_one_neighbour():
    rng = random.Random(11)
    seen = {mutate(rng, X) for _ in range(100)}
    assert len(seen) > 3


# ------------------------------------------------------------- archive_insert
def test_first_program_claims_its_empty_cell():
    archive = archive_insert({}, X, 1.0)
    assert archive == {(1, 0): (X, 1.0)}


def test_worse_program_does_not_evict_the_incumbent():
    archive = archive_insert({}, X, 1.0)
    archive = archive_insert(archive, ("num", 1.0), 5.0)
    assert archive[(1, 0)] == (X, 1.0)


def test_better_program_takes_over_the_cell():
    archive = archive_insert({}, X, 5.0)
    archive = archive_insert(archive, ("num", 1.0), 1.0)
    assert archive[(1, 0)] == (("num", 1.0), 1.0)


def test_an_equally_good_program_leaves_the_incumbent_alone():
    """Строгое сравнение: иначе содержимое архива зависит от порядка вставок."""
    archive = archive_insert({}, X, 2.0)
    archive = archive_insert(archive, ("num", 1.0), 2.0)
    assert archive[(1, 0)] == (X, 2.0)


def test_insert_returns_a_new_archive_and_leaves_the_old_one_alone():
    before = archive_insert({}, X, 1.0)
    archive_insert(before, ("num", 5.0), 0.5)
    assert list(before) == [(1, 0)]


def test_programs_in_different_cells_coexist():
    archive = archive_insert({}, X, 1.0)
    archive = archive_insert(archive, ("num", 5.0), 9.0)
    assert len(archive) == 2


# -------------------------------------------------------------------- best_of
def test_champion_is_the_lowest_scoring_entry():
    archive = archive_insert(archive_insert({}, X, 3.0), ("num", 5.0), 1.0)
    assert best_of(archive) == (("num", 5.0), 1.0)


def test_ties_resolve_to_the_smaller_cell():
    archive = archive_insert(archive_insert({}, X, 2.0), ("num", 5.0), 2.0)
    assert best_of(archive) == (X, 2.0)


def test_champion_survives_any_insertion_order():
    """Элитизм не должен зависеть от того, в каком порядке подъехали кандидаты."""
    entries = [
        (X, 3.0),
        (("num", 1.0), 0.5),
        (("num", 5.0), 7.0),
        (("add", X, ("num", 3.0)), 2.0),
    ]
    champions = set()
    for order in itertools.permutations(entries):
        archive = {}
        for expr, score in order:
            archive = archive_insert(archive, expr, score)
        champions.add(best_of(archive))
    assert champions == {(("num", 1.0), 0.5)}


def test_empty_archive_has_no_champion():
    with pytest.raises(ValueError):
        best_of({})


# --------------------------------------------------------------------- evolve
def test_champion_score_never_gets_worse_across_generations():
    """Элитизм: архив помнит чемпиона, поэтому история не растёт."""
    out = evolve(random.Random(3), X, 300, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                 [-1.5, 0.5, 2.5], QUADRATIC)
    assert out["history"] == sorted(out["history"], reverse=True)


def test_evolution_improves_on_the_seed_program():
    train = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    holdout = [-1.5, 0.5, 2.5]
    seed_score = 0.5 * (mse(X, train, QUADRATIC) + mse(X, holdout, QUADRATIC))
    out = evolve(random.Random(3), X, 400, train, holdout, QUADRATIC)
    assert out["score"] < seed_score


def test_same_seed_reproduces_the_whole_run():
    args = (X, 150, [-1.0, 0.0, 1.0, 2.0], [0.5, 1.5], QUADRATIC)
    assert evolve(random.Random(21), *args) == evolve(random.Random(21), *args)


def test_holdout_run_scores_by_the_average_of_both_splits():
    train, holdout = [-1.0, 0.0, 1.0, 2.0], [0.5, 1.5]
    out = evolve(random.Random(5), X, 200, train, holdout, QUADRATIC)
    assert out["score"] == APPROX(0.5 * (out["train_mse"] + out["holdout_mse"]))


def test_without_a_holdout_the_search_signal_is_blind_to_it():
    """Петля буквально не видит отложенных точек — «score» это только train."""
    train, holdout = [-1.0, 0.0, 1.0, 2.0], [0.5, 1.5]
    out = evolve(random.Random(5), X, 200, train, holdout, QUADRATIC,
                 use_holdout=False)
    assert out["score"] == APPROX(out["train_mse"])


def test_a_train_only_evaluator_flatters_itself():
    """Две точки недоопределяют квадратичную цель: прямая ложится на них
    идеально, а на отложенных разъезжается. Это reward hacking в мягчайшей
    форме — оптимизируем измеримое, а не нужное."""
    train, holdout = [0.0, 1.0], [-2.0, 2.0, 3.0]
    out = evolve(random.Random(2), X, 800, train, holdout, QUADRATIC,
                 use_holdout=False)
    assert out["train_mse"] < 0.5
    assert out["gap"] > 1.0


def test_a_perfect_seed_is_not_ruined_by_the_loop():
    out = evolve(random.Random(1), X, 200, [0.0, 1.0, 2.0], [0.5], IDENTITY)
    assert out["train_mse"] == APPROX(0.0)
    assert out["holdout_mse"] == APPROX(0.0)


def test_history_has_one_entry_per_generation():
    out = evolve(random.Random(1), X, 37, [0.0, 1.0], [0.5], IDENTITY)
    assert len(out["history"]) == 37
