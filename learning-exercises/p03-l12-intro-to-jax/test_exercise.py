"""Тесты к уроку «Знакомство с JAX: функциональный стиль руками». Правь exercise.py."""

import pytest

from exercise import (
    grad,
    mse,
    normal,
    predict,
    prng_key,
    split_key,
    train_linear,
    tree_map,
    value_and_grad,
    vmap,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
NUMERIC = lambda x: pytest.approx(x, abs=1e-6)


# --------------------------------------------------------------- prng_key
def test_prng_key_is_deterministic():
    assert prng_key(0) == prng_key(0)


def test_different_seeds_give_different_keys():
    assert prng_key(0) != prng_key(1)


# -------------------------------------------------------------- split_key
def test_split_key_returns_the_requested_number_of_keys():
    assert len(split_key(prng_key(0), 3)) == 3


def test_split_key_children_are_all_distinct():
    """Один ключ дважды — два «независимых» слоя с одинаковыми весами."""
    children = split_key(prng_key(0), 4)
    assert len(set(children)) == 4


def test_split_key_never_returns_the_parent():
    parent = prng_key(0)
    assert parent not in split_key(parent, 4)


def test_split_key_is_reproducible():
    assert split_key(prng_key(7), 3) == split_key(prng_key(7), 3)


# ----------------------------------------------------------------- normal
def test_normal_returns_the_requested_length():
    assert len(normal(prng_key(0), 5)) == 5


def test_normal_is_pure_and_repeatable():
    """Тот же ключ — те же числа. Глобальный random здесь всё ломает."""
    assert normal(prng_key(0), 4) == APPROX(normal(prng_key(0), 4))


def test_normal_depends_on_the_key():
    assert normal(prng_key(0), 4) != APPROX(normal(prng_key(1), 4))


def test_normal_scale_controls_the_spread():
    small = normal(prng_key(3), 400, scale=0.1)
    big = normal(prng_key(3), 400, scale=10.0)
    assert max(abs(v) for v in small) < max(abs(v) for v in big)


# --------------------------------------------------------------- tree_map
def test_tree_map_over_a_flat_list():
    assert tree_map(lambda v: v * 2, [1.0, 2.0]) == APPROX([2.0, 4.0])


def test_tree_map_walks_two_trees_in_parallel():
    assert tree_map(lambda a, b: a + b, [1.0, 2.0], [10.0, 20.0]) == APPROX([11.0, 22.0])


def test_tree_map_keeps_the_dict_structure():
    out = tree_map(lambda v: -v, {"w": [1.0, 2.0], "b": 3.0})
    assert out == {"w": pytest.approx([-1.0, -2.0]), "b": pytest.approx(-3.0)}


def test_tree_map_handles_nesting():
    out = tree_map(lambda v: v + 1, {"layer": {"w": [[1.0], [2.0]]}})
    assert out["layer"]["w"] == [pytest.approx([2.0]), pytest.approx([3.0])]


def test_tree_map_is_the_whole_optimizer():
    """Обновление всех весов сразу — одна строка, без .parameters()."""
    params = {"w": [1.0, 2.0], "b": 0.5}
    grads = {"w": [10.0, 20.0], "b": 5.0}
    updated = tree_map(lambda p, g: p - 0.1 * g, params, grads)
    assert updated["w"] == pytest.approx([0.0, 0.0])
    assert updated["b"] == pytest.approx(0.0)


def test_tree_map_rejects_different_sequence_lengths_instead_of_truncating():
    with pytest.raises(ValueError, match="structure mismatch"):
        tree_map(lambda a, b: a + b, [1.0, 2.0], [10.0])


def test_tree_map_rejects_different_dictionary_keys():
    with pytest.raises(ValueError, match="structure mismatch"):
        tree_map(lambda a, b: a + b, {"w": 1.0}, {"b": 2.0})


def test_tree_map_rejects_a_leaf_where_the_other_tree_has_a_container():
    with pytest.raises(ValueError, match="structure mismatch"):
        tree_map(lambda a, b: a + b, {"w": 1.0}, {"w": [2.0]})


# ------------------------------------------------------------------- grad
def test_grad_of_a_square():
    assert grad(lambda p: p[0] ** 2)([3.0]) == NUMERIC([6.0])


def test_grad_of_a_product_is_the_other_factor():
    assert grad(lambda p: p[0] * p[1])([2.0, 5.0]) == NUMERIC([5.0, 2.0])


def test_grad_of_a_constant_is_zero():
    assert grad(lambda p: 7.0)([1.0, 2.0]) == NUMERIC([0.0, 0.0])


def test_grad_returns_a_function_not_a_number():
    """jax.grad превращает функцию в функцию — её можно передать дальше."""
    df = grad(lambda p: p[0] ** 3)
    assert callable(df)
    assert df([2.0]) == pytest.approx([12.0], abs=1e-4)


def test_grad_does_not_mutate_the_point():
    point = [1.0, 2.0]
    grad(lambda p: p[0] * p[1])(point)
    assert point == APPROX([1.0, 2.0])


def test_grad_uses_the_central_difference():
    """Односторонняя разность на f = x^3 промахнётся сильнее допуска."""
    assert grad(lambda p: p[0] ** 3)([1.0]) == pytest.approx([3.0], abs=1e-8)


# -------------------------------------------------------- value_and_grad
def test_value_and_grad_returns_both():
    value, gradient = value_and_grad(lambda p: p[0] ** 2)([3.0])
    assert value == APPROX(9.0)
    assert gradient == NUMERIC([6.0])


def test_value_and_grad_agrees_with_grad():
    f = lambda p: p[0] ** 2 + 3 * p[1]
    _, gradient = value_and_grad(f)([1.0, 2.0])
    assert gradient == NUMERIC(grad(f)([1.0, 2.0]))


# ------------------------------------------------------------------- vmap
def test_vmap_maps_a_scalar_function_over_a_batch():
    assert vmap(lambda x: x * 2)([1.0, 2.0, 3.0]) == APPROX([2.0, 4.0, 6.0])


def test_vmap_maps_a_vector_function_over_a_batch():
    assert vmap(lambda row: sum(row))([[1.0, 2.0], [3.0, 4.0]]) == APPROX([3.0, 7.0])


def test_vmap_of_an_empty_batch_is_empty():
    assert vmap(lambda x: x * 2)([]) == []


def test_vmap_composes_with_grad_for_per_example_gradients():
    """Пер-пример градиенты одной строкой — то, ради чего JAX и берут."""
    per_example = vmap(grad(lambda p: p[0] ** 2))
    flat = [g for row in per_example([[1.0], [2.0], [3.0]]) for g in row]
    assert flat == NUMERIC([2.0, 4.0, 6.0])


# ---------------------------------------------------------------- predict
def test_predict_is_a_dot_product_plus_bias():
    assert predict({"w": [2.0, 3.0], "b": 1.0}, [1.0, 1.0]) == APPROX(6.0)


# -------------------------------------------------------------------- mse
def test_mse_of_a_perfect_model_is_zero():
    assert mse({"w": [1.0], "b": 0.0}, [[1.0], [2.0]], [1.0, 2.0]) == APPROX(0.0)


def test_mse_averages_the_squared_errors():
    assert mse({"w": [0.0], "b": 0.0}, [[1.0], [1.0]], [1.0, 3.0]) == APPROX(5.0)


# ----------------------------------------------------------- train_linear
def test_train_linear_recovers_a_one_feature_line():
    params = train_linear(prng_key(0), [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0])
    assert params["w"][0] == pytest.approx(2.0, abs=0.01)
    assert params["b"] == pytest.approx(1.0, abs=0.01)


def test_train_linear_recovers_two_features():
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    ys = [2.0, 3.0, 5.0, 7.0]
    params = train_linear(prng_key(0), xs, ys)
    assert params["w"] == pytest.approx([2.0, 3.0], abs=0.02)


def test_train_linear_does_not_touch_its_inputs():
    """Чистота: входные данные обязаны остаться ровно такими же."""
    xs = [[1.0], [2.0]]
    ys = [3.0, 5.0]
    train_linear(prng_key(0), xs, ys, steps=20)
    assert xs == [[1.0], [2.0]]
    assert ys == APPROX([3.0, 5.0])


def test_train_linear_is_reproducible_for_the_same_key():
    xs, ys = [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0]
    first = train_linear(prng_key(2), xs, ys, steps=30)
    second = train_linear(prng_key(2), xs, ys, steps=30)
    assert first["w"] == APPROX(second["w"])
    assert first["b"] == APPROX(second["b"])


def test_train_linear_actually_reduces_the_loss():
    xs, ys = [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0]
    key = prng_key(5)
    start = {"w": normal(key, 1, scale=0.1), "b": 0.0}
    trained = train_linear(key, xs, ys, steps=100)
    assert mse(trained, xs, ys) < mse(start, xs, ys)
