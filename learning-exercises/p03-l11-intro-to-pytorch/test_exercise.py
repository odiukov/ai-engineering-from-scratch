"""Тесты к уроку «Знакомство с PyTorch: собираем autograd руками». Правь exercise.py."""

import pytest

from exercise import Linear, Module, Tensor, fit_line, mse_loss, randn, sgd_step

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


# ------------------------------------------------------------------ Tensor
def test_tensor_shape_is_rows_by_columns():
    assert Tensor([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)


def test_tensor_copies_the_incoming_rows():
    """Иначе два тензора поделят один список и правка одного испортит другой."""
    source = [[1.0, 2.0]]
    t = Tensor(source)
    source[0][0] = 99.0
    assert flat(t.data) == APPROX([1.0, 2.0])


def test_zero_grad_clears_the_gradient():
    t = Tensor([[2.0]], requires_grad=True)
    t.mul(t).backward()
    t.zero_grad()
    assert flat(t.grad) == APPROX([0.0])


# --------------------------------------------------------------- Tensor.add
def test_add_is_elementwise():
    out = Tensor([[1.0, 2.0]]).add(Tensor([[10.0, 20.0]]))
    assert flat(out.data) == APPROX([11.0, 22.0])


def test_add_broadcasts_a_single_row_over_the_batch():
    """Так работает bias в nn.Linear: одна строка на весь батч."""
    out = Tensor([[1.0, 2.0], [3.0, 4.0]]).add(Tensor([[10.0, 20.0]]))
    assert flat(out.data) == APPROX([11.0, 22.0, 13.0, 24.0])


def test_add_passes_the_gradient_to_both_operands_unchanged():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)
    a.add(b).backward()
    assert flat(a.grad) == APPROX([1.0, 1.0])
    assert flat(b.grad) == APPROX([1.0, 1.0])


def test_broadcast_bias_collects_gradient_from_every_row():
    """Строка использовалась дважды — значит и градиентов ей приходит два."""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = Tensor([[10.0, 20.0]], requires_grad=True)
    x.add(bias).backward()
    assert flat(bias.grad) == APPROX([2.0, 2.0])


# --------------------------------------------------------------- Tensor.mul
def test_mul_by_a_number_scales_every_element():
    assert flat(Tensor([[1.0, 2.0]]).mul(3.0).data) == APPROX([3.0, 6.0])


def test_mul_by_a_tensor_is_elementwise():
    out = Tensor([[1.0, 2.0]]).mul(Tensor([[4.0, 5.0]]))
    assert flat(out.data) == APPROX([4.0, 10.0])


def test_mul_gradient_of_each_factor_is_the_other_factor():
    a = Tensor([[2.0, 3.0]], requires_grad=True)
    b = Tensor([[5.0, 7.0]], requires_grad=True)
    a.mul(b).backward()
    assert flat(a.grad) == APPROX([5.0, 7.0])
    assert flat(b.grad) == APPROX([2.0, 3.0])


def test_squaring_a_tensor_gives_gradient_two_x():
    """Ромб на ленте: x встречается дважды, вклады обязаны сложиться."""
    t = Tensor([[3.0]], requires_grad=True)
    t.mul(t).backward()
    assert flat(t.grad) == APPROX([6.0])


def test_mul_by_a_number_scales_the_gradient():
    t = Tensor([[1.0, 2.0]], requires_grad=True)
    t.mul(-4.0).backward()
    assert flat(t.grad) == APPROX([-4.0, -4.0])


# ------------------------------------------------------------ Tensor.matmul
def test_matmul_shape_is_rows_of_left_by_columns_of_right():
    out = Tensor([[1.0, 2.0], [3.0, 4.0]]).matmul(Tensor([[1.0], [1.0]]))
    assert out.shape == (2, 1)


def test_matmul_value_is_the_dot_product():
    assert flat(Tensor([[1.0, 2.0]]).matmul(Tensor([[1.0], [1.0]])).data) == APPROX([3.0])


def test_matmul_gradient_of_the_left_operand_is_the_right_transposed():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[10.0], [20.0]], requires_grad=True)
    a.matmul(b).backward()
    assert flat(a.grad) == APPROX([10.0, 20.0])


def test_matmul_gradient_of_the_right_operand_is_the_left_transposed():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[10.0], [20.0]], requires_grad=True)
    a.matmul(b).backward()
    assert flat(b.grad) == APPROX([1.0, 2.0])


# ---------------------------------------------------------- Tensor.backward
def test_backward_seeds_the_root_with_ones():
    """Стартовый градиент — единицы, то есть это t.sum().backward()."""
    t = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    out = t.mul(1.0)
    out.backward()
    assert flat(out.grad) == APPROX([1.0, 1.0, 1.0])


def test_backward_walks_a_chain_of_three_operations():
    t = Tensor([[2.0]], requires_grad=True)
    # ((t * 3) + t) * t  =  4t^2, производная 8t = 16 при t = 2
    y = t.mul(3.0).add(t).mul(t)
    y.backward()
    assert flat(t.grad) == APPROX([16.0])


# ------------------------------------------------------------------ Module
def test_module_forward_is_not_implemented():
    """Контраст: база отказывается считать, рабочий подкласс — считает.

    Одной первой половины мало — она проходит и на пустой заготовке, где
    ещё ничего не написано.
    """
    with pytest.raises(NotImplementedError):
        Module().forward(Tensor([[1.0]]))
    assert len(Linear(2, 3, seed=0).parameters()) == 2


def test_linear_registers_weight_and_bias_automatically():
    """Ради этого в PyTorch и существует nn.Module: веса искать не надо."""
    assert len(Linear(2, 3, seed=0).parameters()) == 2


def test_parameters_recurse_into_nested_modules():
    class Net(Module):
        def __init__(self):
            self.first = Linear(2, 4, seed=0)
            self.second = Linear(4, 1, seed=1)

    assert len(Net().parameters()) == 4


def test_parameters_skip_tensors_without_requires_grad():
    class WithBuffer(Module):
        def __init__(self):
            self.layer = Linear(2, 2, seed=0)
            self.running_mean = Tensor([[0.0, 0.0]])

    assert len(WithBuffer().parameters()) == 2


def test_module_zero_grad_clears_every_parameter():
    layer = Linear(2, 2, seed=0)
    layer.forward(Tensor([[1.0, 1.0]])).backward()
    layer.zero_grad()
    assert flat(layer.weight.grad) == APPROX([0.0] * 4)


# ------------------------------------------------------------------ Linear
def test_linear_weight_shape_is_in_by_out():
    assert Linear(2, 3, seed=0).weight.shape == (2, 3)


def test_linear_output_shape_keeps_the_batch():
    out = Linear(2, 3, seed=0).forward(Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert out.shape == (2, 3)


def test_linear_bias_starts_at_zero():
    assert flat(Linear(4, 3, seed=0).bias.data) == APPROX([0.0, 0.0, 0.0])


def test_linear_is_reproducible_for_the_same_seed():
    a = Linear(3, 2, seed=9).forward(Tensor([[1.0, 2.0, 3.0]]))
    b = Linear(3, 2, seed=9).forward(Tensor([[1.0, 2.0, 3.0]]))
    assert flat(a.data) == APPROX(flat(b.data))


# ------------------------------------------------------------------- randn
def test_randn_is_reproducible_and_seed_dependent():
    assert len(randn(2, 3, seed=0)) == 2
    assert len(randn(2, 3, seed=0)[0]) == 3
    assert flat(randn(4, 4, seed=1)) == APPROX(flat(randn(4, 4, seed=1)))
    assert flat(randn(4, 4, seed=1)) != APPROX(flat(randn(4, 4, seed=2)))


# ---------------------------------------------------------------- mse_loss
def test_mse_loss_returns_squared_errors_elementwise():
    assert flat(mse_loss(Tensor([[3.0]]), Tensor([[1.0]])).data) == APPROX([4.0])


def test_mse_loss_gradient_is_twice_the_error():
    predicted = Tensor([[3.0]], requires_grad=True)
    mse_loss(predicted, Tensor([[1.0]])).backward()
    assert flat(predicted.grad) == APPROX([4.0])


# ---------------------------------------------------------------- sgd_step
def test_sgd_step_moves_against_the_gradient():
    t = Tensor([[1.0]], requires_grad=True)
    t.mul(2.0).backward()
    sgd_step([t], lr=0.1)
    assert flat(t.data) == APPROX([0.8])


def test_sgd_step_does_nothing_when_the_gradient_is_zero():
    t = Tensor([[5.0]], requires_grad=True)
    sgd_step([t], lr=100.0)
    assert flat(t.data) == APPROX([5.0])


# ---------------------------------------------------------------- fit_line
def test_fit_line_recovers_the_slope_and_the_intercept():
    w, b = fit_line()
    assert w == pytest.approx(3.0, abs=0.05)
    assert b == pytest.approx(-1.0, abs=0.05)


def test_fit_line_follows_a_different_target_line():
    w, b = fit_line(slope=-2.0, intercept=5.0)
    assert w == pytest.approx(-2.0, abs=0.05)
    assert b == pytest.approx(5.0, abs=0.05)


def test_fit_line_is_reproducible():
    assert fit_line(seed=4) == APPROX(fit_line(seed=4))
