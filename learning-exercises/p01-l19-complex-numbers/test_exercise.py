"""Тесты к уроку «Комплексные числа». Правь exercise.py."""

import math

import pytest

from exercise import (
    c_abs,
    c_conj,
    c_div,
    c_mul,
    c_pow,
    from_polar,
    roots_of_unity,
    to_polar,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ANGLE = lambda x: pytest.approx(x, abs=1e-5)

I = (0.0, 1.0)
ONE = (1.0, 0.0)


def flat(pairs):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for pair in pairs for x in pair]


# -------------------------------------------------------------------- c_mul
def test_c_mul_matches_the_lesson_example():
    assert c_mul((3, 2), (1, 4)) == APPROX((-5.0, 14.0))


def test_i_times_i_is_minus_one():
    """Всё определение мнимой единицы: два поворота на 90 градусов = разворот."""
    assert c_mul(I, I) == APPROX((-1.0, 0.0))


def test_c_mul_is_not_componentwise():
    """Покомпонентное (8, -3) — самая частая ошибка; правильный ответ другой."""
    assert c_mul((2, 3), (4, -1)) == APPROX((11.0, 10.0))


def test_multiplying_by_i_rotates_by_ninety_degrees():
    """Точка (3, 4) после умножения на i становится (-4, 3): поворот на pi/2."""
    assert c_mul((3, 4), I) == APPROX((-4.0, 3.0))


# ------------------------------------------------------------------- c_conj
def test_conjugate_flips_only_the_imaginary_sign():
    assert c_conj((3, 2)) == APPROX((3.0, -2.0))


def test_conjugate_twice_gives_back_the_original():
    assert c_conj(c_conj((-7, 5))) == APPROX((-7.0, 5.0))


def test_conjugate_of_a_real_number_changes_nothing():
    assert c_conj((4, 0)) == APPROX((4.0, 0.0))


# -------------------------------------------------------------------- c_abs
def test_abs_of_three_four_is_five():
    assert c_abs((3, 4)) == APPROX(5.0)


def test_abs_ignores_the_sign_of_the_imaginary_part():
    assert c_abs((0, -2)) == APPROX(2.0)
    assert c_abs(c_conj((3, 4))) == APPROX(c_abs((3, 4)))


def test_abs_is_multiplicative():
    """Смысловое свойство: |z*w| = |z| * |w|, длины перемножаются."""
    z, w = (3, 2), (1, 4)
    assert c_abs(c_mul(z, w)) == APPROX(c_abs(z) * c_abs(w))


def test_z_times_its_conjugate_is_real_and_equals_abs_squared():
    """z * conj(z) = |z|^2 — мнимая часть обязана занулиться."""
    z = (3, 2)
    re, im = c_mul(z, c_conj(z))
    assert im == APPROX(0.0)
    assert re == APPROX(c_abs(z) ** 2)


# -------------------------------------------------------------------- c_div
def test_div_matches_hand_computation():
    assert c_div((5, 2), (1, -3)) == APPROX((-0.1, 1.7))


def test_one_over_i_is_minus_i():
    assert c_div(ONE, I) == APPROX((0.0, -1.0))


def test_div_undoes_mul():
    """Разделив произведение на один из множителей, получаем второй."""
    z, w = (3, 2), (1, 4)
    assert c_div(c_mul(z, w), w) == APPROX((3.0, 2.0))


def test_dividing_by_zero_raises():
    """Ловушка: у комплексного нуля нет обратного — нужна ошибка, не nan."""
    with pytest.raises(ZeroDivisionError):
        c_div((1, 1), (0, 0))


# ----------------------------------------------------------------- to_polar
def test_to_polar_of_three_four():
    r, theta = to_polar((3, 4))
    assert r == ANGLE(5.0)
    assert theta == ANGLE(math.atan2(4, 3))


def test_phase_uses_atan2_not_atan():
    """Для (-1, -1) atan(im/re) выдаст +pi/4; правильный угол в третьей четверти."""
    _, theta = to_polar((-1, -1))
    assert theta == ANGLE(-3 * math.pi / 4)


def test_phase_of_a_negative_real_is_pi():
    """На re < 0, im = 0 деление im/re дало бы 0 — а угол здесь pi."""
    r, theta = to_polar((-2, 0))
    assert r == ANGLE(2.0)
    assert abs(theta) == ANGLE(math.pi)


def test_to_polar_of_zero_has_zero_radius():
    r, _ = to_polar((0, 0))
    assert r == ANGLE(0.0)


# --------------------------------------------------------------- from_polar
def test_from_polar_at_quarter_turn_is_i():
    assert from_polar(1, math.pi / 2) == ANGLE((0.0, 1.0))


def test_from_polar_with_zero_angle_is_purely_real():
    assert from_polar(3, 0) == ANGLE((3.0, 0.0))


def test_from_polar_round_trips_to_polar():
    """Смысловое свойство: from_polar(*to_polar(z)) возвращает тот же z."""
    for z in [(3.0, 4.0), (-1.0, 2.0), (-5.0, -0.5), (0.0, -7.0)]:
        assert from_polar(*to_polar(z)) == ANGLE(z)


# -------------------------------------------------------------------- c_pow
def test_pow_agrees_with_repeated_multiplication():
    z = (3, 2)
    assert c_pow(z, 3) == ANGLE(c_mul(c_mul(z, z), z))


def test_pow_zero_is_one():
    assert c_pow((3, 2), 0) == ANGLE((1.0, 0.0))


def test_de_moivre_scales_the_magnitude():
    """|z^n| = |z|^n: длина возводится в степень, угол просто умножается."""
    z = (1.5, -0.5)
    assert c_abs(c_pow(z, 5)) == ANGLE(c_abs(z) ** 5)


def test_negative_power_is_the_reciprocal():
    z = (0.0, 1.0)
    assert c_mul(z, c_pow(z, -1)) == ANGLE((1.0, 0.0))


def test_twelve_sixth_turns_return_to_the_start():
    """e^(i*pi/6) в 12-й степени — полный оборот, снова единица."""
    assert c_pow(from_polar(1, math.pi / 6), 12) == ANGLE((1.0, 0.0))


# ----------------------------------------------------------- roots_of_unity
def test_fourth_roots_are_the_four_compass_points():
    assert flat(roots_of_unity(4)) == ANGLE(flat([(1, 0), (0, 1), (-1, 0), (0, -1)]))


def test_every_root_lies_on_the_unit_circle():
    assert [c_abs(r) for r in roots_of_unity(8)] == ANGLE([1.0] * 8)


def test_roots_sum_to_zero():
    """Симметрия: восемь равномерно расставленных векторов гасят друг друга."""
    re = sum(r[0] for r in roots_of_unity(8))
    im = sum(r[1] for r in roots_of_unity(8))
    assert (re, im) == ANGLE((0.0, 0.0))


def test_each_root_to_the_nth_power_is_one():
    """На то они и корни из единицы: w_k^n = 1 для любого k."""
    n = 6
    for w in roots_of_unity(n):
        assert c_pow(w, n) == ANGLE((1.0, 0.0))


def test_multiplying_a_root_by_the_primitive_root_gives_the_next_one():
    roots = roots_of_unity(8)
    primitive = roots[1]
    assert flat([c_mul(w, primitive) for w in roots[:-1]]) == ANGLE(flat(roots[1:]))


def test_roots_of_unity_rejects_non_positive_n():
    """Ловушка: n = 0 привело бы к делению на ноль внутри."""
    with pytest.raises(ValueError):
        roots_of_unity(0)
