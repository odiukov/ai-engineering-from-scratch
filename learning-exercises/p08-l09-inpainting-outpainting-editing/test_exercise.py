"""Тесты к уроку «Инпейнтинг, аутпейнтинг и редактирование». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    build_inpaint_input,
    dilate_mask,
    invert_mask,
    reinject_known,
    repaint_timesteps,
    restore_known,
    sdedit_noise,
    sdedit_start_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def schedule(T):
    """Линейное расписание DDPM: список alpha_bar_t. Урок 06, если забыл."""
    bars, cum = [], 1.0
    for t in range(T):
        beta = 1e-4 + (0.02 - 1e-4) * t / (T - 1)
        cum *= (1.0 - beta)
        bars.append(cum)
    return bars


def mean_std(values):
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def correlation(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / len(a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b) / len(b))
    return cov / (sa * sb)


# -------------------------------------------------------------- invert_mask
def test_invert_mask_flips_every_position():
    assert invert_mask([True, False, False]) == [False, True, True]


def test_outpainting_is_inpainting_with_the_mask_inverted():
    """Двойное обращение возвращает исходную маску — аутпейнтинг и инпейнтинг
    это одна и та же операция, отличается только маска."""
    mask = [True, False, True, True]
    assert invert_mask(invert_mask(mask)) == mask


def test_invert_mask_does_not_mutate_the_input():
    mask = [True, False]
    invert_mask(mask)
    assert mask == [True, False]


# -------------------------------------------------------------- dilate_mask
def test_dilation_with_zero_radius_changes_nothing():
    mask = [False, True, False, False]
    assert dilate_mask(mask, 0) == mask


def test_dilation_grows_the_mask_by_the_radius():
    assert dilate_mask([False, True, False, False], 1) == [True, True, True, False]


def test_dilated_mask_always_contains_the_original():
    """Расширение только добавляет позиции, отнимать не может."""
    mask = [False, True, False, True, False, False]
    grown = dilate_mask(mask, 2)
    assert all(g for m, g in zip(mask, grown) if m)


def test_dilating_an_empty_mask_keeps_it_empty():
    assert dilate_mask([False] * 5, 3) == [False] * 5


def test_dilation_does_not_wrap_around_the_edges():
    """Отрицательный индекс в Python — это конец списка. Если не обрезать
    границы, маска замкнётся в кольцо и первый элемент зацепит последний."""
    assert dilate_mask([False, False, False, True], 1) == [False, False, True, True]


# ------------------------------------------------------- build_inpaint_input
def test_inpaint_input_stacks_three_blocks():
    assert build_inpaint_input([0.5], [2.0], [True]) == pytest.approx([0.5, 2.0, 1.0])


def test_mask_channel_is_encoded_as_zero_and_one():
    got = build_inpaint_input([0.0, 0.0], [0.0, 0.0], [False, True])
    assert got[-2:] == pytest.approx([0.0, 1.0])


def test_inpaint_input_carries_the_source_so_the_model_can_see_around_the_hole():
    """Ровно из-за этого блока «правильная» инпейнт-модель не оставляет швов:
    закодированный исходник приходит на вход целиком, включая то, что под маской."""
    source = [1.0, 2.0, 3.0]
    got = build_inpaint_input([0.0, 0.0, 0.0], source, [True, True, True])
    assert got[3:6] == pytest.approx(source)
    assert len(got) == 9


# ------------------------------------------------------------ reinject_known
def test_reinjection_leaves_the_masked_positions_untouched():
    """Под маской работает денойзер — переинъекция туда не лезет."""
    bars = schedule(40)
    x_t = [9.0, 9.0, 9.0]
    out = reinject_known(x_t, [True, False, True], [0.0, 0.0, 0.0], 20, bars, random.Random(0))
    assert out[0] == APPROX(9.0)
    assert out[2] == APPROX(9.0)
    assert out[1] != APPROX(9.0)


def test_reinjection_at_zero_noise_puts_the_known_pixels_back_exactly():
    """При alpha_bar = 1 шума нет, и известные позиции обязаны стать ровно clean."""
    out = reinject_known([9.0, 9.0], [True, False], [0.0, 5.0], 0, [1.0], random.Random(0))
    assert out == pytest.approx([9.0, 5.0])


def test_reinjection_does_not_mutate_the_noisy_input():
    bars = schedule(40)
    x_t = [1.0, 2.0]
    reinject_known(x_t, [False, False], [0.0, 0.0], 10, bars, random.Random(0))
    assert x_t == pytest.approx([1.0, 2.0])


def test_reinjection_draws_noise_only_for_the_known_positions():
    """Лишние обращения к rng ломают воспроизводимость всей цепочки."""
    bars = schedule(40)
    mask = [True, False, True, False, False]      # известных позиций три
    used = random.Random(0)
    reinject_known([0.0] * 5, mask, [0.0] * 5, 10, bars, used)

    reference = random.Random(0)
    for _ in range(3):
        reference.gauss(0, 1)
    assert used.gauss(0, 1) == APPROX(reference.gauss(0, 1))


def test_reinjected_values_follow_the_forward_process_statistics():
    """Уровень шума обязан совпадать с шагом t, иначе модель увидит невозможное."""
    bars = schedule(100)
    t, clean = 60, 2.0
    rng = random.Random(1)
    got = [reinject_known([0.0], [False], [clean], t, bars, rng)[0] for _ in range(3000)]
    m, s = mean_std(got)
    assert m == pytest.approx(math.sqrt(bars[t]) * clean, abs=0.05)
    assert s == pytest.approx(math.sqrt(1 - bars[t]), abs=0.05)


# ------------------------------------------------------------- restore_known
def test_restore_puts_the_known_pixels_back_bit_for_bit():
    assert restore_known([1.0, 2.0], [True, False], [9.0, 9.0]) == pytest.approx([1.0, 9.0])


def test_restore_keeps_everything_the_denoiser_produced_under_the_mask():
    generated = [0.11, 0.22, 0.33]
    out = restore_known(generated, [True, True, True], [9.0, 9.0, 9.0])
    assert out == pytest.approx(generated)


def test_restore_with_an_empty_mask_returns_the_source():
    """Нечего перегенерировать — на выходе ровно исходная картинка."""
    assert restore_known([0.0, 0.0], [False, False], [3.0, 4.0]) == pytest.approx([3.0, 4.0])


def test_restore_does_not_mutate_its_inputs():
    x, clean = [1.0, 2.0], [9.0, 9.0]
    restore_known(x, [True, False], clean)
    assert x == pytest.approx([1.0, 2.0])
    assert clean == pytest.approx([9.0, 9.0])


# --------------------------------------------------------- sdedit_start_step
def test_zero_strength_means_no_noising_at_all():
    assert sdedit_start_step(0.0, 1000) == 0


def test_full_strength_noises_the_whole_schedule():
    assert sdedit_start_step(1.0, 1000) == 1000


def test_strength_is_a_fraction_of_the_schedule():
    assert sdedit_start_step(0.3, 1000) == 300
    assert sdedit_start_step(0.6, 50) == 30


def test_strength_outside_the_unit_interval_is_clamped():
    """Отрицательное число шагов бессмысленно, больше T — тоже."""
    assert sdedit_start_step(-0.5, 40) == 0
    assert sdedit_start_step(2.0, 40) == 40


# -------------------------------------------------------------- sdedit_noise
def test_zero_steps_returns_a_copy_of_the_source():
    x0 = [1.0, 2.0]
    got = sdedit_noise(x0, 0, schedule(40), random.Random(0))
    assert got == pytest.approx(x0)
    assert got is not x0


def test_sdedit_noise_does_not_mutate_the_source():
    x0 = [1.0, 2.0]
    sdedit_noise(x0, 20, schedule(40), random.Random(0))
    assert x0 == pytest.approx([1.0, 2.0])


def test_higher_strength_keeps_less_of_the_source():
    """Обрыв верности: чем больше steps, тем меньше в результате исходника."""
    bars = schedule(1000)
    rng = random.Random(2)
    source = [rng.gauss(0, 1) for _ in range(2000)]

    rng = random.Random(3)
    light = sdedit_noise(source, sdedit_start_step(0.3, 1000), bars, rng)
    rng = random.Random(3)
    heavy = sdedit_noise(source, sdedit_start_step(0.9, 1000), bars, rng)

    assert correlation(source, light) > correlation(source, heavy)
    assert correlation(source, light) > 0.5
    assert abs(correlation(source, heavy)) < 0.2


def test_sdedit_noise_uses_the_alpha_bar_of_the_requested_step():
    bars = schedule(100)
    steps, x0 = 60, 2.0
    rng = random.Random(4)
    got = [sdedit_noise([x0], steps, bars, rng)[0] for _ in range(3000)]
    m, s = mean_std(got)
    assert m == pytest.approx(math.sqrt(bars[steps - 1]) * x0, abs=0.05)
    assert s == pytest.approx(math.sqrt(1 - bars[steps - 1]), abs=0.05)


# ---------------------------------------------------------- repaint_timesteps
def test_without_jumps_repaint_is_a_plain_descent():
    assert repaint_timesteps(5, 2, 0) == [4, 3, 2, 1, 0]


def test_repaint_starts_at_the_top_and_ends_at_zero():
    steps = repaint_timesteps(20, 5, 2)
    assert steps[0] == 19
    assert steps[-1] == 0


def test_jumps_make_the_chain_revisit_timesteps():
    """В этом весь RePaint: участок переденойзивается по нескольку раз."""
    plain = repaint_timesteps(20, 5, 0)
    jumpy = repaint_timesteps(20, 5, 2)
    assert len(jumpy) > len(plain)


def test_every_timestep_is_still_visited_at_least_once():
    steps = repaint_timesteps(20, 4, 3)
    assert set(steps) == set(range(20))


def test_a_jump_at_least_as_long_as_the_interval_is_rejected():
    """jump_length >= jump_every означает, что спуск не спускается,
    и цикл не закончится никогда. Это ValueError, а не зависание."""
    with pytest.raises(ValueError):
        repaint_timesteps(20, 3, 3)
