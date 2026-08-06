"""Тесты к уроку «Генерация видео». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    attention_pairs,
    condition_on_first_frame,
    flicker_score,
    frame_deltas,
    patch_tokens,
    patchify,
    position_embedding,
    sample_frames,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в один."""
    return [v for row in M for v in row]


def mean_flicker(coupling, n_clips=400, n_frames=6, seed=0):
    rng = random.Random(seed)
    return sum(flicker_score(sample_frames(n_frames, rng, coupling)) for _ in range(n_clips)) / n_clips


# ------------------------------------------------------- position_embedding
def test_position_embedding_has_the_requested_length():
    assert len(position_embedding(3, 8)) == 8


def test_embedding_of_position_zero_alternates_zero_and_one():
    assert position_embedding(0, 4) == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_neighbouring_frames_get_more_similar_embeddings():
    """Смысл позиционного кодирования: рядом стоящие кадры выглядят похоже."""
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    e3 = position_embedding(3, 8)
    assert dot(e3, position_embedding(4, 8)) > dot(e3, position_embedding(30, 8))


def test_embedding_of_dimension_two_does_not_divide_by_zero():
    assert position_embedding(3, 2) == pytest.approx([math.sin(3.0), math.cos(3.0)])


# ------------------------------------------------------------------ patchify
def test_patchify_splits_the_clip_into_equal_blocks():
    assert patchify([1.0, 2.0, 3.0, 4.0], 2) == [[1.0, 2.0], [3.0, 4.0]]


def test_patch_size_one_gives_one_token_per_frame():
    assert patchify([1.0, 2.0], 1) == [[1.0], [2.0]]


def test_concatenating_the_patches_rebuilds_the_clip():
    """Патчификация ничего не теряет и не переставляет — это просто нарезка."""
    clip = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert flat(patchify(clip, 3)) == pytest.approx(clip)


def test_a_clip_that_does_not_divide_evenly_is_rejected():
    """Молча отбросить хвост — потерять кадры, которые никто потом не найдёт."""
    with pytest.raises(ValueError):
        patchify([1.0, 2.0, 3.0], 2)


# --------------------------------------------------------------- patch_tokens
def test_token_length_is_patch_size_plus_position_dimension():
    tokens = patch_tokens([1.0, 2.0, 3.0, 4.0], 2, 4)
    assert all(len(tok) == 6 for tok in tokens)


def test_token_count_matches_the_patch_count():
    assert len(patch_tokens([1.0] * 8, 2, 4)) == 4


def test_each_token_starts_with_its_own_patch_values():
    tokens = patch_tokens([1.0, 2.0, 3.0, 4.0], 2, 2)
    assert tokens[0][:2] == pytest.approx([1.0, 2.0])
    assert tokens[1][:2] == pytest.approx([3.0, 4.0])


def test_identical_patches_still_differ_by_position():
    """Без позиционного вложения два одинаковых кадра стали бы неразличимы,
    и порядок времени потерялся бы — прямой путь к мерцанию."""
    tokens = patch_tokens([5.0, 5.0], 1, 4)
    assert tokens[0] != pytest.approx(tokens[1])


def test_positions_are_numbered_by_patch_not_by_frame():
    """Патч из двух кадров получает ОДНУ позицию, а не две."""
    tokens = patch_tokens([1.0, 2.0, 3.0, 4.0], 2, 4)
    assert tokens[0][2:] == pytest.approx(position_embedding(0, 4))
    assert tokens[1][2:] == pytest.approx(position_embedding(1, 4))


# ------------------------------------------------------------ attention_pairs
def test_full_attention_is_quadratic_in_the_token_count():
    assert attention_pairs(8, 16) == (8 * 16) ** 2


def test_factorized_attention_is_spatial_plus_temporal():
    assert attention_pairs(8, 16, True) == 8 * 16 ** 2 + 16 * 8 ** 2


def test_factorization_saves_more_as_the_clip_grows():
    """Отсюда «полное 3-D в 16-100 раз дороже»: экономия растёт с размером."""
    small = attention_pairs(8, 16) / attention_pairs(8, 16, True)
    big = attention_pairs(32, 64) / attention_pairs(32, 64, True)
    assert big > small > 1.0


def test_a_single_frame_makes_factorization_pointless():
    """Для одной картинки разбивать нечего: факторизация только добавляет работу."""
    assert attention_pairs(1, 64, True) > attention_pairs(1, 64)


# --------------------------------------------------- frame_deltas / flicker
def test_frame_deltas_are_one_shorter_than_the_clip():
    assert len(frame_deltas([1.0, 2.0, 3.0, 4.0])) == 3


def test_a_still_clip_has_zero_deltas():
    assert frame_deltas([2.0, 2.0, 2.0]) == pytest.approx([0.0, 0.0])


def test_deltas_ignore_the_direction_of_motion():
    """Подъём и спуск одинаковой величины мерцают одинаково."""
    assert frame_deltas([0.0, 1.0]) == pytest.approx(frame_deltas([1.0, 0.0]))


def test_flicker_score_is_the_mean_of_the_deltas():
    assert flicker_score([1.0, 1.5, 1.0]) == APPROX(0.5)


def test_a_still_clip_does_not_flicker():
    assert flicker_score([2.0, 2.0, 2.0]) == APPROX(0.0)


def test_a_smooth_ramp_flickers_less_than_a_jittery_clip():
    assert flicker_score([0.0, 0.1, 0.2, 0.3]) < flicker_score([0.0, 3.0, -2.0, 4.0])


# -------------------------------------------------------------- sample_frames
def test_full_coupling_makes_every_frame_identical():
    """coupling = 1 — идеальная когерентность: клип превращается в стоп-кадр."""
    clip = sample_frames(5, random.Random(0), coupling=1.0)
    assert clip == pytest.approx([clip[0]] * 5)
    assert flicker_score(clip) == APPROX(0.0)


def test_independent_frames_are_the_flicker_baseline():
    """Покадровая диффузия: разность двух независимых N(0,1) даёт E|.| = 2/sqrt(pi)."""
    assert mean_flicker(0.0, seed=1) == pytest.approx(2.0 / math.sqrt(math.pi), abs=0.05)


def test_stronger_coupling_reduces_flicker():
    """Главное утверждение урока: связанные кадры дают более плавное движение."""
    scores = [mean_flicker(c, seed=2) for c in (0.0, 0.3, 0.6, 0.9)]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


def test_coupling_does_not_change_the_variance_of_a_single_frame():
    """Множитель sqrt(1 - coupling^2) держит дисперсию кадра единичной —
    иначе клип просто становился бы тише, и сравнивать было бы нечего."""
    def frame_std(coupling):
        rng = random.Random(3)
        vals = [sample_frames(1, rng, coupling)[0] for _ in range(6000)]
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    assert frame_std(0.0) == pytest.approx(1.0, abs=0.05)
    assert frame_std(0.7) == pytest.approx(1.0, abs=0.05)


def test_sampling_is_reproducible_for_a_fixed_seed():
    a = sample_frames(6, random.Random(9), coupling=0.5)
    b = sample_frames(6, random.Random(9), coupling=0.5)
    assert a == pytest.approx(b)


# ------------------------------------------------- condition_on_first_frame
def test_the_first_frame_is_pinned_exactly():
    assert condition_on_first_frame([1.0, 1.5, 3.0], 10.0)[0] == APPROX(10.0)


def test_conditioning_keeps_the_motion_intact():
    """Сдвиг общий для всех кадров, значит движение не меняется — это I2V."""
    clip = [1.0, 1.5, 3.0, 2.5]
    assert frame_deltas(condition_on_first_frame(clip, 10.0)) == pytest.approx(frame_deltas(clip))


def test_the_offset_propagates_to_every_frame():
    """Условие на нулевом кадре доезжает до последнего — кадры связаны."""
    assert condition_on_first_frame([1.0, 1.5, 3.0], 10.0) == pytest.approx([10.0, 10.5, 12.0])


def test_pinning_a_frame_to_its_own_value_changes_nothing():
    clip = [2.0, 3.0, 1.0]
    assert condition_on_first_frame(clip, 2.0) == pytest.approx(clip)


def test_conditioning_does_not_mutate_the_clip():
    clip = [1.0, 1.5, 3.0]
    condition_on_first_frame(clip, 10.0)
    assert clip == pytest.approx([1.0, 1.5, 3.0])
