"""Тесты к уроку «Emu3: генерация картинок и видео обычным next-token». Правь exercise.py."""

import random

import pytest

from exercise import (
    GUIDANCE_RANGE,
    RECOMMENDED_TEMPERATURE,
    cfg_logits,
    frames_in_clip,
    generation_seconds,
    image_tokens,
    sample_image_tokens,
    sample_token,
    softmax,
    video_tokens,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- image_tokens
def test_image_tokens_for_the_emu3_default():
    assert image_tokens(512, 512, 8) == 4096


def test_image_tokens_grow_quadratically_with_the_side():
    """Удвоил сторону — учетверил время генерации. Вот почему 4K в один проход не бывает."""
    assert image_tokens(1024, 1024, 8) == 4 * image_tokens(512, 512, 8)
    assert image_tokens(2048, 2048, 8) == 16 * image_tokens(512, 512, 8)


def test_image_tokens_rejects_a_non_divisible_side():
    with pytest.raises(ValueError):
        image_tokens(500, 512, 8)


def test_image_tokens_rejects_a_zero_reduction():
    with pytest.raises(ValueError):
        image_tokens(512, 512, 0)


# ------------------------------------------------------------- video_tokens
def test_video_tokens_for_a_four_second_clip():
    assert video_tokens(256, 256, 32, 4, 4) == 32768


def test_video_tokens_scale_linearly_with_the_frame_count():
    assert video_tokens(256, 256, 64, 4, 4) == 2 * video_tokens(256, 256, 32, 4, 4)


def test_spatial_and_temporal_compression_trade_off_at_equal_cost():
    """4x4x4 и 8x8x1 стоят одинаково; разница в том, ЧТО сохраняется — детали или время."""
    assert video_tokens(256, 256, 32, 4, 4) == video_tokens(256, 256, 32, 8, 1)


def test_a_video_costs_far_more_than_a_single_frame():
    assert video_tokens(256, 256, 32, 4, 4) == 8 * image_tokens(256, 256, 4)


def test_video_tokens_rejects_frames_not_divisible_by_the_temporal_reduction():
    with pytest.raises(ValueError):
        video_tokens(256, 256, 30, 4, 4)


# ------------------------------------------------------------ frames_in_clip
def test_frames_in_clip_for_four_seconds_at_eight_fps():
    assert frames_in_clip(4.0, 8) == 32


def test_frames_in_clip_scales_with_both_arguments():
    assert frames_in_clip(8.0, 8) == 2 * frames_in_clip(4.0, 8)
    assert frames_in_clip(4.0, 16) == 2 * frames_in_clip(4.0, 8)


def test_frames_in_clip_rejects_a_zero_fps():
    with pytest.raises(ValueError):
        frames_in_clip(4.0, 0)


# -------------------------------------------------------- generation_seconds
def test_generation_seconds_for_one_emu3_image():
    """4096 токенов на 30 tok/s — те самые «две минуты на картинку»."""
    assert generation_seconds(4096, 30) == pytest.approx(136.53, abs=0.01)


def test_generation_seconds_is_linear_in_the_token_count():
    assert generation_seconds(8192, 30) == APPROX(2 * generation_seconds(4096, 30))


def test_generation_seconds_rejects_a_zero_rate():
    with pytest.raises(ValueError):
        generation_seconds(4096, 0)


# ---------------------------------------------------------------- cfg_logits
def test_cfg_at_weight_one_is_the_conditional_logits():
    """Вес 1.0 — это ровно «guidance выключен»."""
    assert cfg_logits([2.0, 0.0], [1.0, 1.0], 1.0) == APPROX([2.0, 0.0])


def test_cfg_at_weight_zero_is_the_unconditional_logits():
    assert cfg_logits([2.0, 0.0], [1.0, 1.0], 0.0) == APPROX([1.0, 1.0])


def test_cfg_widens_the_gap_as_guidance_grows():
    gaps = [
        cfg_logits([2.0, 0.0], [1.0, 1.0], g)[0]
        - cfg_logits([2.0, 0.0], [1.0, 1.0], g)[1]
        for g in (1.0, 3.0, 5.0, 7.0)
    ]
    assert gaps == sorted(gaps)
    assert gaps[0] < gaps[-1]


def test_cfg_is_linear_in_the_guidance_weight():
    low = cfg_logits([2.0, 0.0], [1.0, 1.0], 1.0)
    high = cfg_logits([2.0, 0.0], [1.0, 1.0], 3.0)
    middle = cfg_logits([2.0, 0.0], [1.0, 1.0], 2.0)
    assert middle == APPROX([(a + b) / 2 for a, b in zip(low, high)])


def test_cfg_rejects_mismatched_lengths():
    """zip молча обрезал бы длинный список и спрятал баг."""
    with pytest.raises(ValueError):
        cfg_logits([2.0, 0.0, 1.0], [1.0, 1.0], 3.0)


# -------------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([0.3, -1.2, 4.0, 2.5])) == pytest.approx(1.0, abs=1e-12)


def test_softmax_is_shift_invariant():
    """Общая добавка к логитам ничего не меняет — на этом и стоит защита от переполнения."""
    assert softmax([0.0, 1.0, 2.0]) == pytest.approx(
        softmax([500.0, 501.0, 502.0]), abs=1e-12
    )


def test_softmax_survives_huge_logits():
    """Наивный math.exp(1001) — это OverflowError, а не «медленно»."""
    assert softmax([1000.0, 1001.0]) == pytest.approx([0.26894142, 0.73105858], abs=1e-8)


def test_low_temperature_concentrates_on_the_argmax():
    assert softmax([0.0, 1.0], 0.01) == pytest.approx([0.0, 1.0], abs=1e-9)


def test_the_generation_temperature_is_sharper_than_the_perception_one():
    """0.8 против 1.0: генерация должна быть увереннее, чем ответ на вопрос."""
    sharp = softmax([0.0, 1.0], RECOMMENDED_TEMPERATURE["generation"])
    soft = softmax([0.0, 1.0], RECOMMENDED_TEMPERATURE["perception"])
    assert max(sharp) > max(soft)


def test_softmax_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError):
        softmax([0.0, 1.0], 0.0)


# --------------------------------------------------------------- sample_token
def test_sample_token_is_reproducible_for_the_same_seed():
    logits = [0.5, 1.5, -0.5]
    a = [sample_token(logits, random.Random(3)) for _ in range(5)]
    b = [sample_token(logits, random.Random(3)) for _ in range(5)]
    assert a == b


def test_sample_token_frequencies_match_the_distribution():
    rng = random.Random(11)
    draws = [sample_token([0.0, 0.0, 0.0], rng) for _ in range(6000)]
    for index in (0, 1, 2):
        assert draws.count(index) / 6000 == pytest.approx(1 / 3, abs=0.03)


def test_sample_token_never_picks_a_vanishing_probability():
    rng = random.Random(2)
    draws = {sample_token([0.0, -1000.0], rng) for _ in range(500)}
    assert draws == {0}


def test_a_low_temperature_always_picks_the_argmax():
    rng = random.Random(5)
    draws = {sample_token([0.0, 1.0, 0.5], rng, 0.001) for _ in range(100)}
    assert draws == {1}


def test_sample_token_stays_inside_the_vocabulary():
    rng = random.Random(9)
    draws = [sample_token([0.1, 0.2, 0.3, 0.4], rng) for _ in range(200)]
    assert all(0 <= t < 4 for t in draws)


# -------------------------------------------------------- sample_image_tokens
def test_sample_image_tokens_returns_the_requested_count():
    assert len(sample_image_tokens(64, [1.0, 0.0], [0.0, 1.0], random.Random(0))) == 64


def test_sample_image_tokens_is_reproducible_for_the_same_seed():
    a = sample_image_tokens(32, [1.0, 0.0], [0.0, 1.0], random.Random(1))
    b = sample_image_tokens(32, [1.0, 0.0], [0.0, 1.0], random.Random(1))
    assert a == b


def test_stronger_guidance_follows_the_prompt_more_often():
    """Смысл CFG глазами: чем выше вес, тем послушнее картинка промпту."""
    weak = sample_image_tokens(300, [1.0, 0.0], [0.0, 1.0], random.Random(0), 1.0)
    strong = sample_image_tokens(
        300, [1.0, 0.0], [0.0, 1.0], random.Random(0), GUIDANCE_RANGE[1]
    )
    assert strong.count(0) > weak.count(0)


def test_sample_image_tokens_rejects_a_negative_count():
    with pytest.raises(ValueError):
        sample_image_tokens(-1, [1.0, 0.0], [0.0, 1.0], random.Random(0))
