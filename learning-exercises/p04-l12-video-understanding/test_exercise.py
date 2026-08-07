"""Тесты к уроку «Понимание видео — моделирование времени». Правь exercise.py."""

import random

import pytest

from exercise import (
    conv2plus1d_mid_channels,
    inflate_2d_to_3d,
    multi_clip_indices,
    sample_dense,
    sample_uniform,
    temporal_conv,
    temporal_mean_pool,
    top_k_accuracy,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в один."""
    out = []
    for item in M:
        if isinstance(item, list):
            out.extend(flat(item))
        else:
            out.append(item)
    return out


# ----------------------------------------------------------- sample_uniform
def test_uniform_sampling_spreads_indices_across_the_whole_clip():
    assert sample_uniform(10, 5) == [0, 2, 4, 6, 8]
    assert sample_uniform(300, 3) == [0, 100, 200]


def test_uniform_sampling_always_returns_exactly_T_indices():
    """Батч не соберётся, если у клипов разная длина."""
    for total in (1, 3, 7, 8, 9, 100):
        assert len(sample_uniform(total, 8)) == 8


def test_short_clip_is_padded_with_the_last_frame():
    """Ловушка: кадров меньше, чем T. Добираем повторами последнего."""
    assert sample_uniform(3, 5) == [0, 1, 2, 2, 2]


def test_uniform_step_is_fractional_not_integer_division():
    """int(total // T) на 10 кадрах и T=4 дал бы [0,2,4,6] вместо [0,2,5,7]."""
    assert sample_uniform(10, 4) == [0, 2, 5, 7]


# ------------------------------------------------------------- sample_dense
def test_dense_sampling_returns_consecutive_frames():
    """3D-свёртке нужны соседние кадры: между далёкими нет никакого движения."""
    indices = sample_dense(100, 8, random.Random(0))
    assert indices == list(range(indices[0], indices[0] + 8))


def test_dense_sampling_is_reproducible_for_the_same_seed():
    a = sample_dense(100, 8, random.Random(42))
    b = sample_dense(100, 8, random.Random(42))
    assert a == b


def test_dense_sampling_can_reach_the_very_last_frame():
    """Правый конец диапазона стартов валиден, иначе хвост видео не учится."""
    rng = random.Random(7)
    seen = {sample_dense(12, 8, rng)[0] for _ in range(200)}
    assert seen == set(range(0, 5))


def test_dense_sampling_pads_a_short_clip_like_the_uniform_one():
    assert sample_dense(3, 5, random.Random(0)) == [0, 1, 2, 2, 2]


# ------------------------------------------------------- multi_clip_indices
def test_multi_clip_covers_both_ends_of_the_video():
    assert multi_clip_indices(10, 4, 2) == [[0, 1, 2, 3], [6, 7, 8, 9]]


def test_a_single_clip_is_taken_from_the_middle():
    assert multi_clip_indices(10, 4, 1) == [[3, 4, 5, 6]]


def test_every_clip_has_the_same_length():
    clips = multi_clip_indices(100, 16, 5)
    assert len(clips) == 5
    assert all(len(clip) == 16 for clip in clips)


def test_clip_starts_are_non_decreasing_and_inside_the_video():
    clips = multi_clip_indices(64, 8, 4)
    starts = [clip[0] for clip in clips]
    assert starts == sorted(starts)
    assert clips[-1][-1] == 63


def test_short_video_gives_identical_padded_clips():
    assert multi_clip_indices(3, 4, 2) == [[0, 1, 2, 2], [0, 1, 2, 2]]


# ------------------------------------------------------ temporal_mean_pool
def test_mean_pool_averages_each_feature_across_frames():
    assert temporal_mean_pool([[1.0, 2.0], [3.0, 4.0]]) == APPROX([2.0, 3.0])


def test_pooling_one_frame_returns_that_frame():
    assert temporal_mean_pool([[5.0, -1.0]]) == APPROX([5.0, -1.0])


def test_mean_pool_is_order_invariant():
    """Потолок схемы 2D+pool: «открыть дверь» и «закрыть дверь» неразличимы."""
    forward = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
    backward = list(reversed(forward))
    assert temporal_mean_pool(backward) == APPROX(temporal_mean_pool(forward))


def test_identical_frames_pool_to_themselves():
    frame = [0.3, -0.7, 4.0]
    assert temporal_mean_pool([frame] * 8) == APPROX(frame)


# ---------------------------------------------------------- temporal_conv
def test_length_one_kernel_just_scales_the_signal():
    assert temporal_conv([1.0, 2.0, 3.0], [1.0]) == APPROX([1.0, 2.0, 3.0])


def test_valid_convolution_shortens_the_signal():
    assert len(temporal_conv([0.0] * 10, [1.0, 1.0, 1.0])) == 8


def test_motion_kernel_flips_sign_when_the_clip_is_reversed():
    """То, чего не умеет среднее: направление движения видно по знаку."""
    kernel = [-1.0, 0.0, 1.0]
    assert temporal_conv([0.0, 1.0, 2.0], kernel) == APPROX([2.0])
    assert temporal_conv([2.0, 1.0, 0.0], kernel) == APPROX([-2.0])


def test_motion_kernel_is_blind_to_a_static_clip():
    """Ничего не двигается — отклик нулевой, независимо от яркости."""
    assert temporal_conv([7.0] * 6, [-1.0, 0.0, 1.0]) == APPROX([0.0] * 4)


def test_kernel_is_not_flipped_before_the_product():
    """В сетях это корреляция, а не настоящая свёртка, хоть и зовут свёрткой."""
    assert temporal_conv([1.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == APPROX([1.0])


# --------------------------------------------------------- top_k_accuracy
def test_top1_counts_only_the_highest_scoring_class():
    assert top_k_accuracy([[0.1, 0.9]], [1]) == APPROX(1.0)
    assert top_k_accuracy([[0.1, 0.9]], [0]) == APPROX(0.0)


def test_top_k_forgives_a_miss_that_lands_inside_the_top_k():
    assert top_k_accuracy([[0.5, 0.3, 0.2]], [1], k=2) == APPROX(1.0)
    assert top_k_accuracy([[0.5, 0.3, 0.2]], [2], k=2) == APPROX(0.0)


def test_accuracy_never_falls_when_k_grows():
    scores = [[0.2, 0.5, 0.3], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]]
    labels = [2, 1, 0]
    values = [top_k_accuracy(scores, labels, k) for k in (1, 2, 3)]
    assert values == sorted(values)
    assert values[-1] == APPROX(1.0)


def test_tied_scores_count_as_a_hit():
    """Сортировка списка тут не нужна: считаем только строго большие скоры."""
    assert top_k_accuracy([[0.5, 0.5]], [0]) == APPROX(1.0)
    assert top_k_accuracy([[0.5, 0.5]], [1]) == APPROX(1.0)


def test_video_level_averaging_beats_clip_level_on_a_noisy_clip():
    """Два окна из одного видео: одно ошибается, усреднение спасает."""
    clip_scores = [[0.4, 0.6], [0.9, 0.1]]
    clip_level = top_k_accuracy(clip_scores, [0, 0])
    video_level = top_k_accuracy([temporal_mean_pool(clip_scores)], [0])
    assert clip_level == APPROX(0.5)
    assert video_level == APPROX(1.0)


# -------------------------------------------------------- inflate_2d_to_3d
def test_inflation_repeats_the_kernel_along_time():
    assert inflate_2d_to_3d([[1.0]], 2) == [[[0.5]], [[0.5]]]


def test_time_kernel_of_one_leaves_the_weights_alone():
    assert flat(inflate_2d_to_3d([[4.0, -2.0]], 1)) == APPROX([4.0, -2.0])


def test_inflated_kernel_keeps_the_total_weight_of_the_2d_one():
    """Деление на time_kernel — то, что спасает статистики batch norm."""
    kernel = [[1.0, 2.0], [3.0, -4.0]]
    inflated = inflate_2d_to_3d(kernel, 5)
    assert sum(flat(inflated)) == APPROX(sum(flat(kernel)))


def test_inflation_gives_the_same_response_as_2d_on_a_static_clip():
    """Ровно то, ради чего трюк: предобученные веса работают с первого прохода."""
    kernel = [[1.0, 2.0], [0.5, -1.0]]
    frame = [[3.0, 1.0], [2.0, 4.0]]
    response_2d = sum(
        kernel[i][j] * frame[i][j] for i in range(2) for j in range(2)
    )
    inflated = inflate_2d_to_3d(kernel, 3)
    response_3d = sum(
        inflated[t][i][j] * frame[i][j] for t in range(3) for i in range(2) for j in range(2)
    )
    assert response_3d == APPROX(response_2d)


# ------------------------------------------------- conv2plus1d_mid_channels
def test_mid_channels_matches_the_paper_formula():
    assert conv2plus1d_mid_channels(3, 64, 3) == 23
    assert conv2plus1d_mid_channels(64, 64, 3) == 144


def test_factorised_block_has_about_as_many_parameters_as_the_3d_one():
    """Смысл формулы: та же ёмкость, но с нелинейностью посередине."""
    for in_c, out_c, k in ((3, 64, 3), (64, 128, 3), (256, 256, 3)):
        mid = conv2plus1d_mid_channels(in_c, out_c, k)
        factorised = in_c * mid * k * k + mid * out_c * k
        plain_3d = in_c * out_c * k ** 3
        # недобор меньше одного шага по mid: ровно эффект целочисленного деления
        assert factorised <= plain_3d
        assert plain_3d - factorised < in_c * k * k + out_c * k


def test_mid_channels_is_a_whole_number():
    value = conv2plus1d_mid_channels(17, 33, 3)
    assert isinstance(value, int)


def test_wider_layers_need_more_middle_channels():
    widths = [conv2plus1d_mid_channels(c, c, 3) for c in (16, 32, 64, 128)]
    assert widths == sorted(widths)
    assert len(set(widths)) == 4
