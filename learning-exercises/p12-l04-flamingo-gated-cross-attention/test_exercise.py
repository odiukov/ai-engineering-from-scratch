"""Тесты к уроку «Flamingo и gated cross-attention». Правь exercise.py."""

import math

import pytest

from exercise import (
    IMAGE,
    TEXT,
    build_few_shot_prompt,
    cross_attention,
    gated_cross_attention_step,
    gated_residual,
    interleaved_cross_mask,
    most_recent_image,
    perceiver_resampler,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-5)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def ramp(n, dim):
    """Детерминированные векторы: никакого глобального random в тестах."""
    return [[math.sin(i * dim + d) * 0.3 for d in range(dim)] for i in range(n)]


# ---------------------------------------------------------- cross_attention
def test_cross_attention_mixes_values_by_similarity():
    out = cross_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[4.0], [0.0]])
    assert flat(out) == ROUGH([2.6790462])


def test_cross_attention_returns_one_output_per_query():
    out = cross_attention(ramp(5, 4), ramp(37, 4), ramp(37, 6))
    assert len(out) == 5
    assert all(len(o) == 6 for o in out)


def test_cross_attention_output_stays_inside_the_value_range():
    """Выход — выпуклая комбинация values, вылезти за их диапазон он не может."""
    values = [[-3.0], [8.0], [0.5]]
    out = cross_attention(ramp(4, 2), ramp(3, 2), values)
    assert all(-3.0 <= o[0] <= 8.0 for o in out)


def test_cross_attention_survives_huge_logits():
    """Наивный exp падает; сдвиг на максимум — нет."""
    out = cross_attention([[500.0, 500.0]], [[500.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]])
    assert out[0][0] == ROUGH(1.0)


def test_cross_attention_rejects_mismatched_keys_and_values():
    with pytest.raises(ValueError):
        cross_attention([[1.0]], [[1.0], [2.0]], [[1.0]])


# ------------------------------------------------------ perceiver_resampler
def test_resampler_output_length_is_the_latent_count():
    """Ради этого он и нужен: 12 патчей и 300 патчей дают одинаковую длину."""
    latents = ramp(8, 4)
    assert len(perceiver_resampler(ramp(12, 4), latents)) == 8
    assert len(perceiver_resampler(ramp(300, 4), latents)) == 8


def test_resampler_with_zero_blocks_returns_the_latents_unchanged():
    latents = ramp(4, 3)
    assert flat(perceiver_resampler(ramp(20, 3), latents, blocks=0)) == APPROX(
        flat(latents)
    )


def test_resampler_actually_changes_the_latents():
    latents = ramp(4, 3)
    got = perceiver_resampler(ramp(20, 3), latents, blocks=1)
    assert flat(got) != APPROX(flat(latents))


def test_resampler_does_not_mutate_the_input_latents():
    """Латенты — параметры модели; испортишь их, и следующая картинка поедет криво."""
    latents = ramp(4, 3)
    before = flat(latents)
    perceiver_resampler(ramp(20, 3), latents, blocks=3)
    assert flat(latents) == APPROX(before)


def test_resampler_rejects_negative_block_count():
    with pytest.raises(ValueError):
        perceiver_resampler(ramp(4, 2), ramp(2, 2), blocks=-1)


# ----------------------------------------------------------- gated_residual
def test_zero_gate_is_an_exact_no_op():
    """tanh(0) = 0 ровно, поэтому сравнение точное, без всякого approx."""
    hidden = [[1.0, 2.0], [-3.0, 0.5]]
    assert gated_residual(hidden, [[10.0, 10.0], [10.0, 10.0]], 0.0) == hidden


def test_open_gate_adds_the_visual_branch():
    got = gated_residual([[1.0]], [[10.0]], 2.0)
    assert got[0][0] == ROUGH(1.0 + math.tanh(2.0) * 10.0)


def test_gate_contribution_never_exceeds_the_cross_branch():
    """|tanh| <= 1: визуальная ветка не может затереть текстовое состояние."""
    hidden, cross = [[1.0, -2.0]], [[5.0, 7.0]]
    for alpha in (-100.0, -1.0, 0.3, 100.0):
        got = gated_residual(hidden, cross, alpha)
        for h, c, g in zip(hidden[0], cross[0], got[0]):
            assert abs(g - h) <= abs(c) + 1e-12


def test_negative_gate_flips_the_sign_of_the_contribution():
    up = gated_residual([[0.0]], [[4.0]], 1.5)
    down = gated_residual([[0.0]], [[4.0]], -1.5)
    assert up[0][0] == ROUGH(-down[0][0])
    assert up[0][0] > 0


def test_gate_saturates_instead_of_exploding():
    """alpha=50 и alpha=5000 дают практически одно и то же — tanh упирается в 1."""
    a = gated_residual([[0.0]], [[3.0]], 50.0)
    b = gated_residual([[0.0]], [[3.0]], 5000.0)
    assert a[0][0] == ROUGH(3.0)
    assert b[0][0] == ROUGH(3.0)


def test_gated_residual_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        gated_residual([[1.0, 2.0]], [[1.0]], 0.5)


# ------------------------------------------- gated_cross_attention_step
def test_the_whole_block_is_a_no_op_at_initialization():
    """Flamingo на шаге 0 — это в точности замороженная LLM, слово в слово."""
    hidden = [[0.5, -1.0], [2.0, 3.0]]
    assert gated_cross_attention_step(hidden, ramp(6, 2), 0.0) == hidden


def test_the_block_changes_hidden_when_the_gate_opens():
    hidden = [[0.5, -1.0], [2.0, 3.0]]
    got = gated_cross_attention_step(hidden, ramp(6, 2), 1.0)
    assert flat(got) != APPROX(flat(hidden))


def test_the_block_keeps_the_sequence_shape():
    """Визуальные токены не попадают во входную последовательность LLM."""
    hidden = ramp(7, 4)
    got = gated_cross_attention_step(hidden, ramp(64, 4), 0.8)
    assert len(got) == 7
    assert all(len(v) == 4 for v in got)


def test_the_block_equals_attention_then_gating():
    hidden, visual = ramp(3, 4), ramp(9, 4)
    cross = cross_attention(hidden, visual, visual)
    assert flat(gated_cross_attention_step(hidden, visual, 0.7)) == APPROX(
        flat(gated_residual(hidden, cross, 0.7))
    )


# --------------------------------------------------------- most_recent_image
def test_text_before_the_first_image_owns_nothing():
    assert most_recent_image([TEXT, TEXT, IMAGE]) == [None, None, 0]


def test_each_position_gets_the_nearest_preceding_image():
    assert most_recent_image([TEXT, IMAGE, TEXT, IMAGE, TEXT]) == [None, 0, 0, 1, 1]


def test_an_image_owns_itself():
    assert most_recent_image([IMAGE, IMAGE, IMAGE]) == [0, 1, 2]


def test_most_recent_image_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        most_recent_image([IMAGE, "audio"])


# ---------------------------------------------------- interleaved_cross_mask
def test_mask_width_is_images_times_tokens_per_image():
    mask = interleaved_cross_mask([IMAGE, TEXT, IMAGE, TEXT], 3)
    assert len(mask) == 4
    assert all(len(row) == 6 for row in mask)


def test_text_sees_only_the_most_recent_image_not_the_earlier_one():
    """Сознательный выбор Flamingo: подпись привязана к ближайшей картинке."""
    mask = interleaved_cross_mask([IMAGE, TEXT, IMAGE, TEXT], 2)
    assert mask[3] == [False, False, True, True]


def test_no_position_can_see_a_later_image():
    """Причинность: визуальные токены из будущего недоступны."""
    seq = [TEXT, IMAGE, TEXT, IMAGE, TEXT, TEXT]
    tokens = 2
    mask = interleaved_cross_mask(seq, tokens)
    owners = most_recent_image(seq)
    for row, owner in zip(mask, owners):
        limit = 0 if owner is None else (owner + 1) * tokens
        assert not any(row[limit:])


def test_positions_before_any_image_see_nothing():
    mask = interleaved_cross_mask([TEXT, TEXT, IMAGE], 2)
    assert mask[0] == [False, False]
    assert mask[1] == [False, False]
    assert mask[2] == [True, True]


def test_every_row_after_the_first_image_sees_exactly_one_block():
    mask = interleaved_cross_mask([IMAGE, TEXT, TEXT, IMAGE, TEXT], 4)
    for row in mask:
        assert sum(row) in (0, 4)


def test_mask_rejects_nonpositive_tokens_per_image():
    with pytest.raises(ValueError):
        interleaved_cross_mask([IMAGE], 0)


# ------------------------------------------------------ build_few_shot_prompt
def test_few_shot_prompt_alternates_image_and_caption():
    got = build_few_shot_prompt(
        [("cat.jpg", "A photo of a cat."), ("dog.jpg", "A photo of a dog.")],
        "bird.jpg",
    )
    assert [kind for kind, _ in got] == [IMAGE, TEXT, IMAGE, TEXT, IMAGE]


def test_few_shot_prompt_ends_on_an_unanswered_image():
    """Незакрытый шаблон — вся механика in-context обучения."""
    got = build_few_shot_prompt([("cat.jpg", "a cat")], "bird.jpg")
    assert got[-1] == (IMAGE, "bird.jpg")


def test_zero_examples_is_the_zero_shot_case():
    assert build_few_shot_prompt([], "bird.jpg") == [(IMAGE, "bird.jpg")]


def test_prompt_composes_with_the_interleaved_mask():
    """Промпт и маска — две половины одного механизма, они обязаны стыковаться."""
    prompt = build_few_shot_prompt([("a.jpg", "alpha"), ("b.jpg", "beta")], "c.jpg")
    kinds = [kind for kind, _ in prompt]
    mask = interleaved_cross_mask(kinds, 1)
    assert len(mask[0]) == 3
    assert mask[-1] == [False, False, True]


def test_prompt_rejects_an_empty_caption():
    with pytest.raises(ValueError):
        build_few_shot_prompt([("cat.jpg", "   ")], "bird.jpg")
