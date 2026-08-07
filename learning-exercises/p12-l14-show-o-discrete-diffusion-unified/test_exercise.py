"""Тесты к уроку «Show-o и masked discrete diffusion». Правь exercise.py."""

import pytest

from exercise import (
    MASK,
    compression_ratio,
    cosine_schedule,
    linear_schedule,
    sample_masked,
    softmax,
    top_k_confident,
    unmask_counts,
    unmask_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

VOCAB = 4


def make_predict(calls=None):
    """Детерминированная заглушка вместо transformer.

    Позиция i «хочет» токен i % VOCAB, и чем правее позиция, тем увереннее.
    calls — список, куда пишется по единице на каждый forward-проход.
    """

    def predict(tokens):
        if calls is not None:
            calls.append(1)
        rows = []
        for i in range(len(tokens)):
            row = [0.0] * VOCAB
            row[i % VOCAB] = 1.0 + 0.1 * i
            rows.append(row)
        return rows

    return predict


# ---------------------------------------------------------- cosine_schedule
def test_cosine_schedule_has_one_value_per_step_boundary():
    assert len(cosine_schedule(8)) == 9


def test_cosine_schedule_runs_from_all_masked_to_none_masked():
    ratios = cosine_schedule(8)
    assert ratios[0] == APPROX(1.0)
    assert ratios[-1] == APPROX(0.0)


def test_cosine_schedule_never_remasks():
    ratios = cosine_schedule(16)
    assert all(a > b for a, b in zip(ratios, ratios[1:]))


def test_cosine_schedule_rejects_zero_steps():
    with pytest.raises(ValueError):
        cosine_schedule(0)


def test_cosine_keeps_more_tokens_masked_at_the_halfway_point():
    """В этом весь смысл косинуса: раскрытия сдвинуты к концу, где контекста больше."""
    T = 8
    assert cosine_schedule(T)[T // 2] > linear_schedule(T)[T // 2]


# ---------------------------------------------------------- linear_schedule
def test_linear_schedule_values():
    assert linear_schedule(4) == APPROX([1.0, 0.75, 0.5, 0.25, 0.0])


def test_linear_schedule_steps_are_all_equal():
    ratios = linear_schedule(10)
    deltas = [a - b for a, b in zip(ratios, ratios[1:])]
    assert deltas == APPROX([deltas[0]] * len(deltas))


# ------------------------------------------------------------ unmask_counts
def test_unmask_counts_worked_example():
    assert unmask_counts([1.0, 0.5, 0.0], 8) == [4, 4]


def test_unmask_counts_unmask_every_token_exactly_once():
    """Что бы ни было в расписании, к концу раскрыты все токены и ни один дважды."""
    for schedule in (cosine_schedule(8), linear_schedule(8)):
        counts = unmask_counts(schedule, 16)
        assert sum(counts) == 16
        assert all(c >= 0 for c in counts)


def test_linear_schedule_unmasks_the_same_number_every_step():
    assert unmask_counts(linear_schedule(8), 16) == [2] * 8


def test_cosine_schedule_accelerates_towards_the_end():
    counts = unmask_counts(cosine_schedule(8), 16)
    assert all(a <= b for a, b in zip(counts, counts[1:]))
    assert counts[-1] > counts[0]


def test_unmask_counts_rejects_a_growing_schedule():
    """Растущая доля маски означала бы «замаскировать обратно» — так нельзя."""
    with pytest.raises(ValueError):
        unmask_counts([1.0, 0.2, 0.6, 0.0], 10)


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, -2.0, 0.5])) == APPROX(1.0)


def test_softmax_keeps_the_order_of_logits():
    probs = softmax([0.1, 3.0, -1.0])
    assert probs[1] > probs[0] > probs[2]


def test_softmax_survives_huge_logits():
    """Наивный exp(1000) падает с OverflowError — нужен сдвиг на максимум."""
    assert softmax([1000.0, 0.0])[0] == APPROX(1.0)


# ----------------------------------------------------------- top_k_confident
def test_top_k_confident_picks_the_most_confident_position():
    tokens = [MASK, 5, MASK]
    logits = [[2.0, 0.0], [0.0, 0.0], [9.0, 0.0]]
    assert top_k_confident(tokens, logits, 1) == [2]


def test_top_k_confident_never_touches_an_already_known_token():
    """Позиция 1 самая уверенная, но она уже раскрыта — её не переписывают."""
    tokens = [MASK, 7, MASK]
    logits = [[1.0, 0.0], [50.0, 0.0], [2.0, 0.0]]
    assert 1 not in top_k_confident(tokens, logits, 2)


def test_top_k_confident_returns_all_masked_when_k_is_too_big():
    tokens = [MASK, 3, MASK]
    logits = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert top_k_confident(tokens, logits, 99) == [0, 2]


def test_top_k_confident_rejects_k_that_unmasks_nothing():
    with pytest.raises(ValueError):
        top_k_confident([MASK], [[1.0, 0.0]], 0)


# --------------------------------------------------------------- unmask_step
def test_unmask_step_worked_example():
    assert unmask_step([MASK, MASK], [[0.0, 9.0], [0.0, 0.0]], 1) == [1, MASK]


def test_unmask_step_commits_exactly_k_tokens():
    tokens = [MASK] * 5
    logits = [[float(i), 0.0] for i in range(5)]
    result = unmask_step(tokens, logits, 3)
    assert sum(1 for t in result if t == MASK) == 2


def test_unmask_step_leaves_known_tokens_alone():
    """Основа inpainting: заданные заранее токены переживают шаг нетронутыми."""
    tokens = [1, MASK, 1]
    logits = [[9.0, 0.0], [0.0, 9.0], [9.0, 0.0]]
    assert unmask_step(tokens, logits, 1) == [1, 1, 1]


def test_unmask_step_does_not_mutate_its_input():
    tokens = [MASK, MASK]
    unmask_step(tokens, [[0.0, 9.0], [0.0, 0.0]], 1)
    assert tokens == [MASK, MASK]


# -------------------------------------------------------------- sample_masked
def test_sample_masked_leaves_no_mask_behind():
    traces = sample_masked(make_predict(), [MASK] * 16, 8)
    assert MASK not in traces[-1]


def test_sample_masked_costs_T_forward_passes_not_one_per_token():
    """Ключевое обещание параллельного декодирования: 8 проходов на 64 токена."""
    calls = []
    sample_masked(make_predict(calls), [MASK] * 64, 8)
    assert len(calls) <= 8


def test_sample_masked_keeps_prefilled_tokens_which_is_inpainting():
    """Позиция 0 задана как 2, хотя заглушка предсказала бы 0. Она обязана выжить."""
    tokens = [MASK] * 8
    tokens[0] = 2
    traces = sample_masked(make_predict(), tokens, 4)
    assert traces[-1][0] == 2
    assert traces[-1][1] == 1  # соседнюю позицию сэмплер всё-таки заполнил сам


def test_sample_masked_trace_starts_with_the_input_state():
    tokens = [MASK] * 8
    traces = sample_masked(make_predict(), tokens, 4)
    assert traces[0] == tokens
    assert len(traces) >= 2


def test_sample_masked_on_a_full_sequence_calls_the_model_zero_times():
    calls = []
    traces = sample_masked(make_predict(calls), [0, 1, 2, 3], 8)
    assert calls == []
    assert traces == [[0, 1, 2, 3]]


# ----------------------------------------------------------- compression_ratio
def test_compression_ratio_worked_example():
    assert compression_ratio(512, 512, 1024, 16384) == pytest.approx(438.857, abs=1e-3)


def test_bigger_codebook_compresses_less():
    small = compression_ratio(512, 512, 1024, 1024)
    big = compression_ratio(512, 512, 1024, 65536)
    assert small > big


def test_more_tokens_compress_less():
    assert compression_ratio(512, 512, 512, 16384) > compression_ratio(512, 512, 2048, 16384)


def test_compression_ratio_rejects_a_degenerate_codebook():
    """vocab = 1 это log2 = 0 бит на токен и деление на ноль."""
    with pytest.raises(ValueError):
        compression_ratio(512, 512, 1024, 1)
