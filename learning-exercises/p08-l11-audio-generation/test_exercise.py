"""Тесты к уроку «Генерация звука: нейронный кодек и токенный авторегрессор». Правь exercise.py."""

import random

import pytest

from exercise import (
    codec_token_count,
    delay_streams,
    generate_tokens,
    next_token_probs,
    rvq_decode,
    rvq_encode,
    train_bigram,
    undelay_streams,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

BOOKS = [[0.0, 0.5, 1.0], [-0.2, -0.1, 0.0, 0.1, 0.2], [-0.05, 0.0, 0.05]]


def rvq_error(value, codebooks):
    """Абсолютная ошибка восстановления после прохода по codebooks."""
    return abs(value - rvq_decode(rvq_encode(value, codebooks), codebooks))


# ------------------------------------------------------- codec_token_count
def test_codec_token_count_multiplies_frames_by_codebooks():
    assert codec_token_count(5) == 3000
    assert codec_token_count(1, 50, 4) == 200


def test_codec_token_count_is_linear_in_duration():
    assert codec_token_count(30) == 6 * codec_token_count(5)


def test_codec_token_count_doubles_with_twice_the_codebooks():
    """Каждый слой RVQ — отдельный поток индексов, длина растёт линейно."""
    assert codec_token_count(10, 75, 16) == 2 * codec_token_count(10, 75, 8)


def test_codec_beats_raw_waveform_by_orders_of_magnitude():
    """Пять секунд при 24 кГц — 120 000 отсчётов; кодек оставляет 3000."""
    raw_samples = 5 * 24000
    assert codec_token_count(5, 75, 8) * 30 < raw_samples


# -------------------------------------------------------------- rvq_encode
def test_rvq_encode_picks_the_nearest_code_at_the_first_layer():
    assert rvq_encode(0.62, BOOKS)[0] == 1


def test_rvq_encode_second_layer_quantizes_the_residual_not_the_value():
    """Остаток 0.62 - 0.5 = 0.12, и второй слой берёт код 0.1, а не 0.5."""
    assert rvq_encode(0.62, BOOKS)[:2] == [1, 3]


def test_rvq_encode_returns_one_index_per_codebook():
    assert len(rvq_encode(0.37, BOOKS)) == len(BOOKS)


def test_rvq_encode_of_an_exactly_representable_value_is_exact():
    assert rvq_error(0.6, BOOKS) == APPROX(0.0)


# -------------------------------------------------------------- rvq_decode
def test_rvq_decode_sums_the_selected_codes():
    assert rvq_decode([1, 3], BOOKS[:2]) == APPROX(0.6)
    assert rvq_decode([2, 0], BOOKS[:2]) == APPROX(0.8)


def test_rvq_error_shrinks_as_layers_are_added():
    """Каждый слой снимает остаток предыдущего — ошибка не растёт никогда."""
    errors = [rvq_error(0.62, BOOKS[:k]) for k in (1, 2, 3)]
    assert errors[0] > errors[1]
    assert errors[2] <= errors[1]


def test_rvq_cannot_repair_a_bad_first_layer():
    """Значение вне диапазона первого кодбука: качество кодека — потолок всего."""
    assert rvq_error(5.0, BOOKS) > 3.0


# ------------------------------------------------------------ train_bigram
def test_train_bigram_counts_observed_transitions():
    assert train_bigram([[0, 1, 2]], 3) == [[1, 2, 1], [1, 1, 2], [1, 1, 1]]


def test_train_bigram_never_leaves_a_zero():
    """Сглаживание Лапласа: неувиденный переход маловероятен, но возможен."""
    counts = train_bigram([[0, 0, 0]], 4)
    assert all(x > 0 for row in counts for x in row)


def test_train_bigram_is_directional():
    """counts[a][b] — переход из a в b; обратный переход не наблюдался."""
    counts = train_bigram([[0, 1]], 2)
    assert counts[0][1] > counts[1][0]


def test_train_bigram_learns_the_style_pattern():
    """На «речевом» чередовании argmax каждой строки — следующий токен по кругу."""
    counts = train_bigram([[i % 8 for i in range(24)]] * 4, 8)
    for i in range(8):
        assert max(range(8), key=lambda j: counts[i][j]) == (i + 1) % 8


# -------------------------------------------------------- next_token_probs
def test_next_token_probs_normalizes_the_counts_row():
    counts = train_bigram([[0, 1], [0, 1]], 2)
    assert next_token_probs(counts, 0) == APPROX([0.25, 0.75])


def test_next_token_probs_always_sums_to_one():
    counts = train_bigram([[0, 1, 2, 0]], 3)
    for temp in (0.2, 1.0, 5.0):
        assert sum(next_token_probs(counts, 1, temp)) == pytest.approx(1.0, abs=1e-9)


def test_low_temperature_sharpens_and_high_temperature_flattens():
    counts = train_bigram([[0, 1], [0, 1]], 2)
    cold = max(next_token_probs(counts, 0, 0.2))
    warm = max(next_token_probs(counts, 0, 1.0))
    hot = max(next_token_probs(counts, 0, 50.0))
    assert cold > warm > hot > 0.5


def test_next_token_probs_rejects_a_nonpositive_temperature():
    counts = train_bigram([[0, 1]], 2)
    with pytest.raises(ValueError):
        next_token_probs(counts, 0, 0.0)


def test_next_token_probs_survives_a_tiny_temperature_on_a_big_vocabulary():
    """Наивное p ** (1/temperature) здесь обнуляет всё и падает делением на ноль."""
    counts = train_bigram([[0, 7]], 40)
    probs = next_token_probs(counts, 0, temperature=0.002)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert probs[7] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------- generate_tokens
def test_generate_tokens_starts_with_the_prompt_token_and_has_the_asked_length():
    counts = train_bigram([[0, 1, 2, 3]], 4)
    out = generate_tokens(counts, 2, 10, random.Random(0))
    assert out[0] == 2
    assert len(out) == 10


def test_generate_tokens_of_length_one_is_just_the_prompt():
    counts = train_bigram([[0, 1]], 2)
    assert generate_tokens(counts, 1, 1, random.Random(0)) == [1]


def test_generate_tokens_is_reproducible_for_a_given_seed():
    counts = train_bigram([[0, 1, 2, 3, 0]], 4)
    a = generate_tokens(counts, 0, 30, random.Random(7))
    b = generate_tokens(counts, 0, 30, random.Random(7))
    assert a == b


def test_generate_tokens_stays_inside_the_vocabulary():
    counts = train_bigram([[0, 1, 2, 3, 0]], 4)
    assert all(0 <= t < 4 for t in generate_tokens(counts, 0, 200, random.Random(3)))


def test_cold_sampling_reproduces_the_trained_style():
    """Стиль «ramp» с шагом 3: холодная температура обязана выдать его точно."""
    counts = train_bigram([[(i * 3) % 8 for i in range(24)]] * 4, 8)
    out = generate_tokens(counts, 0, 6, random.Random(0), temperature=0.01)
    assert out == [0, 3, 6, 1, 4, 7]


def test_two_styles_generate_different_sequences():
    """Обусловливание на стиль — это просто разные матрицы переходов."""
    speech = train_bigram([[i % 8 for i in range(24)]] * 4, 8)
    music = train_bigram([[(i * 3) % 8 for i in range(24)]] * 4, 8)
    cold = dict(temperature=0.01)
    assert generate_tokens(speech, 0, 6, random.Random(0), **cold) != \
        generate_tokens(music, 0, 6, random.Random(0), **cold)


# ----------------------------------------------------------- delay_streams
def test_delay_streams_shifts_each_stream_by_its_index():
    assert delay_streams([[1, 2], [3, 4]], pad=0) == [[1, 2, 0], [0, 3, 4]]


def test_delay_streams_pads_every_stream_to_the_same_length():
    delayed = delay_streams([[1, 2, 3]] * 4, pad=-1)
    assert {len(s) for s in delayed} == {3 + 4 - 1}


def test_delay_streams_puts_exactly_k_pads_in_front_of_stream_k():
    delayed = delay_streams([[5, 5]] * 3, pad=-1)
    for k, stream in enumerate(delayed):
        assert stream[:k] == [-1] * k
        assert stream[k] == 5


def test_delayed_layout_is_far_shorter_than_a_flat_concatenation():
    """Плоская склейка дала бы K*T токенов, сдвиг — всего T+K-1 колонок."""
    streams = [[0] * 1500 for _ in range(8)]
    assert len(delay_streams(streams)[0]) == 1507 < 8 * 1500


def test_single_stream_needs_no_delay():
    assert delay_streams([[1, 2, 3]], pad=0) == [[1, 2, 3]]


# --------------------------------------------------------- undelay_streams
def test_undelay_is_the_inverse_of_delay():
    streams = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert undelay_streams(delay_streams(streams, pad=-1)) == streams


def test_undelay_drops_the_padding():
    assert undelay_streams([[1, 2, 0], [0, 3, 4]]) == [[1, 2], [3, 4]]


def test_undelay_roundtrip_holds_for_eight_codebooks():
    streams = [[k * 10 + t for t in range(12)] for k in range(8)]
    assert undelay_streams(delay_streams(streams)) == streams
