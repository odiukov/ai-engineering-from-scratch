"""Тесты к уроку «Chameleon и ранняя фузия: картинка как токены». Правь exercise.py."""

import pytest

from exercise import (
    BOI,
    EOI,
    TEXT_VOCAB,
    compression_ratio,
    decode_document,
    dequantize,
    encode_document,
    nearest_code,
    qk_norm,
    quantize,
    reconstruction_mse,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CODEBOOK = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in rows for x in row]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------- nearest_code
def test_nearest_code_picks_the_closest_entry():
    assert nearest_code([0.9, 0.1], CODEBOOK) == 1


def test_nearest_code_breaks_ties_toward_the_lower_index():
    """Иначе токенизация одной и той же картинки поплывёт между запусками."""
    assert nearest_code([0.5, 0.0], [[0.0, 0.0], [1.0, 0.0]]) == 0


def test_nearest_code_rejects_an_empty_codebook():
    with pytest.raises(ValueError):
        nearest_code([0.5, 0.0], [])


def test_nearest_code_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        nearest_code([0.5, 0.0, 0.1], CODEBOOK)


# ----------------------------------------------------------------- quantize
def test_quantize_maps_every_vector():
    assert quantize([[0.9, 0.1], [0.1, 0.9]], CODEBOOK) == [1, 2]


def test_quantize_of_codebook_entries_returns_their_own_indices():
    assert quantize(CODEBOOK, CODEBOOK) == [0, 1, 2, 3]


def test_quantize_is_stable_after_a_round_trip():
    """Второе квантование уже ничего не меняет — потеря происходит один раз."""
    once = quantize([[0.9, 0.1], [0.2, 0.8]], CODEBOOK)
    twice = quantize(dequantize(once, CODEBOOK), CODEBOOK)
    assert twice == once


# --------------------------------------------------------------- dequantize
def test_dequantize_returns_the_codebook_vectors():
    assert flat(dequantize([1, 0], CODEBOOK)) == APPROX([1.0, 0.0, 0.0, 0.0])


def test_dequantize_returns_copies_not_the_codebook_itself():
    """Ловушка: вернув сами записи, ты дашь испортить книгу для всех картинок."""
    book = [[1.0, 2.0]]
    restored = dequantize([0], book)
    restored[0][0] = 99.0
    assert book[0][0] == APPROX(1.0)


def test_dequantize_lands_only_on_codebook_entries():
    """Реконструкция не может вернуть исходный вектор — только запись книги."""
    restored = dequantize(quantize([[0.9, 0.1]], CODEBOOK), CODEBOOK)
    assert restored[0] in CODEBOOK


# --------------------------------------------------------- reconstruction_mse
def test_reconstruction_mse_is_zero_on_exact_matches():
    assert reconstruction_mse(CODEBOOK, CODEBOOK) == APPROX(0.0)


def test_reconstruction_mse_on_a_midpoint():
    assert reconstruction_mse([[0.5, 0.0]], [[0.0, 0.0], [1.0, 0.0]]) == APPROX(0.125)


def test_a_bigger_codebook_never_reconstructs_worse():
    """Потолок реконструкции двигается только вверх — вот почему Emu3 выиграл."""
    small = [[0.0, 0.0], [1.0, 1.0]]
    big = small + [[0.5, 0.5], [0.2, 0.9]]
    vectors = [[0.4, 0.6], [0.9, 0.2], [0.1, 0.1]]
    assert reconstruction_mse(vectors, big) <= reconstruction_mse(vectors, small)


def test_reconstruction_mse_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        reconstruction_mse([], CODEBOOK)


# ---------------------------------------------------------- compression_ratio
def test_compression_ratio_for_chameleon():
    assert compression_ratio(512, 512, 1024, 8192) == pytest.approx(472.6, abs=0.1)


def test_emu3_compresses_less_than_chameleon():
    """Меньше сжатие — выше потолок качества: весь компромисс урока одной строкой."""
    chameleon = compression_ratio(512, 512, 1024, 8192)
    emu3 = compression_ratio(512, 512, 4096, 32768)
    assert emu3 < chameleon


def test_compression_ratio_falls_when_the_codebook_grows():
    assert compression_ratio(512, 512, 1024, 32768) < compression_ratio(
        512, 512, 1024, 8192
    )


def test_compression_ratio_rejects_a_single_entry_codebook():
    """При K = 1 бит на токен ноль, и деление сорвалось бы."""
    with pytest.raises(ValueError):
        compression_ratio(512, 512, 1024, 1)


# ----------------------------------------------------------- encode_document
def test_encode_document_shifts_image_codes_and_wraps_them():
    assert encode_document([("text", [1, 2]), ("image", [0, 5])]) == [
        1,
        2,
        BOI,
        TEXT_VOCAB + 0,
        TEXT_VOCAB + 5,
        EOI,
    ]


def test_encode_document_keeps_text_ids_untouched():
    assert encode_document([("text", [0, 31])]) == [0, 31]


def test_encode_document_rejects_a_code_outside_the_codebook():
    with pytest.raises(ValueError):
        encode_document([("image", [99])])


def test_encode_document_rejects_an_unknown_chunk_kind():
    with pytest.raises(ValueError):
        encode_document([("audio", [1, 2])])


# ----------------------------------------------------------- decode_document
def test_decode_document_round_trips_through_encode():
    parts = [("text", [1, 2]), ("image", [0, 5]), ("text", [7])]
    assert decode_document(encode_document(parts)) == parts


def test_decode_document_merges_consecutive_text_runs():
    """Два текстовых куска подряд неотличимы от одного — так и должно быть."""
    ids = encode_document([("text", [1]), ("text", [2])])
    assert decode_document(ids) == [("text", [1, 2])]


def test_decode_document_rejects_a_closing_tag_without_an_opening_one():
    with pytest.raises(ValueError):
        decode_document([1, EOI])


def test_decode_document_rejects_an_unterminated_image():
    with pytest.raises(ValueError):
        decode_document([BOI, TEXT_VOCAB + 3])


def test_decode_document_rejects_a_text_id_inside_an_image():
    """Молча «починить» такую последовательность значит отправить мусор в пиксели."""
    with pytest.raises(ValueError):
        decode_document([BOI, 3, EOI])


# ------------------------------------------------------------------- qk_norm
def test_qk_norm_centers_the_vector():
    normed = qk_norm([1.0, 3.0, 8.0, 4.0])
    assert sum(normed) == pytest.approx(0.0, abs=1e-9)


def test_qk_norm_gives_unit_variance():
    normed = qk_norm([1.0, 3.0, 8.0, 4.0])
    mean = sum(normed) / len(normed)
    variance = sum((x - mean) ** 2 for x in normed) / len(normed)
    assert variance == pytest.approx(1.0, abs=1e-4)


def test_qk_norm_of_a_constant_vector_is_zero_not_nan():
    """Дисперсия ровно ноль: без eps здесь было бы 0/0."""
    assert qk_norm([5.0, 5.0, 5.0]) == APPROX([0.0, 0.0, 0.0])


def test_qk_norm_ignores_the_scale_of_its_input():
    assert qk_norm([10.0, 30.0, 50.0]) == pytest.approx(
        qk_norm([1.0, 3.0, 5.0]), abs=1e-4
    )


def test_qk_norm_bounds_the_attention_logit():
    """Ради этого QK-Norm и придуман: логит внимания не уносит в миллионы."""
    q = [1000.0, -2000.0, 3000.0, 500.0]
    k = [900.0, -1800.0, 2500.0, 400.0]
    assert dot(q, k) > 1e6
    assert abs(dot(qk_norm(q), qk_norm(k))) <= len(q) + 1e-6


def test_qk_norm_rejects_an_empty_vector():
    with pytest.raises(ValueError):
        qk_norm([])
