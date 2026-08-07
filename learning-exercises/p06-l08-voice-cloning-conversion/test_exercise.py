"""Тесты к уроку «Клонирование и конверсия голоса». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    bit_accuracy,
    consent_gate,
    detect_watermark,
    embed_watermark,
    knn_convert,
    secs,
    speaker_embedding,
    swap_speaker,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Разворачивает список списков в плоский список: pytest.approx не умеет вложенные."""
    return [x for row in M for x in row]


# --------------------------------------------------------- speaker_embedding
def test_speaker_embedding_is_unit_length():
    emb = speaker_embedding([[3.0, 4.0], [3.0, 4.0]])
    assert math.sqrt(sum(x * x for x in emb)) == APPROX(1.0)


def test_speaker_embedding_ignores_frame_order():
    """Порядок кадров — это то, ЧТО сказано. Тембр от него не зависит."""
    frames = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]
    assert speaker_embedding(frames) == APPROX(speaker_embedding(list(reversed(frames))))


def test_speaker_embedding_is_scale_invariant():
    """Громче записали — диктор тот же. Нормировка это и обеспечивает."""
    frames = [[1.0, 2.0], [3.0, 0.5]]
    loud = [[10 * x for x in f] for f in frames]
    assert speaker_embedding(loud) == APPROX(speaker_embedding(frames))


def test_speaker_embedding_rejects_input_without_a_direction():
    """Ни кадров, ни ненулевого среднего — нормировать нечего."""
    with pytest.raises(ValueError):
        speaker_embedding([])
    with pytest.raises(ValueError):
        speaker_embedding([[1.0, 0.0], [-1.0, 0.0]])


# ---------------------------------------------------------------------- secs
def test_secs_of_the_same_speaker_is_one():
    assert secs([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == APPROX(1.0)


def test_secs_of_orthogonal_speakers_is_zero():
    assert secs([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_secs_does_not_depend_on_vector_length():
    """Косинус — про угол. Умножить один вектор на 5 ничего не меняет."""
    a, b = [1.0, 2.0], [3.0, -1.0]
    assert secs(a, [5 * x for x in b]) == APPROX(secs(a, b))


def test_secs_above_the_070_threshold_means_indistinguishable():
    """Порог из урока: SECS > 0.70 — клон неотличим для большинства слушателей."""
    assert secs([1.0, 0.0], [1.0, 1.0]) > 0.70


def test_secs_rejects_vectors_without_a_defined_angle():
    with pytest.raises(ValueError):
        secs([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        secs([1.0, 0.0], [1.0, 0.0, 0.0])


# --------------------------------------------------------------- knn_convert
def test_knn_convert_picks_the_nearest_pool_frame():
    assert flat(knn_convert([[0.0]], [[5.0], [0.1]])) == APPROX([0.1])


def test_knn_convert_averages_k_neighbours():
    assert flat(knn_convert([[0.0]], [[1.0], [3.0]], k=2)) == APPROX([2.0])


def test_knn_convert_preserves_the_number_of_frames():
    """Конверсия меняет КТО говорит, а не ЧТО и не как долго."""
    source = [[0.0], [1.0], [2.0], [3.0]]
    assert len(knn_convert(source, [[0.5], [2.5]])) == len(source)


def test_knn_convert_output_frames_come_from_the_target_pool():
    """При k=1 каждый выданный кадр — буквально кадр целевого диктора."""
    pool = [[0.0, 1.0], [4.0, 4.0], [-2.0, 0.0]]
    out = knn_convert([[0.1, 0.9], [3.5, 4.2], [-1.9, 0.1]], pool)
    assert out == pool


def test_knn_convert_breaks_ties_by_lower_index():
    """Ровно посередине между двумя кадрами — берём тот, что раньше в пуле."""
    assert flat(knn_convert([[0.5]], [[0.0], [1.0]])) == APPROX([0.0])


def test_knn_convert_rejects_a_pool_that_cannot_supply_k_neighbours():
    with pytest.raises(ValueError):
        knn_convert([[0.0]], [[1.0], [2.0]], k=3)
    with pytest.raises(ValueError):
        knn_convert([[0.0]], [])


# -------------------------------------------------------------- swap_speaker
def test_swap_speaker_moves_a_frame_to_the_target():
    assert flat(swap_speaker([[1.0, 0.0]], [1.0, 0.0], [0.0, 1.0])) == APPROX([0.0, 1.0])


def test_swap_speaker_preserves_differences_between_frames():
    """Контент — это то, чем кадры отличаются. Общий сдвиг его не трогает."""
    frames = [[1.0, 0.0], [4.0, 2.0]]
    out = swap_speaker(frames, [1.0, 1.0], [-3.0, 0.5])
    before = [b - a for a, b in zip(frames[0], frames[1])]
    after = [b - a for a, b in zip(out[0], out[1])]
    assert after == APPROX(before)


def test_swap_speaker_to_the_same_speaker_changes_nothing():
    frames = [[1.0, 2.0], [3.0, 4.0]]
    emb = [0.6, 0.8]
    assert flat(swap_speaker(frames, emb, emb)) == APPROX(flat(frames))


def test_swap_speaker_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        swap_speaker([[1.0, 0.0]], [1.0, 0.0], [0.0, 1.0, 0.0])


# ------------------------------------------------- embed / detect_watermark
def test_watermark_roundtrip_recovers_the_bits():
    rng = random.Random(0)
    wav = [rng.uniform(-1.0, 1.0) for _ in range(64)]
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    assert detect_watermark(embed_watermark(wav, bits), len(bits)) == bits


def test_watermark_is_inaudible():
    """Каждый сэмпл сдвигается не больше чем на delta/2 — потому знак и не слышно."""
    rng = random.Random(1)
    wav = [rng.uniform(-1.0, 1.0) for _ in range(32)]
    marked = embed_watermark(wav, [1, 0], delta=0.02)
    assert max(abs(a - b) for a, b in zip(wav, marked)) <= 0.02 / 2 + 1e-12


def test_watermark_survives_light_noise():
    """Знак должен пережить перекодирование — здесь его моделирует шум."""
    rng = random.Random(2)
    wav = [rng.uniform(-1.0, 1.0) for _ in range(60)]
    bits = [1, 1, 0, 1, 0]
    marked = embed_watermark(wav, bits, delta=0.04)
    noisy = [x + rng.uniform(-0.005, 0.005) for x in marked]
    assert detect_watermark(noisy, len(bits), delta=0.04) == bits


def test_watermark_majority_vote_outvotes_one_broken_repeat():
    """Бит лежит в трёх сэмплах; один испорчен — голосование всё равно право."""
    wav = [0.0] * 6
    marked = embed_watermark(wav, [1, 0], delta=0.02)
    marked[4] = -0.005  # носитель бита 0 подменён на решётку бита 1
    assert detect_watermark(marked, 2, delta=0.02) == [1, 0]


def test_embed_watermark_rejects_bad_arguments():
    with pytest.raises(ValueError):
        embed_watermark([0.1, 0.2], [])
    with pytest.raises(ValueError):
        embed_watermark([0.1, 0.2], [1, 0], delta=0.0)
    with pytest.raises(ValueError):
        embed_watermark([0.1, 0.2], [1, 2])


def test_detect_watermark_needs_a_carrier_for_every_bit():
    with pytest.raises(ValueError):
        detect_watermark([0.01, -0.01], 5)


# -------------------------------------------------------------- bit_accuracy
def test_bit_accuracy_of_a_perfect_read_is_one():
    assert bit_accuracy([1, 0, 1], [1, 0, 1]) == APPROX(1.0)


def test_bit_accuracy_counts_matching_positions():
    assert bit_accuracy([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)


def test_bit_accuracy_rejects_uncomparable_bit_strings():
    with pytest.raises(ValueError):
        bit_accuracy([1, 0], [1, 0, 1])
    with pytest.raises(ValueError):
        bit_accuracy([], [])


# --------------------------------------------------------------- consent_gate
def _sign(record):
    """Заглушка вместо криптоподписи: детерминированная функция от полей."""
    return f"{record['speaker_id']}:{record['expires_ts']}"


def _record(speaker_id="rohit", expires_ts=2000, signature=None):
    rec = {"speaker_id": speaker_id, "expires_ts": expires_ts}
    rec["signature"] = _sign(rec) if signature is None else signature
    return rec


def test_consent_gate_accepts_a_valid_record_up_to_the_expiry_moment():
    """Момент ровно в expires_ts ещё действителен — граница включительно."""
    assert consent_gate(_record(), "rohit", 1000, _sign) is True
    assert consent_gate(_record(expires_ts=1000), "rohit", 1000, _sign) is True


def test_consent_gate_rejects_a_forged_signature():
    with pytest.raises(ValueError, match="invalid signature"):
        consent_gate(_record(signature="fake"), "rohit", 1000, _sign)


def test_consent_gate_rejects_a_different_speaker():
    with pytest.raises(ValueError, match="speaker mismatch"):
        consent_gate(_record(), "alice", 1000, _sign)


def test_consent_gate_rejects_expired_consent():
    with pytest.raises(ValueError, match="consent expired"):
        consent_gate(_record(expires_ts=500), "rohit", 1000, _sign)


def test_consent_gate_checks_the_signature_first():
    """У неподписанной записи остальные поля читать бессмысленно."""
    bad = _record(speaker_id="alice", expires_ts=1, signature="fake")
    with pytest.raises(ValueError, match="invalid signature"):
        consent_gate(bad, "rohit", 1000, _sign)


def test_consent_gate_raises_on_a_record_without_expiry():
    """Молча пропустить бессрочное согласие нельзя — пусть падает."""
    rec = {"speaker_id": "rohit"}
    rec["signature"] = "ok"
    with pytest.raises(KeyError):
        consent_gate(rec, "rohit", 1000, lambda r: "ok")
