"""Тесты к уроку «Анти-спуфинг и водяные знаки в аудио». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    add_noise,
    bit_recovery_accuracy,
    chip_sequence,
    detect_watermark,
    eer,
    embed_watermark,
    is_suspicious,
    spectral_rolloff,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

PAYLOAD = [1, 0, 1, 1, 0, 1, 0, 0]


def host_signal(n=16000, freq=440.0, sr=16000.0, amp=0.5):
    """Чистая синусоида: «живая» запись, в которую нечего вкладывать."""
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


# ----------------------------------------------------------- spectral_rolloff
def test_rolloff_of_low_frequency_energy_is_the_first_bin():
    assert spectral_rolloff([10, 0, 0, 0]) == 0


def test_rolloff_follows_the_percentile():
    """Чем больше доля энергии, тем выше по спектру приходится подниматься."""
    spec = [1] * 100
    assert spectral_rolloff(spec, 0.5) < spectral_rolloff(spec, 0.95)


def test_rolloff_of_silence_is_zero_not_a_crash():
    """Ловушка: деление на нулевую суммарную энергию."""
    assert spectral_rolloff([0, 0, 0]) == 0


def test_rolloff_rejects_a_percentile_above_one():
    with pytest.raises(ValueError):
        spectral_rolloff([1, 2, 3], percentile=1.5)


# --------------------------------------------------------------- is_suspicious
def test_energy_smeared_into_the_top_reads_as_synthetic():
    assert is_suspicious([0.01] * 90 + [10] * 10) is True


def test_naturally_decaying_spectrum_reads_as_real():
    assert is_suspicious([1 / (k + 1) for k in range(100)]) is False


def test_raising_the_ratio_makes_the_detector_more_forgiving():
    spec = [0.01] * 90 + [10] * 10
    assert is_suspicious(spec, ratio=0.99) is False


# --------------------------------------------------------------- chip_sequence
def test_chips_are_only_plus_and_minus_one():
    assert set(chip_sequence(200, seed=3)) == {1.0, -1.0}


def test_chip_sequence_has_the_requested_length():
    assert len(chip_sequence(37, seed=1)) == 37


def test_same_seed_gives_the_same_key():
    """Без воспроизводимости детектор не найдёт то, что вложил генератор."""
    assert chip_sequence(64, seed=7) == chip_sequence(64, seed=7)


def test_different_seeds_give_different_keys():
    assert chip_sequence(64, seed=7) != chip_sequence(64, seed=8)


def test_chip_sequence_is_roughly_balanced():
    """Смещённый ключ добавил бы к записи постоянную составляющую — щелчок."""
    assert abs(sum(chip_sequence(10000, seed=0))) < 500


# -------------------------------------------------------------- embed_watermark
def test_watermarking_does_not_change_the_length():
    host = host_signal(800)
    assert len(embed_watermark(host, PAYLOAD)) == 800


def test_watermark_perturbation_never_exceeds_strength():
    """Неслышимость — не пожелание, а условие: знак живёт под -26 dBFS."""
    host = host_signal(800)
    marked = embed_watermark(host, PAYLOAD, strength=0.05)
    assert max(abs(a - b) for a, b in zip(marked, host)) == pytest.approx(0.05, abs=1e-9)


def test_embedding_does_not_mutate_the_original_recording():
    host = host_signal(800)
    before = list(host)
    embed_watermark(host, PAYLOAD)
    assert host == before


def test_embedding_rejects_a_non_binary_payload():
    with pytest.raises(ValueError):
        embed_watermark(host_signal(800), [1, 2, 0])


def test_embedding_rejects_a_signal_shorter_than_the_payload():
    with pytest.raises(ValueError):
        embed_watermark([0.0, 0.0], PAYLOAD)


# -------------------------------------------------------------------- add_noise
def test_add_noise_is_reproducible_from_the_same_rng():
    sig = host_signal(500)
    assert add_noise(sig, 10.0, random.Random(4)) == add_noise(sig, 10.0, random.Random(4))


def test_measured_snr_matches_the_requested_one():
    sig = host_signal(16000)
    noisy = add_noise(sig, 10.0, random.Random(0))
    ps = sum(x * x for x in sig) / len(sig)
    pe = sum((a - b) ** 2 for a, b in zip(noisy, sig)) / len(sig)
    assert 10 * math.log10(ps / pe) == pytest.approx(10.0, abs=0.5)


def test_lower_snr_means_a_louder_attack():
    sig = host_signal(4000)
    mild = add_noise(sig, 20.0, random.Random(1))
    harsh = add_noise(sig, 0.0, random.Random(1))
    err = lambda n: sum((a - b) ** 2 for a, b in zip(n, sig))
    assert err(harsh) > err(mild)


# ------------------------------------------------------------ detect_watermark
def test_payload_survives_a_clean_round_trip():
    marked = embed_watermark(host_signal(), PAYLOAD)
    score, bits = detect_watermark(marked, len(PAYLOAD))
    assert bits == PAYLOAD
    assert score > 0.8


def test_detector_stays_quiet_on_an_unmarked_recording():
    """Ложное срабатывание на живой записи хуже пропуска фейка."""
    score, _ = detect_watermark(host_signal(), len(PAYLOAD))
    assert score < 0.3


def test_watermark_survives_noise_at_ten_decibels():
    marked = embed_watermark(host_signal(), PAYLOAD)
    attacked = add_noise(marked, 10.0, random.Random(0))
    score, bits = detect_watermark(attacked, len(PAYLOAD))
    assert bit_recovery_accuracy(PAYLOAD, bits) == 1.0
    assert score > 0.8


def test_detector_without_the_key_recovers_nothing():
    """Атакующий не знает seed — корреляция для него шум около нуля."""
    marked = embed_watermark(host_signal(), PAYLOAD, seed=0)
    score, _ = detect_watermark(marked, len(PAYLOAD), seed=99)
    assert score < 0.3


def test_detector_rejects_a_nonpositive_payload_size():
    with pytest.raises(ValueError):
        detect_watermark(host_signal(100), 0)


# ------------------------------------------------------- bit_recovery_accuracy
def test_untouched_payload_scores_one():
    assert bit_recovery_accuracy([1, 0, 1], [1, 0, 1]) == APPROX(1.0)


def test_fully_inverted_payload_scores_zero():
    assert bit_recovery_accuracy([1, 0, 1], [0, 1, 0]) == APPROX(0.0)


def test_half_recovered_payload_is_no_better_than_a_coin():
    assert bit_recovery_accuracy([1, 0], [1, 1]) == APPROX(0.5)


def test_bit_recovery_rejects_payloads_of_different_length():
    with pytest.raises(ValueError):
        bit_recovery_accuracy([1, 0, 1], [1, 0])


# -------------------------------------------------------------------------- eer
def test_perfectly_separated_scores_give_zero_eer():
    assert eer([0.9, 0.95], [0.1, 0.2]) == APPROX(0.0)


def test_indistinguishable_scores_give_half():
    """Совпадающие распределения — это подбрасывание монетки, EER 0.5."""
    assert eer([0.0, 1.0], [0.0, 1.0]) == APPROX(0.5)


def test_inverted_detector_gives_eer_one():
    """Фейкам ставит выше, чем живым: ошибается всегда, а не иногда."""
    assert eer([0.1, 0.2], [0.9, 0.95]) == APPROX(1.0)


def test_eer_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        eer([], [0.1, 0.2])
