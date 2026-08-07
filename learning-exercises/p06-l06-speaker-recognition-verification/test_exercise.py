"""Тесты к уроку «Распознавание и верификация диктора». Правь exercise.py."""

import math

import pytest

from exercise import (
    aam_margin_logit,
    cosine_score,
    enroll_speaker,
    equal_error_rate,
    false_rates,
    identify,
    l2_normalize,
    verify,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ANGLE = lambda x: pytest.approx(x, abs=1e-5)


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


# -------------------------------------------------------------- l2_normalize
def test_normalized_vector_has_unit_length():
    assert _norm(l2_normalize([3.0, 4.0])) == pytest.approx(1.0, abs=1e-12)


def test_normalization_keeps_the_direction():
    assert l2_normalize([3.0, 4.0]) == APPROX([0.6, 0.8])


def test_normalizing_twice_changes_nothing():
    once = l2_normalize([1.0, 2.0, 3.0])
    assert l2_normalize(once) == APPROX(once)


def test_zero_vector_does_not_divide_by_zero():
    """Ловушка: тихий или обрезанный клип даёт нулевой эмбеддинг."""
    assert l2_normalize([0.0, 0.0]) == APPROX([0.0, 0.0])


# -------------------------------------------------------------- cosine_score
def test_same_direction_scores_one():
    assert cosine_score([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0, abs=1e-12)


def test_opposite_direction_scores_minus_one():
    assert cosine_score([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-12)


def test_score_is_the_dot_product_of_normalized_vectors():
    """Ловушка урока: без L2-нормализации в скоре участвует громкость записи."""
    a, b = [1.0, 2.0, 3.0], [-1.0, 4.0, 0.5]
    ua, ub = l2_normalize(a), l2_normalize(b)
    assert cosine_score(a, b) == APPROX(sum(x * y for x, y in zip(ua, ub)))


def test_score_ignores_embedding_magnitude():
    quiet = [0.001, 0.002]
    loud = [100.0, 200.0]
    assert cosine_score(quiet, loud) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------ enroll_speaker
def test_enrollment_returns_a_unit_vector():
    assert _norm(enroll_speaker([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(
        1.0, abs=1e-12
    )


def test_single_clean_sample_is_just_normalized():
    assert enroll_speaker([[3.0, 4.0]]) == APPROX([0.6, 0.8])


def test_averaging_several_samples_beats_one_noisy_one():
    """Ловушка: одна шумная запись отравляет эталон. Три записи гасят шум."""
    true_voice = [1.0, 0.0, 0.0]
    clean_a = [1.0, 0.1, 0.0]
    clean_b = [1.0, -0.1, 0.0]
    noisy = [1.0, 0.0, 1.2]
    anchor = enroll_speaker([clean_a, clean_b, noisy])
    assert cosine_score(anchor, true_voice) > cosine_score(noisy, true_voice)


def test_enrollment_without_samples_is_rejected():
    with pytest.raises(ValueError):
        enroll_speaker([])


# --------------------------------------------------------------------- verify
def test_same_speaker_passes_the_threshold():
    assert verify([1.0, 0.0], [2.0, 0.0], 0.5) is True


def test_different_speaker_is_rejected():
    assert verify([1.0, 0.0], [0.0, 1.0], 0.5) is False


def test_threshold_is_inclusive():
    """Скор ровно на пороге — это «принять». Граница должна быть одна и та же
    и здесь, и в false_rates, иначе EER посчитается не по той системе."""
    assert verify([1.0, 0.0], [1.0, 0.0], 1.0) is True


def test_raising_the_threshold_only_makes_the_system_stricter():
    enroll, test = [1.0, 0.2], [1.0, 0.0]
    accepted = [verify(enroll, test, t / 10) for t in range(0, 11)]
    assert all(a >= b for a, b in zip(accepted, accepted[1:]))


# ---------------------------------------------------------------- false_rates
def test_a_threshold_below_everything_lets_all_imposters_in():
    far, frr = false_rates([0.8, 0.9], [0.1, 0.2], 0.0)
    assert (far, frr) == APPROX((1.0, 0.0))


def test_a_threshold_above_everything_rejects_all_genuine_users():
    far, frr = false_rates([0.8, 0.9], [0.1, 0.2], 1.0)
    assert (far, frr) == APPROX((0.0, 1.0))


def test_a_threshold_between_the_classes_makes_no_errors():
    assert false_rates([0.8, 0.9], [0.1, 0.2], 0.5) == APPROX((0.0, 0.0))


def test_far_falls_and_frr_rises_with_the_threshold():
    """Две ручки одного рычага: строже к чужим — строже и к своим."""
    same = [0.3, 0.5, 0.7, 0.9]
    diff = [0.2, 0.4, 0.6, 0.8]
    rates = [false_rates(same, diff, t / 10) for t in range(0, 11)]
    fars = [r[0] for r in rates]
    frrs = [r[1] for r in rates]
    assert all(a >= b for a, b in zip(fars, fars[1:]))
    assert all(a <= b for a, b in zip(frrs, frrs[1:]))


# ----------------------------------------------------------- equal_error_rate
def test_separable_scores_give_a_zero_eer():
    """Ловушка: инициализация «лучшего» значением (1.0, 1.0) даёт нулевой
    разрыв, который строгое «меньше» никогда не побьёт, и EER всегда 1.0."""
    eer, _ = equal_error_rate([0.9, 0.95], [0.1, 0.2])
    assert eer == APPROX(0.0)


def test_at_the_returned_threshold_far_equals_frr():
    """Определение EER: это точка пересечения двух кривых ошибок."""
    same = [i / 100 for i in range(50, 100, 5)]
    diff = [i / 100 for i in range(0, 45, 5)] + [0.55]
    eer, threshold = equal_error_rate(same, diff)
    far, frr = false_rates(same, diff, threshold)
    assert far == APPROX(frr)
    assert eer == APPROX(far)


def test_eer_reports_the_threshold_that_produced_it():
    same = [i / 100 for i in range(50, 100, 5)]
    diff = [i / 100 for i in range(0, 45, 5)] + [0.55]
    assert equal_error_rate(same, diff) == APPROX((0.1, 0.55))


def test_better_separated_speakers_give_a_lower_eer():
    good = equal_error_rate([0.8, 0.85, 0.9, 0.95], [0.0, 0.05, 0.1, 0.15])[0]
    bad = equal_error_rate([0.4, 0.5, 0.6, 0.7], [0.35, 0.45, 0.55, 0.65])[0]
    assert good < bad


def test_eer_stays_inside_zero_and_one():
    same = [0.1, 0.3, 0.5, 0.7]
    diff = [0.2, 0.4, 0.6, 0.8]
    eer, _ = equal_error_rate(same, diff)
    assert 0.0 <= eer <= 1.0


# ------------------------------------------------------------------- identify
def test_identification_returns_the_closest_speaker():
    bank = [[1.0, 0.0], [0.0, 1.0]]
    assert identify([1.0, 0.1], bank, ["alice", "bob"]) == "alice"


def test_without_a_threshold_someone_is_always_named():
    """Ловушка: closed-set всегда кого-то называет, даже если говорит чужой."""
    bank = [[1.0, 0.0], [0.9, 0.1]]
    assert identify([-1.0, 0.0], bank, ["alice", "bob"]) is not None


def test_an_unknown_speaker_is_rejected_in_open_set():
    bank = [[1.0, 0.1], [1.0, 0.0]]
    assert identify([0.0, 1.0], bank, ["alice", "bob"], threshold=0.5) is None


def test_identification_ignores_the_loudness_of_the_test_clip():
    bank = [[1.0, 0.0], [0.0, 1.0]]
    loud = identify([50.0, 5.0], bank, ["alice", "bob"])
    quiet = identify([0.005, 0.0005], bank, ["alice", "bob"])
    assert loud == quiet == "alice"


# ------------------------------------------------------------ aam_margin_logit
def test_zero_margin_is_plain_scaled_cosine():
    assert aam_margin_logit(0.5, margin=0.0, scale=30.0) == ANGLE(15.0)


def test_margin_lowers_the_logit_of_the_correct_class():
    """В этом весь смысл AAM: задача усложняется, и сеть разводит классы дальше."""
    assert aam_margin_logit(1.0, margin=0.2, scale=30.0) < aam_margin_logit(
        1.0, margin=0.0, scale=30.0
    )


def test_bigger_margin_keeps_lowering_the_logit():
    logits = [aam_margin_logit(0.9, margin=m / 10, scale=30.0) for m in range(0, 8)]
    assert all(a > b for a, b in zip(logits, logits[1:]))


def test_a_cosine_slightly_above_one_is_clamped():
    """Ловушка: cos приходит из float-арифметики и бывает 1.0000000002,
    а acos от такого — это ValueError, а не единица."""
    assert aam_margin_logit(1.0000000002, margin=0.2, scale=30.0) == ANGLE(
        30.0 * math.cos(0.2)
    )
