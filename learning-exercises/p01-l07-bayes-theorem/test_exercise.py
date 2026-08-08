"""Тесты к уроку «Теорема Байеса». Правь exercise.py."""

import pytest

from exercise import (
    bayes_posterior,
    beta_map,
    beta_mean,
    beta_update,
    laplace_probability,
    mle_probability,
    naive_bayes_predict,
    sequential_posterior,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TRAIN_DOCS = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]
TRAIN_LABELS = ["spam"] * 5 + ["ham"] * 7


# ---------------------------------------------------------- bayes_posterior
def test_bayes_posterior_medical_test_stays_below_one_percent():
    """Классика: тест на 99% точный, но болезнь редкая — вера всего 1%."""
    assert bayes_posterior(0.0001, 0.99, 0.01) == pytest.approx(0.0098039, abs=1e-6)


def test_bayes_posterior_spam_word():
    assert bayes_posterior(0.3, 0.05, 0.001) == pytest.approx(0.955414, abs=1e-6)


def test_bayes_posterior_with_a_useless_test_returns_the_prior():
    """Если P(+|H) = P(+|не H), улика ничего не говорит — вера не меняется."""
    assert bayes_posterior(0.3, 0.2, 0.2) == APPROX(0.3)


def test_bayes_posterior_without_false_positives_is_certainty():
    assert bayes_posterior(0.4, 0.9, 0.0) == APPROX(1.0)


def test_bayes_posterior_rises_when_the_prior_rises():
    """При тех же самых likelihood редкая гипотеза всегда проигрывает частой."""
    rare = bayes_posterior(0.001, 0.99, 0.01)
    common = bayes_posterior(0.5, 0.99, 0.01)
    assert rare < common


def test_bayes_posterior_stays_a_probability():
    assert 0.0 <= bayes_posterior(0.2, 0.7, 0.3) <= 1.0


# ----------------------------------------------------- sequential_posterior
def test_sequential_posterior_without_data_is_the_prior():
    assert sequential_posterior(0.0001, 0.99, 0.01, 0) == APPROX(0.0001)


def test_sequential_posterior_of_one_test_matches_plain_bayes():
    assert sequential_posterior(0.0001, 0.99, 0.01, 1) == APPROX(
        bayes_posterior(0.0001, 0.99, 0.01)
    )


def test_two_positive_tests_push_belief_near_one_half():
    """1% после первого теста превращается в 49.5% после второго."""
    assert sequential_posterior(0.0001, 0.99, 0.01, 2) == pytest.approx(0.495, abs=1e-3)


def test_sequential_posterior_grows_monotonically_with_evidence():
    values = [sequential_posterior(0.01, 0.9, 0.1, n) for n in range(5)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_sequential_posterior_never_leaves_the_unit_interval():
    assert 0.0 <= sequential_posterior(0.01, 0.95, 0.05, 20) <= 1.0


# ---------------------------------------------------------- mle_probability
def test_mle_probability_is_the_plain_frequency():
    assert mle_probability(7, 10) == APPROX(0.7)


def test_mle_probability_of_an_unseen_event_is_exactly_zero():
    """Та самая дыра MLE, ради которой придумали сглаживание."""
    assert mle_probability(0, 10) == APPROX(0.0)


def test_mle_probability_of_every_trial_succeeding_is_one():
    assert mle_probability(10, 10) == APPROX(1.0)


# ------------------------------------------------------ laplace_probability
def test_laplace_probability_never_returns_zero():
    assert laplace_probability(0, 10, 5) > 0


def test_laplace_probability_add_one_formula():
    assert laplace_probability(3, 6, 4) == APPROX(0.4)


def test_laplace_probability_with_alpha_zero_is_mle():
    assert laplace_probability(7, 10, 5, alpha=0) == APPROX(mle_probability(7, 10))


def test_laplace_probabilities_over_the_vocabulary_sum_to_one():
    """Знаменатель растёт на alpha * vocab_size именно ради этого свойства."""
    counts = [3, 0, 5, 2]
    total = sum(counts)
    vocab_size = len(counts)
    assert sum(
        laplace_probability(c, total, vocab_size) for c in counts
    ) == APPROX(1.0)


def test_large_alpha_pulls_probabilities_toward_uniform():
    smoothed = laplace_probability(9, 10, 4, alpha=1000)
    assert smoothed == pytest.approx(0.25, abs=1e-2)


# -------------------------------------------------------------- beta_update
def test_beta_update_adds_successes_to_a_and_failures_to_b():
    assert beta_update((1, 1), 7, 3) == (8, 4)


def test_beta_update_chains_into_the_next_day():
    assert beta_update(beta_update((1, 1), 7, 3), 5, 5) == (13, 9)


def test_beta_update_is_order_independent():
    """Последовательное и пакетное обновление дают один и тот же ответ."""
    stepwise = beta_update(beta_update((1, 1), 7, 3), 5, 5)
    batch = beta_update((1, 1), 12, 8)
    assert stepwise == batch


def test_beta_update_without_data_changes_nothing():
    assert beta_update((2, 5), 0, 0) == (2, 5)


# ---------------------------------------------------------------- beta_mean
def test_beta_mean_of_the_uniform_prior_is_a_half():
    assert beta_mean((1, 1)) == APPROX(0.5)


def test_beta_mean_after_seven_heads_out_of_ten():
    assert beta_mean(beta_update((1, 1), 7, 3)) == APPROX(8 / 12)


def test_balanced_new_data_pulls_the_mean_back_toward_a_half():
    day2 = beta_mean(beta_update((1, 1), 7, 3))
    day3 = beta_mean(beta_update((8, 4), 5, 5))
    assert 0.5 < day3 < day2


def test_beta_mean_converges_to_the_frequency_as_data_grows():
    """При тысяче наблюдений prior перестаёт влиять."""
    assert beta_mean(beta_update((1, 1), 700, 300)) == pytest.approx(0.7, abs=1e-3)


# ----------------------------------------------------------------- beta_map
def test_beta_map_of_the_uniform_prior_is_a_half_by_convention():
    """Ловушка: (a-1)/(a+b-2) здесь даёт 0/0, у плоской Beta моды нет."""
    assert beta_map((1, 1)) == APPROX(0.5)


def test_beta_map_with_a_beta_two_two_prior_shrinks_the_mle():
    """7 орлов из 10: MLE даёт 0.7, MAP с prior Beta(2,2) — 0.667."""
    posterior = beta_update((2, 2), 7, 3)
    assert beta_map(posterior) == pytest.approx(2 / 3, abs=1e-9)


def test_beta_map_under_a_uniform_prior_equals_mle():
    """Равномерный prior не добавляет информации — MAP схлопывается в MLE."""
    assert beta_map(beta_update((1, 1), 7, 3)) == APPROX(mle_probability(7, 10))


def test_beta_map_differs_from_beta_mean():
    params = (9, 5)
    assert beta_map(params) != pytest.approx(beta_mean(params), abs=1e-6)


def test_beta_map_uses_the_boundary_when_only_one_shape_parameter_exceeds_one():
    assert beta_map((1, 3)) == APPROX(0.0)
    assert beta_map((3, 1)) == APPROX(1.0)


def test_beta_map_rejects_a_u_shaped_distribution_with_two_modes():
    """Beta(a<1,b<1) максимальна и в 0, и в 1: одного MAP-числа нет."""
    with pytest.raises(ValueError, match="две моды"):
        beta_map((0.5, 0.5))


# ---------------------------------------------------- naive_bayes_predict
def test_naive_bayes_recognises_spam():
    assert naive_bayes_predict(TRAIN_DOCS, TRAIN_LABELS, "free money waiting for you") == "spam"


def test_naive_bayes_recognises_ham():
    assert naive_bayes_predict(TRAIN_DOCS, TRAIN_LABELS, "meeting rescheduled to friday") == "ham"


def test_naive_bayes_ignores_letter_case():
    assert naive_bayes_predict(TRAIN_DOCS, TRAIN_LABELS, "FREE PRIZE") == "spam"


def test_naive_bayes_survives_words_it_has_never_seen():
    """Без сглаживания незнакомое слово обнулило бы оба класса."""
    assert naive_bayes_predict(
        TRAIN_DOCS, TRAIN_LABELS, "zzz qqq wxyz"
    ) in {"spam", "ham"}


def test_naive_bayes_with_a_single_class_always_answers_that_class():
    assert naive_bayes_predict(["hello there"], ["only"], "anything at all") == "only"


def test_naive_bayes_works_in_log_space_on_a_long_document():
    """Произведение 400 вероятностей округляется до 0.0, и классы сравняются.

    В логах счёт остаётся конечным и spam выигрывает уверенно.
    """
    long_spam = " ".join(["free"] * 400)
    assert naive_bayes_predict(TRAIN_DOCS, TRAIN_LABELS, long_spam) == "spam"


def test_naive_bayes_does_not_depend_on_the_order_of_training_examples():
    shuffled = list(reversed(TRAIN_DOCS))
    shuffled_labels = list(reversed(TRAIN_LABELS))
    a = naive_bayes_predict(TRAIN_DOCS, TRAIN_LABELS, "free cash prize")
    b = naive_bayes_predict(shuffled, shuffled_labels, "free cash prize")
    assert a == b
