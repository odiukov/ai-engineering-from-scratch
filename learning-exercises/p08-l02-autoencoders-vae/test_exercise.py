"""Тесты к уроку «Автоэнкодеры и вариационные автоэнкодеры (VAE)». Правь exercise.py."""

import math
import random
import statistics

import pytest

from exercise import (
    decode,
    dense,
    elbo_loss,
    encode,
    kl_divergence_gaussian,
    kl_grad,
    reparameterize,
    sample_from_prior,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def enc_params():
    """Энкодер 2 -> 3 -> (2, 2). Веса подобраны руками, никакого random."""
    return {
        "W1": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        "b1": [0.0, 0.0, 0.0],
        "W_mu": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "b_mu": [0.0, 0.0],
        "W_sig": [[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        "b_sig": [0.0, 0.0],
    }


def dec_params():
    """Декодер 2 -> 3 -> 2."""
    return {
        "W1": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "b1": [0.0, 0.0, 0.0],
        "W_out": [[2.0, 0.0, 1.0], [0.0, 2.0, -1.0]],
        "b_out": [0.5, -0.5],
    }


# ---------------------------------------------------------------- dense
def test_dense_multiplies_rows_and_adds_bias():
    assert dense([[1.0, 0.0], [0.0, 2.0]], [3.0, 4.0], [0.0, 1.0]) == APPROX([3.0, 9.0])


def test_dense_output_length_is_the_row_count():
    assert len(dense([[1.0, 1.0]] * 5, [1.0, 1.0], [0.0] * 5)) == 5


def test_dense_rejects_a_bias_of_the_wrong_length():
    with pytest.raises(ValueError):
        dense([[1.0], [1.0]], [1.0], [0.0])


def test_dense_rejects_a_row_that_does_not_match_x():
    """Ловушка: zip обрезал бы длинную строку и вернул правдоподобную ерунду."""
    with pytest.raises(ValueError):
        dense([[1.0, 2.0, 3.0]], [1.0, 2.0], [0.0])


# --------------------------------------------------------------- encode
def test_encode_worked_example():
    enc = {
        "W1": [[0.0]], "b1": [0.0],
        "W_mu": [[1.0]], "b_mu": [2.0],
        "W_sig": [[1.0]], "b_sig": [-1.0],
    }
    mu, log_sigma2 = encode([1.0], enc)
    assert mu == APPROX([2.0])
    assert log_sigma2 == APPROX([-1.0])


def test_encode_returns_two_vectors_of_latent_size():
    mu, log_sigma2 = encode([1.0, -1.0], enc_params())
    assert len(mu) == 2 and len(log_sigma2) == 2


def test_encode_is_deterministic():
    """Случайность в VAE живёт только в reparameterize, энкодер её не знает."""
    assert encode([0.3, 0.7], enc_params()) == encode([0.3, 0.7], enc_params())


def test_encode_hidden_layer_saturates_through_tanh():
    """tanh зажимает скрытый слой, поэтому огромный вход не даёт огромного mu."""
    mu, _ = encode([1000.0, 1000.0], enc_params())
    assert all(abs(m) < 1.001 for m in mu)


def test_encode_outputs_log_variance_which_may_be_negative():
    """log_sigma2 не ограничен снизу — это и есть причина логарифмировать."""
    _, log_sigma2 = encode([-1.0, 1.0], enc_params())
    assert min(log_sigma2) < 0.0


# -------------------------------------------------------- reparameterize
def test_reparameterize_with_zero_noise_returns_mu():
    assert reparameterize([5.0, -3.0], [2.0, 7.0], [0.0, 0.0]) == APPROX([5.0, -3.0])


def test_reparameterize_worked_example():
    assert reparameterize([0.0, 1.0], [0.0, 0.0], [1.0, -1.0]) == APPROX([1.0, 0.0])


def test_reparameterize_uses_sigma_not_variance():
    """log_sigma2 = 2 значит sigma = e, а не e^2. Легко перепутать половинку."""
    got = reparameterize([0.0], [2.0], [1.0])
    assert got == pytest.approx([math.e], abs=1e-9)


def test_reparameterize_sample_mean_matches_mu():
    """Выборочная статистика та же, что у прямого сэмплирования из q(z|x)."""
    rng = random.Random(0)
    draws = [reparameterize([3.0], [0.0], [rng.gauss(0, 1)])[0] for _ in range(20000)]
    assert statistics.fmean(draws) == pytest.approx(3.0, abs=0.05)


def test_reparameterize_sample_std_matches_sigma():
    rng = random.Random(1)
    sigma = math.exp(0.5 * 1.5)
    draws = [reparameterize([0.0], [1.5], [rng.gauss(0, 1)])[0] for _ in range(20000)]
    assert statistics.stdev(draws) == pytest.approx(sigma, rel=0.05)


def test_reparameterize_passes_gradient_to_mu():
    """dz/dmu = 1: шум стал входом, поэтому производная вообще существует."""
    h = 1e-6
    up = reparameterize([1.0 + h], [0.4], [0.7])[0]
    down = reparameterize([1.0 - h], [0.4], [0.7])[0]
    assert (up - down) / (2 * h) == pytest.approx(1.0, abs=1e-6)


def test_reparameterize_passes_gradient_to_log_sigma2():
    """dz/dlv = 0.5 * sigma * eps — сверяем с центральной разностью."""
    h = 1e-6
    lv, eps = 0.4, 0.7
    up = reparameterize([1.0], [lv + h], [eps])[0]
    down = reparameterize([1.0], [lv - h], [eps])[0]
    expected = 0.5 * math.exp(0.5 * lv) * eps
    assert (up - down) / (2 * h) == pytest.approx(expected, abs=1e-6)


def test_reparameterize_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        reparameterize([0.0, 0.0], [0.0], [0.0])


# --------------------------------------------------------------- decode
def test_decode_worked_example():
    dec = {"W1": [[0.0]], "b1": [0.0], "W_out": [[3.0]], "b_out": [7.0]}
    assert decode([1.0], dec) == APPROX([7.0])


def test_decode_output_has_the_data_dimension():
    assert len(decode([0.5, -0.5], dec_params())) == 2


def test_decode_has_no_output_activation():
    """Выход может выйти за (-1, 1) — иначе MSE-реконструкция была бы обречена."""
    out = decode([10.0, 10.0], dec_params())
    assert max(abs(v) for v in out) > 1.0


def test_decode_of_the_same_z_is_the_same_x_hat():
    assert decode([0.2, 0.9], dec_params()) == decode([0.2, 0.9], dec_params())


# ----------------------------------------------- kl_divergence_gaussian
def test_kl_is_zero_exactly_at_the_prior():
    """q == p ровно при mu = 0 и log_sigma2 = 0 — тогда и только тогда KL = 0."""
    assert kl_divergence_gaussian([0.0, 0.0], [0.0, 0.0]) == APPROX(0.0)


def test_kl_is_never_negative():
    for mu in (-3.0, -0.5, 0.0, 0.5, 3.0):
        for lv in (-4.0, -1.0, 0.0, 1.0, 4.0):
            assert kl_divergence_gaussian([mu], [lv]) >= -1e-12


def test_kl_is_positive_whenever_q_differs_from_the_prior():
    assert kl_divergence_gaussian([0.5], [0.0]) > 0.0
    assert kl_divergence_gaussian([0.0], [0.5]) > 0.0
    assert kl_divergence_gaussian([0.0], [-0.5]) > 0.0


def test_kl_grows_as_mu_moves_away_from_zero():
    assert kl_divergence_gaussian([2.0], [0.0]) > kl_divergence_gaussian([1.0], [0.0])


def test_kl_penalizes_too_wide_and_too_narrow_alike():
    """Штраф идёт в обе стороны: и sigma > 1, и sigma < 1 стоят денег."""
    assert kl_divergence_gaussian([0.0], [1.0]) > 0.0
    assert kl_divergence_gaussian([0.0], [-1.0]) > 0.0


def test_kl_adds_up_over_independent_dimensions():
    one = kl_divergence_gaussian([1.0], [0.0])
    three = kl_divergence_gaussian([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert three == APPROX(3 * one)


def test_kl_worked_example():
    assert kl_divergence_gaussian([1.0], [0.0]) == APPROX(0.5)


# ------------------------------------------------------------- kl_grad
def test_kl_grad_of_mu_matches_numeric_gradient():
    h = 1e-6
    mu, lv = [1.7], [0.4]
    up = kl_divergence_gaussian([mu[0] + h], lv)
    down = kl_divergence_gaussian([mu[0] - h], lv)
    d_mu, _ = kl_grad(mu, lv)
    assert d_mu[0] == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_kl_grad_of_log_sigma2_matches_numeric_gradient():
    h = 1e-6
    mu, lv = [1.7], [0.4]
    up = kl_divergence_gaussian(mu, [lv[0] + h])
    down = kl_divergence_gaussian(mu, [lv[0] - h])
    _, d_lv = kl_grad(mu, lv)
    assert d_lv[0] == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_kl_grad_is_zero_at_the_prior():
    """Ноль градиента в приоре — это минимум KL, дальше тянуть некуда."""
    d_mu, d_lv = kl_grad([0.0, 0.0], [0.0, 0.0])
    assert d_mu == APPROX([0.0, 0.0])
    assert d_lv == APPROX([0.0, 0.0])


def test_kl_grad_pushes_mu_toward_zero():
    """Градиент по mu равен самому mu: шаг против него всегда идёт к нулю."""
    d_mu, _ = kl_grad([2.5], [0.0])
    assert d_mu[0] > 0
    assert 2.5 - 0.1 * d_mu[0] < 2.5


# ------------------------------------------------------------ elbo_loss
def test_elbo_is_zero_on_perfect_reconstruction_at_the_prior():
    assert elbo_loss([1.0], [1.0], [0.0], [0.0]) == pytest.approx((0.0, 0.0, 0.0))


def test_elbo_worked_example():
    assert elbo_loss([1.0], [0.0], [1.0], [0.0], beta=2.0) == pytest.approx(
        (2.0, 1.0, 0.5)
    )


def test_beta_zero_turns_the_vae_into_a_plain_autoencoder():
    """При beta = 0 KL считается, но на total не влияет — латент разъезжается."""
    total, recon, kl = elbo_loss([1.0, 2.0], [0.0, 0.0], [3.0], [1.0], beta=0.0)
    assert kl > 0
    assert total == APPROX(recon)


def test_larger_beta_weights_the_kl_more():
    small = elbo_loss([1.0], [1.0], [2.0], [0.0], beta=0.1)[0]
    large = elbo_loss([1.0], [1.0], [2.0], [0.0], beta=5.0)[0]
    assert large > small


def test_elbo_returns_the_two_terms_separately():
    """Обе половинки нужны в логах порознь: только так видно posterior collapse."""
    total, recon, kl = elbo_loss([1.0, 1.0], [0.0, 0.0], [1.0], [0.0], beta=1.0)
    assert recon == APPROX(2.0)
    assert kl == APPROX(0.5)
    assert total == APPROX(2.5)


def test_negative_beta_raises_value_error():
    with pytest.raises(ValueError):
        elbo_loss([1.0], [1.0], [0.0], [0.0], beta=-1.0)


# ----------------------------------------------------- sample_from_prior
def test_sample_from_prior_needs_no_encoder():
    """Сэмплирование идёт через декодер: нулевые веса выхода дают чистый bias."""
    dec = {
        "W1": [[1.0, 0.0], [0.0, 1.0]],
        "b1": [0.0, 0.0],
        "W_out": [[0.0, 0.0], [0.0, 0.0]],
        "b_out": [7.0, -3.0],
    }
    assert sample_from_prior(dec, 2, random.Random(0)) == APPROX([7.0, -3.0])


def test_sample_from_prior_is_reproducible_from_the_same_seed():
    a = sample_from_prior(dec_params(), 2, random.Random(5))
    b = sample_from_prior(dec_params(), 2, random.Random(5))
    assert a == b


def test_sample_from_prior_varies_across_seeds():
    a = sample_from_prior(dec_params(), 2, random.Random(1))
    b = sample_from_prior(dec_params(), 2, random.Random(2))
    assert a != b


def test_sample_from_prior_has_the_data_dimension():
    assert len(sample_from_prior(dec_params(), 2, random.Random(0))) == 2


def test_zero_latent_dimension_raises_value_error():
    with pytest.raises(ValueError):
        sample_from_prior(dec_params(), 0, random.Random(0))


# --------------------------------------------------- всё вместе, end-to-end
def test_full_pass_encode_reparameterize_decode_keeps_the_shape():
    x = [0.4, -0.9]
    mu, log_sigma2 = encode(x, enc_params())
    z = reparameterize(mu, log_sigma2, [0.0] * len(mu))
    x_hat = decode(z, dec_params())
    assert len(x_hat) == len(x)
    total, recon, kl = elbo_loss(x, x_hat, mu, log_sigma2, beta=1.0)
    assert total == APPROX(recon + kl)


def test_collapsed_encoder_gives_zero_kl_and_a_z_independent_of_x():
    """Posterior collapse на пальцах: mu = 0, lv = 0 для любого x — латент пуст."""
    enc = {
        "W1": [[0.0, 0.0]], "b1": [0.0],
        "W_mu": [[0.0]], "b_mu": [0.0],
        "W_sig": [[0.0]], "b_sig": [0.0],
    }
    mu_a, lv_a = encode([1.0, 2.0], enc)
    mu_b, lv_b = encode([-9.0, 4.0], enc)
    assert (mu_a, lv_a) == (mu_b, lv_b)
    assert kl_divergence_gaussian(mu_a, lv_a) == APPROX(0.0)
