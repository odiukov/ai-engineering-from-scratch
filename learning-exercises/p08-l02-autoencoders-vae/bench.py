"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_IN, _HID, _Z = 32, 48, 16


def _mat(rows, cols):
    return [[_rng.gauss(0, 0.2) for _ in range(cols)] for _ in range(rows)]


_enc = {
    "W1": _mat(_HID, _IN), "b1": [0.0] * _HID,
    "W_mu": _mat(_Z, _HID), "b_mu": [0.0] * _Z,
    "W_sig": _mat(_Z, _HID), "b_sig": [0.0] * _Z,
}
_dec = {
    "W1": _mat(_HID, _Z), "b1": [0.0] * _HID,
    "W_out": _mat(_IN, _HID), "b_out": [0.0] * _IN,
}

_x = [_rng.gauss(0, 1) for _ in range(_IN)]
_mu = [_rng.gauss(0, 1) for _ in range(_Z)]
_lv = [_rng.gauss(0, 0.5) for _ in range(_Z)]
_eps = [_rng.gauss(0, 1) for _ in range(_Z)]
_z = [_rng.gauss(0, 1) for _ in range(_Z)]
_x_hat = [_rng.gauss(0, 1) for _ in range(_IN)]

BENCH = {
    "dense": (_mat(_HID, _IN), _x, [0.0] * _HID),
    "encode": (_x, _enc),
    "reparameterize": (_mu, _lv, _eps),
    "decode": (_z, _dec),
    "kl_divergence_gaussian": (_mu, _lv),
    "kl_grad": (_mu, _lv),
    "elbo_loss": (_x, _x_hat, _mu, _lv, 1.0),
    "sample_from_prior": (_dec, _Z, random.Random(0)),
}
