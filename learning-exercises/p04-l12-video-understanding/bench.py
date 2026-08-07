"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_features = [[_rng.gauss(0.0, 1.0) for _ in range(512)] for _ in range(64)]
_signal = [_rng.gauss(0.0, 1.0) for _ in range(20000)]
_scores = [[_rng.random() for _ in range(400)] for _ in range(2000)]
_labels = [_rng.randrange(400) for _ in range(2000)]
_kernel2d = [[_rng.gauss(0.0, 1.0) for _ in range(7)] for _ in range(7)]

BENCH = {
    "sample_uniform": (3000, 16),
    "sample_dense": (3000, 16, random.Random(0)),
    "multi_clip_indices": (3000, 16, 10),
    "temporal_mean_pool": (_features,),
    "temporal_conv": (_signal, [-1.0, 0.0, 1.0]),
    "top_k_accuracy": (_scores, _labels, 5),
    "inflate_2d_to_3d": (_kernel2d, 3),
    "conv2plus1d_mid_channels": (256, 256, 3),
}
