"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_M, _K, _N = 32, 64, 64
_A = [[random.gauss(0.0, 1.0) for _ in range(_K)] for _ in range(_M)]
_B = [[random.gauss(0.0, 1.0) for _ in range(_N)] for _ in range(_K)]

_DATA = [[random.gauss(0.0, 1.0) for _ in range(16)] for _ in range(2048)]


def _mean_gradient(batch):
    width = len(batch[0])
    return [sum(sample[i] for sample in batch) / len(batch) for i in range(width)]


BENCH = {
    "matmul": (_A, _B),
    "tensor_parallel_matmul": (_A, _B, 8),
    "shard_batch": (_DATA, 8),
    "data_parallel_gradient": (_DATA, 8, _mean_gradient),
    "pipeline_bubble_fraction": (16, 64),
    "memory_budget": (405, 2, "adam", 128, "fsdp"),
    "mixed_precision_savings": (405,),
    "min_gpus_for_fsdp": (405, 80, 1024),
}
