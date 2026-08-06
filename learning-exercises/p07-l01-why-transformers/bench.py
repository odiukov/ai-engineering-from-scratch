"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_xs = [random.uniform(-1.0, 1.0) for _ in range(4096)]

BENCH = {
    "rnn_state": (_xs, 0.9),
    "attention_mean": (_xs,),
    "serial_scan": (_xs,),
    "hillis_steele_scan": (_xs,),
    "scan_rounds": (4096,),
    "attention_memory_cells": (4096, 32, 32),
    "pick_architecture": (2048,),
}
