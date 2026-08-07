"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# Смесь трафика: короткий чат и длинный RAG-префикс вперемешку. Знак
# итогового выигрыша зависит от пропорции, поэтому она и задана явно.
_fleet = (
    [(_rng.randrange(80, 400), _rng.randrange(20, 200)) for _ in range(15000)]
    + [(_rng.randrange(2000, 16000), _rng.randrange(100, 800)) for _ in range(15000)]
)
_rng.shuffle(_fleet)

BENCH = {
    "kv_bytes": (4000,),
    "transfer_ms": (4000, 10.0),
    "phase_ms": (4000, 300),
    "colocated_ms": (4000, 300),
    "disaggregated_ms": (4000, 300, 10.0),
    "disagg_gain_ms": (4000, 300, 10.0),
    # двоичный поиск против линейного перебора до миллиона токенов
    "crossover_prompt_tokens": (10.0,),
    "fleet_report": (_fleet, 10.0),
}
