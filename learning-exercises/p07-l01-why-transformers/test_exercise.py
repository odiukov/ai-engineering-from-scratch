"""Тесты к уроку «Почему трансформеры». Правь exercise.py."""

import random

import pytest

from exercise import (
    attention_memory_cells,
    attention_mean,
    hillis_steele_scan,
    pick_architecture,
    rnn_state,
    scan_rounds,
    serial_scan,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def noise(n, seed=0):
    """Воспроизводимый шум: свой Random, глобальный random не трогаем."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


# --------------------------------------------------------------- rnn_state
def test_rnn_state_of_empty_sequence_is_zero():
    assert rnn_state([], 0.5) == APPROX(0.0)


def test_rnn_state_attenuates_the_distant_past():
    """Единица в начале доходит до конца ослабленной в decay^(n-1) раз."""
    assert rnn_state([1.0, 0.0, 0.0], decay=0.5) == APPROX(0.25)


def test_rnn_state_keeps_the_most_recent_input_intact():
    assert rnn_state([0.0, 0.0, 1.0], decay=0.5) == APPROX(1.0)


def test_rnn_state_depends_on_order():
    """Рекуррентность несёт позицию бесплатно — тем же и платит за неё."""
    assert rnn_state([1.0, 0.0], 0.5) != APPROX(rnn_state([0.0, 1.0], 0.5))


def test_rnn_state_forgets_exponentially_on_long_input():
    """Пятьдесят шагов с decay=0.9 съедают сигнал более чем в 100 раз."""
    xs = [1.0] + [0.0] * 50
    assert rnn_state(xs, 0.9) < 0.01


# ---------------------------------------------------------- attention_mean
def test_attention_mean_averages():
    assert attention_mean([1.0, 0.0, 0.0]) == APPROX(1 / 3)


def test_attention_mean_is_permutation_invariant():
    """Вот она, слепота внимания к порядку — урок 04 будет её лечить."""
    xs = noise(64, seed=1)
    shuffled = list(xs)
    random.Random(2).shuffle(shuffled)
    assert attention_mean(shuffled) == pytest.approx(attention_mean(xs), abs=1e-12)


def test_attention_mean_rejects_an_empty_sequence():
    with pytest.raises(ValueError):
        attention_mean([])


# ------------------------------------------------------------- serial_scan
def test_serial_scan_accumulates():
    assert serial_scan([1.0, 2.0, 3.0]) == APPROX([1.0, 3.0, 6.0])


def test_serial_scan_of_empty_is_empty():
    assert serial_scan([]) == []


def test_serial_scan_last_element_is_the_total_sum():
    xs = noise(200, seed=3)
    assert serial_scan(xs)[-1] == pytest.approx(sum(xs), abs=1e-9)


# ------------------------------------------------------ hillis_steele_scan
def test_hillis_steele_scan_matches_the_serial_one():
    """Тот же ответ другим графом зависимостей — в этом весь смысл."""
    xs = noise(64, seed=4)
    assert hillis_steele_scan(xs) == pytest.approx(serial_scan(xs), abs=1e-9)


def test_hillis_steele_scan_works_on_a_length_that_is_not_a_power_of_two():
    xs = [1.0] * 13
    assert hillis_steele_scan(xs) == APPROX([float(i + 1) for i in range(13)])


def test_hillis_steele_scan_reads_the_previous_round_not_the_current_one():
    """Правка на месте даёт [1, 2, 4, 8, ...] вместо [1, 2, 3, 4, ...]."""
    assert hillis_steele_scan([1.0, 1.0, 1.0, 1.0]) == APPROX([1.0, 2.0, 3.0, 4.0])


def test_hillis_steele_scan_does_not_mutate_the_input():
    xs = [1.0, 2.0, 3.0, 4.0]
    hillis_steele_scan(xs)
    assert xs == [1.0, 2.0, 3.0, 4.0]


def test_hillis_steele_scan_of_a_single_element_is_that_element():
    assert hillis_steele_scan([7.0]) == APPROX([7.0])


# ------------------------------------------------------------- scan_rounds
def test_scan_rounds_of_a_power_of_two():
    assert scan_rounds(1024) == 10


def test_scan_rounds_rounds_up():
    assert scan_rounds(1000) == 10


def test_scan_rounds_of_a_single_element_is_zero():
    assert scan_rounds(1) == 0


def test_scan_rounds_is_logarithmic_not_linear():
    """Глубина параллельного скана растёт на 1, когда длина удваивается."""
    assert scan_rounds(2 ** 20) - scan_rounds(2 ** 19) == 1
    assert scan_rounds(2 ** 20) < 25


# ------------------------------------------------- attention_memory_cells
def test_attention_memory_cells_counts_the_square():
    assert attention_memory_cells(4) == 16


def test_attention_memory_cells_scales_with_heads_and_layers():
    assert attention_memory_cells(8, n_heads=8, n_layers=2) == 1024


def test_doubling_context_quadruples_a_materialized_attention_matrix():
    """Обычная реализация хранит матрицу; FlashAttention считает те же пары блоками."""
    assert attention_memory_cells(2048) == 4 * attention_memory_cells(1024)


# -------------------------------------------------------- pick_architecture
def test_default_case_picks_a_transformer():
    assert pick_architecture(2048) == "transformer"


def test_streaming_inference_picks_a_recurrent_model():
    assert pick_architecture(2048, streaming=True) == "rnn"


def test_million_token_context_picks_linear_attention():
    assert pick_architecture(5_000_000) == "linear-attention"


def test_edge_device_without_a_matmul_unit_picks_a_recurrent_model():
    assert pick_architecture(512, has_matmul_accelerator=False) == "rnn"


def test_streaming_beats_length_in_the_rule_order():
    """Стриминг проверяется первым: длина 5 млн его не перебивает."""
    assert pick_architecture(5_000_000, streaming=True) == "rnn"


def test_one_million_exactly_is_still_a_transformer():
    """Граница строгая: «больше миллиона», а не «миллион и больше»."""
    assert pick_architecture(1_000_000) == "transformer"
