"""Тесты к уроку «Масштабирование: распределённое обучение, FSDP, ZeRO». Правь exercise.py."""

import random

import pytest

from exercise import (
    data_parallel_gradient,
    matmul,
    memory_budget,
    min_gpus_for_fsdp,
    mixed_precision_savings,
    pipeline_bubble_fraction,
    shard_batch,
    tensor_parallel_matmul,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [x for row in M for x in row]


def mean_gradient(batch):
    """Градиент-игрушка: покомпонентное среднее батча."""
    width = len(batch[0])
    return [sum(sample[i] for sample in batch) / len(batch) for i in range(width)]


# -------------------------------------------------------------------- matmul
def test_matmul_multiplies_row_by_column():
    assert flat(matmul([[1, 2]], [[3], [4]])) == APPROX([11])
    assert matmul([], [[1]]) == []


def test_matmul_by_identity_returns_the_input():
    A = [[1.0, 2.0], [3.0, 4.0]]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    assert flat(matmul(A, identity)) == APPROX(flat(A))


def test_matmul_result_has_the_outer_dimensions():
    out = matmul([[1, 1, 1], [2, 2, 2]], [[1, 2, 3, 4]] * 3)
    assert len(out) == 2
    assert len(out[0]) == 4


def test_matmul_is_not_commutative():
    """Порядок множителей меняет ответ — самая частая ошибка в реализации."""
    A = [[1, 2], [3, 4]]
    B = [[0, 1], [0, 0]]
    assert flat(matmul(A, B)) != flat(matmul(B, A))


# ------------------------------------------------------ tensor_parallel_matmul
def test_tensor_parallel_matmul_is_bit_exact():
    """Тензорный параллелизм — не приближение: ответ обязан совпасть с обычным."""
    random.seed(0)
    x = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
    W = [[random.gauss(0, 1) for _ in range(8)] for _ in range(8)]
    for num_gpus in (1, 2, 4, 8):
        assert flat(tensor_parallel_matmul(x, W, num_gpus)) == pytest.approx(flat(matmul(x, W)))


def test_tensor_parallel_matmul_keeps_the_output_shape():
    x = [[1.0, 2.0]]
    W = [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]
    out = tensor_parallel_matmul(x, W, 4)
    assert out == [[1.0, 2.0, 2.0, 4.0]]


def test_tensor_parallel_matmul_on_one_gpu_is_plain_matmul():
    x = [[1.0, 2.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    assert tensor_parallel_matmul(x, W, 1) == matmul(x, W)


def test_tensor_parallel_matmul_refuses_an_uneven_split():
    """Неравные срезы — одна карта тормозит всю группу, лучше упасть сразу."""
    with pytest.raises(ValueError):
        tensor_parallel_matmul([[1.0, 2.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], 2)


# --------------------------------------------------------------- shard_batch
def test_shard_batch_splits_evenly():
    assert shard_batch([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_shard_batch_hands_the_remainder_to_the_first_gpus():
    assert shard_batch([1, 2, 3], 2) == [[1, 2], [3]]
    assert shard_batch([1], 3) == [[1], [], []]


def test_shard_batch_loses_no_samples():
    data = list(range(37))
    for num_gpus in (1, 2, 3, 5, 8):
        shards = shard_batch(data, num_gpus)
        assert [x for shard in shards for x in shard] == data


def test_shard_sizes_differ_by_at_most_one():
    """All-reduce ждёт самую нагруженную карту — перекос стоит времени всем."""
    sizes = [len(s) for s in shard_batch(list(range(37)), 8)]
    assert max(sizes) - min(sizes) <= 1


# ------------------------------------------------------ data_parallel_gradient
def test_data_parallel_gradient_on_one_gpu():
    assert data_parallel_gradient([[1.0], [2.0], [3.0], [4.0]], 1, mean_gradient) == APPROX([2.5])


def test_data_parallel_gradient_matches_the_full_batch_gradient():
    """Смысл data parallelism: быстрее, но ровно тот же шаг."""
    data = [[float(i), float(i * i)] for i in range(12)]
    reference = mean_gradient(data)
    for num_gpus in (1, 2, 3, 4, 5, 8):
        assert data_parallel_gradient(data, num_gpus, mean_gradient) == pytest.approx(reference)


def test_unweighted_averaging_of_uneven_shards_is_wrong():
    """Куски 3+2+2: простое среднее средних перекошено, взвешенное — нет."""
    data = [[float(i)] for i in range(7)]
    shards = shard_batch(data, 3)
    naive = sum(mean_gradient(s)[0] for s in shards) / 3
    assert data_parallel_gradient(data, 3, mean_gradient)[0] == APPROX(3.0)
    assert naive != pytest.approx(3.0, abs=1e-6)


def test_larger_batch_gives_a_less_noisy_gradient():
    """Больший батч — та же оценка, но ближе к истинному среднему."""
    rng = random.Random(0)

    def average_error(batch_size, trials=40):
        total = 0.0
        for _ in range(trials):
            batch = [[rng.gauss(0.0, 1.0)] for _ in range(batch_size)]
            total += abs(data_parallel_gradient(batch, 2, mean_gradient)[0])
        return total / trials

    assert average_error(128) < average_error(4)


# --------------------------------------------------- pipeline_bubble_fraction
def test_one_microbatch_leaves_all_but_one_stage_idle():
    assert pipeline_bubble_fraction(4, 1) == APPROX(0.75)


def test_a_single_stage_has_no_pipeline_at_all():
    assert pipeline_bubble_fraction(1, 1) == APPROX(0.0)
    assert pipeline_bubble_fraction(1, 32) == APPROX(0.0)


def test_more_microbatches_shrink_the_bubble():
    fractions = [pipeline_bubble_fraction(4, m) for m in (1, 4, 8, 16, 32)]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[-1] < 0.1


def test_deeper_pipeline_means_a_bigger_bubble():
    assert pipeline_bubble_fraction(16, 8) > pipeline_bubble_fraction(4, 8)


# ------------------------------------------------------------- memory_budget
def test_seven_billion_parameters_need_84_gigabytes():
    """14 весов + 56 Adam + 14 градиентов, и это ещё без активаций."""
    budget = memory_budget(7)
    assert budget["weights_gb"] == APPROX(14.0)
    assert budget["optimizer_gb"] == APPROX(56.0)
    assert budget["per_gpu_gb"] == APPROX(84.0)


def test_seventy_billion_parameters_do_not_fit_on_one_card():
    budget = memory_budget(70)
    assert budget["per_gpu_gb"] == APPROX(840.0)
    assert budget["fits_on_80gb"] is False


def test_fsdp_on_eight_gpus_still_overflows_a_70b_model():
    """105 ГБ на карту — цифра из урока: восьми карт мало, нужно шестнадцать."""
    assert memory_budget(70, num_gpus=8, sharding="zero3")["per_gpu_gb"] == APPROX(105.0)
    assert memory_budget(70, num_gpus=16, sharding="zero3")["fits_on_80gb"] is True


def test_zero_stages_shard_more_and_more():
    stages = [
        memory_budget(70, num_gpus=8, sharding=s)["per_gpu_gb"]
        for s in ("none", "zero1", "zero2", "zero3")
    ]
    assert stages == sorted(stages, reverse=True)


def test_fsdp_is_the_same_thing_as_zero3():
    """Разные названия, одна механика: PyTorch повторил идею DeepSpeed."""
    fsdp = memory_budget(70, num_gpus=8, sharding="fsdp")
    zero3 = memory_budget(70, num_gpus=8, sharding="zero3")
    assert fsdp == zero3


def test_sharding_across_one_gpu_saves_nothing():
    assert memory_budget(7, sharding="zero3", num_gpus=1) == memory_budget(7, sharding="none")


def test_sgd_needs_half_the_optimizer_memory_of_adam():
    adam = memory_budget(7, optimizer="adam")["optimizer_gb"]
    sgd = memory_budget(7, optimizer="sgd")["optimizer_gb"]
    assert sgd == APPROX(adam / 2)


# ---------------------------------------------------- mixed_precision_savings
def test_mixed_precision_saves_a_quarter_not_a_half():
    result = mixed_precision_savings(7)
    assert result["fp32_gb"] == APPROX(112.0)
    assert result["mixed_gb"] == APPROX(84.0)
    assert result["savings"] == APPROX(0.25)


def test_the_saving_does_not_depend_on_model_size():
    """Доля, а не абсолютное число: у 7B и у 405B экономия одинаковая."""
    savings = [mixed_precision_savings(b)["savings"] for b in (0.124, 7, 70, 405)]
    assert savings == pytest.approx([0.25] * 4)


def test_optimizer_eats_more_than_half_of_the_mixed_precision_budget():
    result = mixed_precision_savings(7)
    assert memory_budget(7)["optimizer_gb"] / result["mixed_gb"] > 0.5


# ------------------------------------------------------------ min_gpus_for_fsdp
def test_min_gpus_matches_the_numbers_from_the_lesson():
    assert min_gpus_for_fsdp(7) == 2
    assert min_gpus_for_fsdp(70) == 11
    assert min_gpus_for_fsdp(405) == 61


def test_large_training_cluster_size_is_not_a_memory_capacity_minimum():
    """Thousands of GPUs buy throughput; the simplified capacity floor is 61."""
    assert min_gpus_for_fsdp(405) < 16384


def test_min_gpus_returns_none_when_the_cluster_is_too_small():
    assert min_gpus_for_fsdp(405, max_gpus=8) is None


def test_bigger_cards_need_fewer_of_them():
    assert min_gpus_for_fsdp(70, gpu_memory_gb=192) < min_gpus_for_fsdp(70, gpu_memory_gb=80)


def test_a_tiny_model_fits_on_a_single_gpu():
    assert min_gpus_for_fsdp(0.124) == 1
