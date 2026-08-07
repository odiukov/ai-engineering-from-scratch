"""Тесты к уроку «Jamba: гибрид SSM и трансформера». Правь exercise.py."""

import pytest

from exercise import (
    count_layer_types,
    inference_memory,
    kv_cache_advantage,
    kv_cache_bytes,
    layer_plan,
    ssm_scan,
    ssm_state_bytes,
    ssm_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CONTEXT_256K = 262144


# ----------------------------------------------------------------- ssm_step
def test_ssm_step_mixes_state_and_input():
    assert ssm_step(2.0, 1.0, 0.9, 0.5) == APPROX(2.3)


def test_ssm_step_with_zero_decay_forgets_the_state():
    assert ssm_step(999.0, 1.0, 0.0, 0.5) == APPROX(0.5)


def test_ssm_step_with_unit_decay_accumulates():
    assert ssm_step(2.0, 3.0, 1.0, 1.0) == APPROX(5.0)


# ----------------------------------------------------------------- ssm_scan
def test_impulse_response_decays_geometrically():
    """Единицу подали один раз — смотрим, сколько её помнят."""
    assert ssm_scan([1.0, 0.0, 0.0, 0.0], 0.5, 1.0, 1.0) == APPROX([1.0, 0.5, 0.25, 0.125])


def test_unit_decay_turns_the_scan_into_a_running_sum():
    assert ssm_scan([1.0, 2.0, 3.0], 1.0, 1.0, 1.0) == APPROX([1.0, 3.0, 6.0])


def test_zero_decay_leaves_no_memory_at_all():
    """a = 0 — слой перестаёт быть последовательным, история не влияет."""
    assert ssm_scan([1.0, 2.0, 3.0], 0.0, 1.0, 2.0) == APPROX([2.0, 4.0, 6.0])


def test_scan_can_be_resumed_from_a_saved_state():
    """Потоковый инференс: состояния хватает, прошлые токены не нужны."""
    xs = [0.7, -1.2, 0.3, 2.0, -0.5]
    a, b, c = 0.8, 0.6, 1.5
    whole = ssm_scan(xs, a, b, c)
    h = 0.0
    for x in xs[:2]:
        h = ssm_step(h, x, a, b)
    assert ssm_scan(xs[2:], a, b, c, h0=h) == APPROX(whole[2:])


def test_scan_length_matches_the_input_length():
    assert len(ssm_scan([0.1] * 17, 0.9, 1.0, 1.0)) == 17


# ---------------------------------------------------------------- layer_plan
def test_every_eighth_layer_is_attention():
    plan = layer_plan(32)
    assert [i for i, (kind, _) in enumerate(plan) if kind == "attention"] == [7, 15, 23, 31]


def test_moe_lands_on_every_other_layer():
    assert [i for i, (_, moe) in enumerate(layer_plan(8)) if moe] == [1, 3, 5, 7]


def test_ratio_one_is_a_pure_transformer():
    assert all(kind == "attention" for kind, _ in layer_plan(4, 1))


def test_ratio_zero_is_a_pure_ssm():
    assert all(kind == "mamba" for kind, _ in layer_plan(4, 0))


def test_layer_plan_rejects_a_negative_ratio():
    with pytest.raises(ValueError):
        layer_plan(8, -1)


# --------------------------------------------------------- count_layer_types
def test_jamba_shape_is_four_attention_and_twenty_eight_mamba():
    assert count_layer_types(layer_plan(32)) == {"attention": 4, "mamba": 28, "moe": 16}


def test_attention_and_mamba_add_up_to_the_plan_length():
    counts = count_layer_types(layer_plan(30, 8, 3))
    assert counts["attention"] + counts["mamba"] == 30


def test_moe_is_counted_apart_from_the_layer_kind():
    """MoE — это про MLP, он навешивается и на attention-слой тоже."""
    counts = count_layer_types(layer_plan(8, 8, 1))
    assert counts["moe"] == 8
    assert counts["attention"] + counts["mamba"] == 8


# ------------------------------------------------------------ kv_cache_bytes
def test_only_attention_layers_pay_for_a_kv_cache():
    """32 слоя, а платят четыре: 16 GiB вместо 128 GiB чистого трансформера."""
    assert kv_cache_bytes(layer_plan(32), 32, 128, CONTEXT_256K) == 17179869184
    assert kv_cache_bytes(layer_plan(32, 1), 32, 128, CONTEXT_256K) == 137438953472


def test_pure_ssm_has_no_kv_cache():
    assert kv_cache_bytes(layer_plan(32, 0), 32, 128, CONTEXT_256K) == 0


def test_kv_cache_grows_linearly_with_the_context():
    plan = layer_plan(32)
    assert kv_cache_bytes(plan, 32, 128, 4000) == 4 * kv_cache_bytes(plan, 32, 128, 1000)


# ----------------------------------------------------------- ssm_state_bytes
def test_ssm_state_does_not_depend_on_the_context_length():
    """Главное свойство SSM: состояние фиксировано, длины в формуле нет."""
    plan = layer_plan(32)
    assert ssm_state_bytes(plan, 4096) == 3670016


def test_pure_transformer_has_no_ssm_state():
    assert ssm_state_bytes(layer_plan(32, 1), 4096) == 0


def test_ssm_state_scales_with_the_state_size():
    plan = layer_plan(32)
    assert ssm_state_bytes(plan, 4096, 32) == 2 * ssm_state_bytes(plan, 4096, 16)


# ---------------------------------------------------------- inference_memory
def test_inference_memory_is_the_sum_of_both_parts():
    budget = inference_memory(layer_plan(32), 4096, 32, 128, CONTEXT_256K)
    assert budget["total"] == budget["kv"] + budget["ssm"]


def test_at_long_context_the_kv_cache_dominates():
    budget = inference_memory(layer_plan(32), 4096, 32, 128, CONTEXT_256K)
    assert budget["kv"] > 1000 * budget["ssm"]


def test_at_short_context_the_ssm_state_dominates():
    """Переворот: на 16 токенах состояние SSM дороже кеша. Гибрид — не панацея."""
    budget = inference_memory(layer_plan(32), 4096, 32, 128, 16)
    assert budget["ssm"] > budget["kv"]


def test_growing_the_context_leaves_the_ssm_part_untouched():
    short = inference_memory(layer_plan(32), 4096, 32, 128, 1000)
    long = inference_memory(layer_plan(32), 4096, 32, 128, 256000)
    assert short["ssm"] == long["ssm"]
    assert long["kv"] > short["kv"]


# -------------------------------------------------------- kv_cache_advantage
def test_one_to_seven_hybrid_saves_eight_times_the_cache():
    """Заявка AI21: 16 GiB против 128 GiB на 256k контекста."""
    assert kv_cache_advantage(32, 8, 32, 128, CONTEXT_256K) == APPROX(8.0)


def test_a_pure_transformer_saves_nothing():
    assert kv_cache_advantage(32, 1, 32, 128, CONTEXT_256K) == APPROX(1.0)


def test_advantage_does_not_depend_on_the_context_length():
    assert kv_cache_advantage(32, 8, 32, 128, 512) == APPROX(
        kv_cache_advantage(32, 8, 32, 128, CONTEXT_256K)
    )


def test_pure_ssm_cannot_be_expressed_as_a_ratio():
    with pytest.raises(ValueError):
        kv_cache_advantage(32, 0, 32, 128, CONTEXT_256K)
