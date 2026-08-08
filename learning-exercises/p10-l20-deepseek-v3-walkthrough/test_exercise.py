"""Тесты к уроку «Разбор архитектуры DeepSeek-V3». Правь exercise.py."""

import random

import pytest

from exercise import (
    DEEPSEEK_V3,
    attention_params,
    balance_bias_step,
    expert_load,
    expert_params,
    gqa_kv_cache_bytes,
    mla_kv_cache_bytes,
    model_parameters,
    route_topk,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CONTEXT_128K = 131072


def skewed_rows(seed=0, tokens=200, experts=8, tilt=1.5):
    """Логиты роутера с перекосом в сторону первых двух экспертов."""
    rng = random.Random(seed)
    return [
        [rng.gauss(0.0, 1.0) + (tilt if e < 2 else 0.0) for e in range(experts)]
        for _ in range(tokens)
    ]


# ------------------------------------------------------------- KV-кеш: MLA
def test_mla_cache_stores_the_latent_and_rope_key_per_token_and_layer():
    assert mla_kv_cache_bytes(61, 512, CONTEXT_128K, 64) == 9210691584


def test_mla_cache_grows_linearly_with_context():
    short = mla_kv_cache_bytes(61, 512, 1000)
    long = mla_kv_cache_bytes(61, 512, 4000)
    assert long == 4 * short


def test_mla_cache_stores_the_rope_key_component_too():
    """KV-латент один, но к нему добавляется несжатый RoPE-хвост ключа."""
    assert mla_kv_cache_bytes(1, 512, 1, 64, bytes_per_element=1) == 576


def test_mla_cache_has_no_hidden_factor_of_two_for_keys_and_values():
    assert mla_kv_cache_bytes(1, 512, 1, 64, bytes_per_element=2) == 2 * (512 + 64)


# ------------------------------------------------------------- KV-кеш: GQA
def test_gqa_cache_counts_keys_and_values_separately():
    assert gqa_kv_cache_bytes(61, 8, 128, CONTEXT_128K) == 32749125632


def test_gqa_cache_doubles_when_kv_heads_double():
    assert gqa_kv_cache_bytes(61, 16, 128, 1000) == 2 * gqa_kv_cache_bytes(61, 8, 128, 1000)


def test_mla_is_about_three_and_a_half_times_smaller_than_gqa():
    """8.6 GiB против 30.5 GiB: RoPE-хвост уменьшает заявленный выигрыш."""
    mla = mla_kv_cache_bytes(61, 512, CONTEXT_128K, 64)
    gqa = gqa_kv_cache_bytes(61, 8, 128, CONTEXT_128K)
    assert gqa / mla == pytest.approx(32 / 9)


# ---------------------------------------------------------- expert_params
def test_swiglu_expert_counts_three_matrices():
    assert expert_params(7168, 2048) == 44040192


def test_expert_params_scale_with_the_hidden_size():
    assert expert_params(2, 5) == 2 * expert_params(1, 5)


def test_top8_plus_shared_expert_equals_the_dense_mlp():
    """9 экспертов по 2048 — это ровно dense-MLP на 18432. Так и задумано."""
    nine_experts = 9 * expert_params(7168, 2048)
    assert nine_experts == expert_params(7168, 18432)


# -------------------------------------------------------- attention_params
def test_mla_block_weight_count():
    assert attention_params(7168, 128, 128, 512) == 255328256


def test_mla_adds_weights_while_it_saves_cache():
    """Сжатие-разжатие KV — это лишние веса ради экономии памяти под кеш."""
    plain = 2 * 7168 * (128 * 128)  # только W_q и W_o
    assert attention_params(7168, 128, 128, 512) > plain


# --------------------------------------------------------- model_parameters
def test_total_matches_the_published_671b():
    total = model_parameters(DEEPSEEK_V3)["total"]
    assert 0.95 < total / 671e9 < 1.05


def test_active_matches_the_published_37b():
    active = model_parameters(DEEPSEEK_V3)["active"]
    assert 30e9 < active < 45e9


def test_sparsity_is_the_active_share():
    counts = model_parameters(DEEPSEEK_V3)
    assert counts["sparsity"] == APPROX(counts["active"] / counts["total"])
    assert 0.04 < counts["sparsity"] < 0.08


def test_routed_experts_eat_almost_the_whole_budget():
    """Больше 90% весов лежит в экспертах, которые на токен не включаются."""
    counts = model_parameters(DEEPSEEK_V3)
    assert (counts["total"] - counts["active"]) / counts["total"] > 0.9


def test_more_experts_grow_total_but_not_active():
    """512 экспертов вместо 256: total удваивается, active не меняется."""
    wider = dict(DEEPSEEK_V3, num_experts=512)
    base, big = model_parameters(DEEPSEEK_V3), model_parameters(wider)
    assert big["total"] > base["total"]
    assert big["active"] == base["active"]


def test_a_dense_model_has_sparsity_one():
    """Все эксперты на слое активны — разреженности нет."""
    dense = dict(DEEPSEEK_V3, num_experts=8, num_experts_per_tok=8, shared_experts=0)
    assert model_parameters(dense)["sparsity"] == APPROX(1.0)


# --------------------------------------------------------------- route_topk
def test_route_topk_picks_the_largest_logits():
    assert route_topk([0.1, 0.9, 0.5], 2) == [1, 2]


def test_route_topk_breaks_ties_by_the_smaller_index():
    assert route_topk([1.0, 1.0, 0.0], 2) == [0, 1]


def test_bias_moves_the_routing_decision():
    assert route_topk([0.1, 0.9, 0.5], 2) == [1, 2]
    assert route_topk([0.1, 0.9, 0.5], 2, [1.0, 0.0, 0.0]) == [0, 1]


def test_route_topk_rejects_k_beyond_the_expert_count():
    with pytest.raises(ValueError):
        route_topk([0.1, 0.9, 0.5], 4)


# -------------------------------------------------------------- expert_load
def test_top_k_activates_exactly_k_experts_per_token():
    rows = skewed_rows(seed=1, tokens=50)
    for k in (1, 2, 4):
        assert sum(expert_load(rows, 8, k)) == 50 * k


def test_expert_load_counts_every_expert_slot():
    assert expert_load([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], 3, 2) == [1, 2, 1]


def test_a_skewed_router_overloads_a_couple_of_experts():
    load = expert_load(skewed_rows(), 8, 2)
    assert max(load) > 3 * min(load)


# -------------------------------------------------------- balance_bias_step
def test_overloaded_expert_gets_a_lower_bias():
    assert balance_bias_step([0.0, 0.0, 0.0], [10, 2, 0], 0.1) == pytest.approx(
        [-0.1, 0.1, 0.1]
    )


def test_balanced_load_leaves_the_bias_alone():
    assert balance_bias_step([0.3, -0.3], [5, 5], 0.1) == pytest.approx([0.3, -0.3])


def test_balance_bias_step_does_not_mutate_its_input():
    bias = [0.0, 0.0]
    balance_bias_step(bias, [7, 1], 0.1)
    assert bias == [0.0, 0.0]


def test_bias_updates_flatten_a_skewed_router():
    """Главное свойство: балансировка без вспомогательного loss работает."""
    rows = skewed_rows()
    before = expert_load(rows, 8, 2)
    bias = [0.0] * 8
    for _ in range(60):
        bias = balance_bias_step(bias, expert_load(rows, 8, 2, bias), 0.05)
    after = expert_load(rows, 8, 2, bias)
    assert max(after) - min(after) < (max(before) - min(before)) / 5
    assert sum(after) == sum(before)
