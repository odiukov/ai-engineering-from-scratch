"""Тесты к уроку «Mixture of Experts: разреженный FFN». Правь exercise.py."""

import random

import pytest

from exercise import (
    apply_expert,
    expert_usage,
    gate_weights,
    moe_forward,
    moe_params,
    router_scores,
    select_experts,
    update_bias,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def skewed_setup(seed=0, n_experts=8, d_model=8, n_tokens=200):
    """Роутер с намеренно неравными строками: часть экспертов побеждает почти
    всегда. Именно на таком перекосе и проверяется балансировка."""
    rng = random.Random(seed)
    W_router = [
        [rng.gauss(0, 1) * (0.3 + e * 0.4) for _ in range(d_model)]
        for e in range(n_experts)
    ]
    tokens = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(n_tokens)]
    return tokens, W_router


# ----------------------------------------------------------- router_scores
def test_router_gives_one_score_per_expert():
    assert len(router_scores([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])) == 3


def test_router_score_is_the_dot_product_with_the_expert_row():
    assert router_scores([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) == APPROX(
        [1.0, 2.0, 3.0]
    )


def test_router_rejects_a_row_of_the_wrong_width():
    with pytest.raises(ValueError):
        router_scores([1.0, 2.0], [[1.0, 0.0], [1.0]])


# ---------------------------------------------------------- select_experts
def test_selection_returns_exactly_top_k_indices():
    assert len(select_experts([1.0, 5.0, 3.0, 2.0], [0.0] * 4, 2)) == 2


def test_selection_follows_the_scores_when_the_bias_is_zero():
    assert select_experts([1.0, 5.0, 3.0], [0.0, 0.0, 0.0], 2) == [1, 2]


def test_the_bias_can_promote_a_low_scoring_expert():
    """Вся балансировка DeepSeek-V3 — это вот этот сдвиг выбора."""
    assert select_experts([1.0, 5.0, 3.0], [9.0, 0.0, 0.0], 2) == [0, 1]


def test_selecting_every_expert_is_the_dense_case():
    """top_k = E — это обычный плотный FFN, никакой разреженности."""
    assert sorted(select_experts([1.0, 5.0, 3.0], [0.0] * 3, 3)) == [0, 1, 2]


def test_selection_rejects_more_experts_than_exist():
    with pytest.raises(ValueError):
        select_experts([1.0, 2.0], [0.0, 0.0], 3)


def test_selection_rejects_a_bias_of_the_wrong_length():
    with pytest.raises(ValueError):
        select_experts([1.0, 2.0, 3.0], [0.0, 0.0], 2)


# ------------------------------------------------------------ gate_weights
def test_gates_sum_to_one():
    assert sum(gate_weights([1.0, 5.0, 3.0, 0.5], [0, 1, 3])) == APPROX(1.0)


def test_equal_scores_split_the_gate_evenly():
    assert gate_weights([2.0, 2.0], [0, 1]) == APPROX([0.5, 0.5])


def test_a_bias_promoted_expert_still_gets_a_small_gate():
    """Ключевое свойство auxiliary-loss-free балансировки: биас решает, КТО
    считает, но вес в смеси остаётся по сырым скорам. Эксперт 0 попал в
    top-2 только из-за биаса 9.0 и получает вес меньше двух процентов."""
    scores = [1.0, 5.0, 3.0]
    chosen = select_experts(scores, [9.0, 0.0, 0.0], 2)
    gates = gate_weights(scores, chosen)
    assert gates[chosen.index(0)] < 0.02


def test_gate_weights_rejects_an_empty_selection():
    with pytest.raises(ValueError):
        gate_weights([1.0, 2.0], [])


# ------------------------------------------------------------ apply_expert
def test_expert_output_width_comes_from_its_matrix():
    assert len(apply_expert([1.0, 2.0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])) == 3


def test_silu_damps_a_negative_activation_without_killing_it():
    """В отличие от ReLU, SiLU оставляет отрицательным входам маленький,
    но не нулевой выход — и градиент вместе с ним."""
    out = apply_expert([1.0], [[-3.0]])[0]
    assert -0.2 < out < 0.0


def test_silu_lets_a_large_positive_activation_through():
    assert apply_expert([1.0], [[20.0]])[0] == pytest.approx(20.0, abs=1e-6)


def test_apply_expert_survives_a_huge_negative_activation():
    """Наивное 1 / (1 + exp(-v)) при v = -800 это OverflowError."""
    assert apply_expert([1.0], [[-800.0]])[0] == pytest.approx(0.0, abs=1e-9)


def test_apply_expert_rejects_a_matrix_of_the_wrong_height():
    with pytest.raises(ValueError):
        apply_expert([1.0, 2.0], [[0.5]])


# ------------------------------------------------------------- moe_forward
def test_forward_returns_a_vector_of_the_expert_hidden_width():
    experts = [[[0.1, 0.2], [0.3, 0.4]] for _ in range(4)]
    W_router = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
    out, _ = moe_forward([1.0, 2.0], experts, W_router, 2, [0.0] * 4)
    assert len(out) == 2


def test_top_one_routing_is_exactly_the_chosen_expert():
    """При top_k = 1 вес смеси равен единице, значит выход слоя — ровно выход
    выбранного эксперта, без всяких поправок."""
    experts = [[[2.0]], [[3.0]], [[-1.0]]]
    W_router = [[1.0], [2.0], [0.5]]
    out, indices = moe_forward([1.0], experts, W_router, 1, [0.0] * 3)
    assert out == APPROX(apply_expert([1.0], experts[indices[0]]))


def test_forward_is_the_gate_weighted_sum_of_the_chosen_experts():
    x = [1.0, -0.5]
    experts = [[[1.0, 0.5], [0.5, 1.0]], [[2.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [1.0, -1.0]]]
    W_router = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scores = router_scores(x, W_router)
    chosen = select_experts(scores, [0.0] * 3, 2)
    gates = gate_weights(scores, chosen)
    expected = [0.0, 0.0]
    for idx, gate in zip(chosen, gates):
        for j, v in enumerate(apply_expert(x, experts[idx])):
            expected[j] += gate * v
    out, _ = moe_forward(x, experts, W_router, 2, [0.0] * 3)
    assert out == APPROX(expected)


def test_unselected_experts_never_run():
    """Проверка разреженности по-честному: невыбранным экспертам подкладываем
    матрицы из NaN. Если реализация их посчитает, NaN просочится в выход."""
    x = [1.0, -0.5]
    W_router = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
    chosen = select_experts(router_scores(x, W_router), [0.0] * 4, 2)
    nan = float("nan")
    experts = [[[nan, nan], [nan, nan]] for _ in range(4)]
    for i in chosen:
        experts[i] = [[1.0, 0.5], [0.5, 1.0]]
    out, _ = moe_forward(x, experts, W_router, 2, [0.0] * 4)
    assert all(v == v for v in out)      # NaN не равен сам себе


def test_the_bias_reroutes_the_token_to_another_expert():
    experts = [[[2.0]], [[3.0]], [[-1.0]]]
    W_router = [[1.0], [2.0], [0.5]]
    _, plain = moe_forward([1.0], experts, W_router, 1, [0.0, 0.0, 0.0])
    _, biased = moe_forward([1.0], experts, W_router, 1, [0.0, 0.0, 10.0])
    assert plain != biased


# ------------------------------------------------------------ expert_usage
def test_every_token_is_counted_exactly_top_k_times():
    tokens, W_router = skewed_setup()
    assert sum(expert_usage(tokens, W_router, 2, [0.0] * 8)) == len(tokens) * 2


def test_an_unbalanced_router_starves_some_experts():
    """Без балансировки разрыв между самым и наименее нагруженным экспертом
    огромен, и параметры голодающих экспертов занимают VRAM впустую."""
    tokens, W_router = skewed_setup()
    usage = expert_usage(tokens, W_router, 2, [0.0] * 8)
    assert max(usage) > 2 * min(usage)


def test_the_bias_loop_evens_out_expert_usage():
    """Пункт урока целиком: тот же роутер, те же токены, ни одного изменения
    в весах — только биас, обновляемый вне функции потерь."""
    tokens, W_router = skewed_setup()
    top_k, n_experts = 2, 8
    target = len(tokens) * top_k / n_experts
    bias = [0.0] * n_experts
    usage = expert_usage(tokens, W_router, top_k, bias)
    start_spread = max(usage) - min(usage)
    for _ in range(40):
        bias = update_bias(bias, usage, target, 0.1)
        usage = expert_usage(tokens, W_router, top_k, bias)
    assert max(usage) - min(usage) < 0.4 * start_spread


# ------------------------------------------------------------- update_bias
def test_an_overused_expert_gets_its_bias_lowered():
    assert update_bias([0.0, 0.0], [10, 2], 6, 0.1) == APPROX([-0.1, 0.1])


def test_an_expert_exactly_on_target_keeps_its_bias():
    assert update_bias([0.5], [6], 6, 0.1) == APPROX([0.5])


def test_update_bias_does_not_mutate_the_old_bias():
    bias = [0.0, 0.0]
    update_bias(bias, [10, 2], 6, 0.1)
    assert bias == [0.0, 0.0]


def test_a_bigger_gamma_moves_the_bias_further():
    slow = update_bias([0.0], [10], 6, 0.05)[0]
    fast = update_bias([0.0], [10], 6, 0.5)[0]
    assert fast < slow < 0.0


# -------------------------------------------------------------- moe_params
def test_more_experts_cost_memory_but_not_compute():
    """Суть MoE: total растёт с числом экспертов, active не двигается."""
    small_total, small_active = moe_params(8, 1000, 2)
    big_total, big_active = moe_params(256, 1000, 2)
    assert big_total > small_total and big_active == small_active


def test_active_parameters_scale_with_top_k():
    assert moe_params(64, 1000, 8)[1] == 4 * moe_params(64, 1000, 2)[1]


def test_a_shared_expert_is_always_active():
    """Shared expert проходит каждый токен, поэтому он и в total, и в active."""
    without = moe_params(256, 1000, 8, 0)
    with_shared = moe_params(256, 1000, 8, 1)
    assert with_shared[1] - without[1] == 1000
    assert with_shared[0] - without[0] == 1000


def test_a_deepseek_shaped_config_activates_a_few_percent():
    """256 маршрутизируемых экспертов плюс один shared, активны восемь:
    примерно 3.5% параметров на токен."""
    total, active = moe_params(256, 1000, 8, 1)
    assert 0.02 < active / total < 0.06


def test_moe_params_rejects_activating_more_experts_than_exist():
    with pytest.raises(ValueError):
        moe_params(4, 1000, 8)
