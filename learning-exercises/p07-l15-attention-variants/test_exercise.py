"""Тесты к уроку «Варианты внимания: sliding window, sparse, differential». Правь exercise.py."""

import random

import pytest

from exercise import (
    causal_mask,
    count_attended,
    diff_attention_row,
    effective_receptive_field,
    kv_cache_bytes,
    masked_attention_row,
    strided_mask,
    swa_mask,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
NEG_INF = float("-inf")


def flat(M):
    """pytest.approx не сравнивает вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def toy_kv(n, d, seed):
    """Детерминированные K и V: тесты обязаны быть воспроизводимы."""
    rng = random.Random(seed)
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    q = [rng.gauss(0, 1) for _ in range(d)]
    return q, K, V


# ------------------------------------------------------------ causal_mask
def test_causal_mask_is_lower_triangular():
    M = causal_mask(5)
    for i in range(5):
        for j in range(5):
            assert M[i][j] == (0.0 if j <= i else NEG_INF)


def test_causal_mask_blocks_the_future_with_minus_infinity():
    """Не нулём: маска складывается с логитами ДО softmax."""
    assert causal_mask(3)[0][2] == NEG_INF


def test_causal_mask_attends_half_the_matrix():
    assert count_attended(causal_mask(100)) == 100 * 101 // 2


# --------------------------------------------------------------- swa_mask
def test_swa_with_full_window_reproduces_causal_mask_bit_identically():
    """Главная проверка урока: window >= n возвращает полное causal-внимание."""
    assert swa_mask(9, 9) == causal_mask(9)
    assert swa_mask(9, 100) == causal_mask(9)


def test_swa_with_window_one_leaves_only_the_diagonal():
    M = swa_mask(6, 1)
    assert count_attended(M) == 6
    assert all(M[i][i] == 0.0 for i in range(6))


def test_swa_forgets_tokens_older_than_the_window():
    M = swa_mask(8, 4)
    assert M[7][3] == NEG_INF
    assert M[7][4] == 0.0


def test_swa_cost_grows_linearly_not_quadratically():
    """N*window вместо N^2/2: удвоили N — стоимость удвоилась, а не учетверилась."""
    a = count_attended(swa_mask(100, 8))
    b = count_attended(swa_mask(200, 8))
    assert b - a == 800


def test_swa_attended_cells_match_the_closed_form():
    n, w = 12, 5
    expected = w * (w + 1) // 2 + (n - w) * w
    assert count_attended(swa_mask(n, w)) == expected


def test_swa_is_never_more_permissive_than_causal():
    swa, causal = swa_mask(10, 3), causal_mask(10)
    for a, b in zip(flat(swa), flat(causal)):
        assert not (a == 0.0 and b == NEG_INF)


# ------------------------------------------------------------ strided_mask
def test_strided_mask_always_lets_everyone_see_token_zero():
    """0 кратен любому stride — вот откуда берётся глобальный токен."""
    M = strided_mask(10, 2, 4)
    assert all(M[i][0] == 0.0 for i in range(10))


def test_strided_mask_adds_every_stride_th_token():
    M = strided_mask(10, 2, 3)
    assert M[9][3] == 0.0 and M[9][6] == 0.0
    assert M[9][4] == NEG_INF


def test_strided_mask_with_stride_one_is_full_causal():
    assert strided_mask(7, 1, 1) == causal_mask(7)


def test_strided_mask_is_a_superset_of_its_window():
    swa = swa_mask(12, 3)
    strided = strided_mask(12, 3, 4)
    assert count_attended(strided) > count_attended(swa)
    for a, b in zip(flat(swa), flat(strided)):
        assert not (a == 0.0 and b == NEG_INF)


def test_strided_mask_stays_causal():
    M = strided_mask(9, 3, 2)
    for i in range(9):
        for j in range(i + 1, 9):
            assert M[i][j] == NEG_INF


# ------------------------------------------------ effective_receptive_field
def test_one_layer_sees_exactly_its_window():
    assert effective_receptive_field(1, 1024) == 1024


def test_stacked_windows_reach_much_further_than_one_window():
    """Информация течёт вперёд через перекрытия — так работал Mistral 7B."""
    assert effective_receptive_field(32, 1024) == 32737
    assert effective_receptive_field(32, 1024) > 30 * 1024


def test_receptive_field_grows_with_depth():
    fields = [effective_receptive_field(L, 8) for L in range(1, 6)]
    assert fields == sorted(fields)
    assert len(set(fields)) == 5


def test_window_one_never_looks_back():
    """Окно из одной позиции не переносит информацию, сколько слоёв ни ставь."""
    assert effective_receptive_field(50, 1) == 1


# ------------------------------------------------- masked_attention_row
def test_attention_weights_sum_to_one():
    q, K, V = toy_kv(7, 4, seed=1)
    _, w = masked_attention_row(q, K, V, causal_mask(7)[6])
    assert sum(w) == APPROX(1.0)


def test_masked_positions_get_exactly_zero_weight():
    """Не «почти ноль», а ровно 0.0: маскированный токен не участвует вообще."""
    q, K, V = toy_kv(7, 4, seed=2)
    row = swa_mask(7, 3)[6]
    _, w = masked_attention_row(q, K, V, row)
    assert [w[j] for j in range(7) if row[j] == NEG_INF] == [0.0] * 4
    assert sum(w) == APPROX(1.0)


def test_single_allowed_position_gets_all_the_weight():
    q, K, V = toy_kv(5, 4, seed=3)
    out, w = masked_attention_row(q, K, V, [0.0, NEG_INF, NEG_INF, NEG_INF, NEG_INF])
    assert w == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert out == pytest.approx(V[0])


def test_swa_with_full_window_gives_the_same_output_as_full_causal():
    """Раз маски совпадают бит в бит, обязаны совпасть и выходы."""
    q, K, V = toy_kv(9, 4, seed=4)
    full, _ = masked_attention_row(q, K, V, causal_mask(9)[8])
    wide, _ = masked_attention_row(q, K, V, swa_mask(9, 9)[8])
    assert wide == pytest.approx(full, abs=1e-12)


def test_narrow_window_changes_the_output():
    """Окно из двух позиций видит другую картину — за экономию платят качеством."""
    q, K, V = toy_kv(9, 4, seed=5)
    full, _ = masked_attention_row(q, K, V, causal_mask(9)[8])
    narrow, _ = masked_attention_row(q, K, V, swa_mask(9, 2)[8])
    assert max(abs(a - b) for a, b in zip(full, narrow)) > 1e-3


def test_attention_output_is_a_convex_combination_of_allowed_values():
    q, K, V = toy_kv(8, 4, seed=6)
    row = swa_mask(8, 4)[7]
    out, _ = masked_attention_row(q, K, V, row)
    allowed = [V[j] for j in range(8) if row[j] == 0.0]
    for j in range(len(out)):
        column = [v[j] for v in allowed]
        assert min(column) - 1e-9 <= out[j] <= max(column) + 1e-9


def test_uniform_scores_give_uniform_weights():
    q = [0.0, 0.0]
    K = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    V = [[1.0], [2.0], [3.0]]
    out, w = masked_attention_row(q, K, V, [0.0, 0.0, 0.0])
    assert w == pytest.approx([1 / 3] * 3)
    assert out == pytest.approx([2.0])


# ---------------------------------------------------- diff_attention_row
def test_diff_attention_with_lambda_zero_is_plain_attention():
    q, K, V = toy_kv(6, 4, seed=7)
    row = causal_mask(6)[5]
    plain, w_plain = masked_attention_row(q, K, V, row)
    out, w = diff_attention_row(q, q, K, K, V, row, lam=0.0)
    assert w == pytest.approx(w_plain, abs=1e-12)
    assert out == pytest.approx(plain, abs=1e-12)


def test_identical_maps_and_lambda_one_cancel_completely():
    """Две одинаковые карты с lam=1 вычитаются в нуль — вычитание работает."""
    q, K, V = toy_kv(6, 4, seed=8)
    out, w = diff_attention_row(q, q, K, K, V, causal_mask(6)[5], lam=1.0)
    assert w == pytest.approx([0.0] * 6, abs=1e-12)
    assert out == pytest.approx([0.0] * 4, abs=1e-12)


def test_diff_weights_sum_to_one_minus_lambda():
    """Нормировка сознательно сломана: именно она и рождала attention sink."""
    q, K, V = toy_kv(6, 4, seed=9)
    for lam in (0.0, 0.3, 0.5, 0.8):
        _, w = diff_attention_row(q, q, K, K, V, causal_mask(6)[5], lam=lam)
        assert sum(w) == pytest.approx(1.0 - lam, abs=1e-12)


def test_diff_attention_can_produce_negative_weights():
    """Softmax отрицательных весов не даёт никогда — а вычитание даёт."""
    K = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    V = [[1.0], [2.0], [3.0], [4.0]]
    q_flat = [0.0, 0.0]          # все логиты равны -> веса ровно равномерные
    q_sink = [10.0, 0.0]         # вторая карта смотрит почти только на позицию 0
    _, w = diff_attention_row(q_flat, q_sink, K, K, V, [0.0] * 4, lam=0.5)
    assert min(w) < 0.0


def test_diff_attention_drains_the_sink_position():
    """Позиция 0 собирает паразитный вес; вторая карта его забирает обратно."""
    K = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    V = [[1.0], [2.0], [3.0], [4.0]]
    q_flat = [0.0, 0.0]
    q_sink = [10.0, 0.0]
    _, w_plain = masked_attention_row(q_flat, K, V, [0.0] * 4)
    _, w_diff = diff_attention_row(q_flat, q_sink, K, K, V, [0.0] * 4, lam=0.5)
    assert w_diff[0] < w_plain[0]
    assert w_plain[0] == APPROX(0.25)


def test_diff_attention_respects_the_mask():
    q, K, V = toy_kv(7, 4, seed=10)
    row = swa_mask(7, 3)[6]
    _, w = diff_attention_row(q, q, K, K, V, row, lam=0.5)
    assert [w[j] for j in range(7) if row[j] == NEG_INF] == [0.0] * 4


# --------------------------------------------------------- kv_cache_bytes
def test_kv_cache_bytes_without_window_is_the_full_prefix():
    assert kv_cache_bytes(80, 8, 128, 131072) == 42_949_672_960


def test_window_shrinks_the_cache_by_the_ratio_of_lengths():
    """N=128K, window=1024 — ровно 128-кратная экономия. Главный выигрыш SWA."""
    full = kv_cache_bytes(80, 8, 128, 131072)
    swa = kv_cache_bytes(80, 8, 128, 131072, window=1024)
    assert full / swa == APPROX(128.0)


def test_window_wider_than_the_sequence_changes_nothing():
    """Окно не может хранить больше токенов, чем есть в последовательности."""
    assert kv_cache_bytes(80, 8, 128, 512, window=999999) == kv_cache_bytes(80, 8, 128, 512)


def test_differential_attention_pays_two_caches():
    single = kv_cache_bytes(80, 8, 128, 131072)
    assert 2 * single == kv_cache_bytes(160, 8, 128, 131072)


def test_gemma3_five_to_one_mix_sits_between_pure_swa_and_full():
    """5 слоёв SWA на 1 глобальный: память режется в разы, но retrieval живёт."""
    full = kv_cache_bytes(60, 8, 128, 131072)
    pure_swa = kv_cache_bytes(60, 8, 128, 131072, window=1024)
    mixed = (
        kv_cache_bytes(50, 8, 128, 131072, window=1024)
        + kv_cache_bytes(10, 8, 128, 131072)
    )
    assert pure_swa < mixed < full
    assert full / mixed > 5.0
    assert mixed / pure_swa > 10.0


def test_fp8_cache_is_half_of_fp16():
    assert kv_cache_bytes(80, 8, 128, 4096, dtype_bytes=1) * 2 == kv_cache_bytes(
        80, 8, 128, 4096, dtype_bytes=2
    )
