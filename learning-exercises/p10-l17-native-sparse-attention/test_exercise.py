"""Тесты к уроку «Native Sparse Attention (DeepSeek NSA)». Правь exercise.py."""

import pytest

from exercise import (
    attend,
    attention_weights,
    compress_blocks,
    keys_per_query,
    nsa_attention,
    selected_branch,
    softmax,
    top_k_blocks,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def toy_sequence(n=8, d=4):
    """Детерминированная последовательность: без rng, но и не константа."""
    K = [[((i * 7 + j * 3) % 11) / 5.0 - 1.0 for j in range(d)] for i in range(n)]
    V = [[float(i), float(i * 2), float(-i)] for i in range(n)]
    return K, V


QUERY = [0.5, -1.0, 0.25, 0.75]


def dense(q, K, V):
    """Эталон, с которым обязана совпасть разреженная схема при полном окне."""
    return attend(attention_weights(q, K), V)


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, 2.0, 3.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0])


# -------------------------------------------------------- attention_weights
def test_identical_keys_give_uniform_attention():
    assert attention_weights([1.0, 0.0], [[0.0, 0.0]] * 4) == pytest.approx([0.25] * 4)


def test_attention_weights_sum_to_one():
    K, _ = toy_sequence()
    assert sum(attention_weights(QUERY, K)) == APPROX(1.0)


def test_empty_key_list_is_rejected():
    """Пустая ветка — это баг раскладки блоков, а не «ноль внимания»."""
    with pytest.raises(ValueError):
        attention_weights(QUERY, [])


# ----------------------------------------------------------- compress_blocks
def test_compression_averages_each_block():
    assert flat(compress_blocks([[1.0], [3.0], [5.0], [7.0]], 2)) == APPROX([2.0, 6.0])


def test_block_size_one_is_the_identity():
    """l = 1 значит «не сжимать» — это опора всех тестов на вырождение."""
    K, _ = toy_sequence()
    assert flat(compress_blocks(K, 1)) == pytest.approx(flat(K))


def test_a_ragged_tail_averages_only_what_it_has():
    """Делить хвост на l вместо его настоящей длины — занижать конспект."""
    assert flat(compress_blocks([[1.0], [3.0], [5.0]], 2)) == APPROX([2.0, 5.0])


def test_a_block_larger_than_the_sequence_gives_one_summary():
    out = compress_blocks([[1.0], [2.0], [3.0]], 100)
    assert len(out) == 1
    assert out[0] == APPROX([2.0])


def test_number_of_summaries_is_ceil_of_n_over_l():
    K, _ = toy_sequence(n=8)
    assert len(compress_blocks(K, 3)) == 3


def test_zero_block_size_is_rejected():
    with pytest.raises(ValueError):
        compress_blocks([[1.0]], 0)


# -------------------------------------------------------------- top_k_blocks
def test_top_k_picks_the_highest_scoring_blocks():
    assert top_k_blocks([0.1, 0.5, 0.2, 0.4], 2) == [1, 3]


def test_top_k_returns_indices_in_ascending_order():
    """Порядок нужен для чтения токенов подряд — так кернел грузит память."""
    assert top_k_blocks([0.9, 0.1, 0.8, 0.2], 2) == [0, 2]


def test_top_k_larger_than_the_pool_returns_everything():
    assert top_k_blocks([0.1, 0.5], 5) == [0, 1]


def test_top_k_breaks_ties_by_lowest_index():
    assert top_k_blocks([0.25] * 4, 2) == [0, 1]


# ----------------------------------------------------------- selected_branch
def test_selected_branch_with_full_window_equals_dense_attention():
    """Главная проверка разреженности: при l=1 и k=n она обязана вырождаться."""
    K, V = toy_sequence()
    got = selected_branch(QUERY, K, V, 1, len(K))
    assert got == pytest.approx(dense(QUERY, K, V))


def test_selected_branch_reads_only_the_chosen_blocks():
    """Значения вне выбранных блоков на ответ не влияют."""
    K, V = toy_sequence(n=8)
    V_changed = [row[:] for row in V]
    V_changed[7] = [999.0, -999.0, 999.0]  # последний блок при l=4, k=1
    a = selected_branch(QUERY, K, V, 4, 1)
    b = selected_branch(QUERY, K, V_changed, 4, 1)
    chosen_last = a != pytest.approx(b)
    # ровно один из двух блоков выбран, значит ровно один из выходов меняется
    V_changed2 = [row[:] for row in V]
    V_changed2[0] = [999.0, -999.0, 999.0]
    c = selected_branch(QUERY, K, V_changed2, 4, 1)
    chosen_first = a != pytest.approx(c)
    assert chosen_last != chosen_first


def test_selected_branch_with_one_block_is_attention_inside_it():
    K, V = toy_sequence(n=4)
    got = selected_branch(QUERY, K, V, 4, 1)
    assert got == pytest.approx(dense(QUERY, K, V))


def test_selecting_more_blocks_than_exist_is_harmless():
    K, V = toy_sequence(n=6)
    got = selected_branch(QUERY, K, V, 2, 99)
    assert got == pytest.approx(dense(QUERY, K, V))


# ------------------------------------------------------------ nsa_attention
def test_window_branch_alone_with_full_window_is_dense_attention():
    K, V = toy_sequence()
    got = nsa_attention(QUERY, K, V, 4, 1, 2, len(K), (0.0, 0.0, 1.0))
    assert got == pytest.approx(dense(QUERY, K, V))


def test_selected_branch_alone_with_full_window_is_dense_attention():
    K, V = toy_sequence()
    got = nsa_attention(QUERY, K, V, 4, len(K), 1, 2, (0.0, 1.0, 0.0))
    assert got == pytest.approx(dense(QUERY, K, V))


def test_compressed_branch_alone_with_block_size_one_is_dense_attention():
    K, V = toy_sequence()
    got = nsa_attention(QUERY, K, V, 1, 1, 4, 2, (1.0, 0.0, 0.0))
    assert got == pytest.approx(dense(QUERY, K, V))


def test_all_gates_at_zero_output_nothing():
    K, V = toy_sequence()
    assert nsa_attention(QUERY, K, V, 2, 2, 4, 3, (0.0, 0.0, 0.0)) == APPROX(
        [0.0, 0.0, 0.0]
    )


def test_gates_do_not_have_to_sum_to_one():
    """В статье это выход MLP: ветки взвешиваются независимо."""
    K, V = toy_sequence()
    once = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (1.0, 1.0, 1.0))
    twice = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (2.0, 2.0, 2.0))
    assert twice == pytest.approx([2 * x for x in once])


def test_the_window_branch_only_sees_recent_tokens():
    """Правка далёкого прошлого не трогает оконную ветку."""
    K, V = toy_sequence(n=8)
    V_old = [row[:] for row in V]
    V_old[0] = [500.0, 500.0, 500.0]
    a = nsa_attention(QUERY, K, V, 8, 1, 4, 2, (0.0, 0.0, 1.0))
    b = nsa_attention(QUERY, K, V_old, 8, 1, 4, 2, (0.0, 0.0, 1.0))
    assert a == pytest.approx(b)


def test_the_compressed_branch_does_see_the_whole_sequence():
    """А сжатая ветка видит: в этом и разница между NSA и скользящим окном."""
    K, V = toy_sequence(n=8)
    V_old = [row[:] for row in V]
    V_old[0] = [500.0, 500.0, 500.0]
    a = nsa_attention(QUERY, K, V, 4, 1, 2, 2, (1.0, 0.0, 0.0))
    b = nsa_attention(QUERY, K, V_old, 4, 1, 2, 2, (1.0, 0.0, 0.0))
    assert a != pytest.approx(b)


def test_nsa_is_a_linear_combination_of_its_three_branches():
    K, V = toy_sequence()
    cmp_only = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (1.0, 0.0, 0.0))
    sel_only = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (0.0, 1.0, 0.0))
    win_only = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (0.0, 0.0, 1.0))
    mixed = nsa_attention(QUERY, K, V, 2, 2, 4, 3, (0.2, 0.5, 0.3))
    expected = [
        0.2 * a + 0.5 * b + 0.3 * c for a, b, c in zip(cmp_only, sel_only, win_only)
    ]
    assert mixed == pytest.approx(expected)


def test_selection_block_size_is_independent_of_compression_block_size():
    K, V = toy_sequence()
    small_compression = nsa_attention(QUERY, K, V, 1, 1, 4, 2, (0.0, 1.0, 0.0))
    large_compression = nsa_attention(QUERY, K, V, 8, 1, 4, 2, (0.0, 1.0, 0.0))
    assert small_compression == pytest.approx(large_compression)


# ----------------------------------------------------------- keys_per_query
def test_budget_matches_the_lesson_numbers_at_64k():
    budget = keys_per_query(64000, 64, 16, 64, 512)
    assert (budget["compressed"], budget["selected"], budget["window"]) == (1000, 1024, 512)
    assert budget["total"] == 2536


def test_reduction_at_64k_is_about_25x():
    assert keys_per_query(64000, 64, 16, 64, 512)["reduction"] == pytest.approx(25.2, abs=0.1)


def test_reduction_at_128k_is_about_36x():
    """Выигрыш растёт с длиной контекста — ради этого всё и затевалось."""
    assert keys_per_query(128000, 64, 16, 64, 512)["reduction"] == pytest.approx(36.2, abs=0.1)


def test_savings_grow_with_context_length():
    short = keys_per_query(16000, 64, 16, 64, 512)["reduction"]
    long = keys_per_query(128000, 64, 16, 64, 512)["reduction"]
    assert long > short


def test_no_branch_can_read_more_keys_than_the_sequence_has():
    budget = keys_per_query(100, 64, 16, 64, 512)
    assert budget["selected"] == 100
    assert budget["window"] == 100


def test_compressed_count_rounds_up():
    """Хвостовой блок существует, даже если он неполный."""
    assert keys_per_query(65, 64, 1, 64, 1)["compressed"] == 2


def test_cost_uses_selection_size_not_compression_size_for_selected_keys():
    budget = keys_per_query(1000, 32, 3, 64, 1)
    assert budget["compressed"] == 32
    assert budget["selected"] == 192


def test_short_context_gives_no_savings():
    """Под 16k три ветки съедают больше, чем экономят — так в уроке и сказано."""
    assert keys_per_query(1024, 64, 16, 64, 512)["reduction"] < 1.0
