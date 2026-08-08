"""Тесты к уроку «Внутренности serving-движка». Правь exercise.py."""

import pytest

from exercise import (
    CHUNK_SIZE,
    KV_BLOCK_SIZE,
    MIXED_WORKLOAD,
    BlockPool,
    OutOfKVBlocks,
    blocks_for,
    chunk_plan,
    contiguous_waste,
    paged_waste,
    percentile,
    schedule_continuous,
    schedule_static,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

POOL_BLOCKS = 1800
REAL_OUTPUT_TOKENS = sum(r[2] for r in MIXED_WORKLOAD)


# -------------------------------------------------------------- blocks_for
def test_blocks_for_an_exact_fit():
    assert blocks_for(16, 16) == 1


def test_blocks_for_rounds_up_on_a_single_extra_token():
    assert blocks_for(17, 16) == 2


def test_blocks_for_nothing_is_nothing():
    assert blocks_for(0, 16) == 0


# --------------------------------------------------------- contiguous_waste
def test_contiguous_waste_on_short_sequences_reserved_for_the_worst_case():
    assert contiguous_waste([1500, 1500], 8192) == APPROX(1 - 3000 / 16384)


def test_contiguous_waste_is_zero_when_everyone_uses_the_full_reservation():
    assert contiguous_waste([8192], 8192) == APPROX(0.0)


def test_contiguous_waste_grows_with_the_reserved_window():
    """Поднял max_len ради длинных промптов — оплатил его на каждом коротком."""
    assert contiguous_waste([1500], 16384) > contiguous_waste([1500], 8192)


# -------------------------------------------------------------- paged_waste
def test_paged_waste_loses_only_the_tail_of_the_last_block():
    assert paged_waste([1500, 1500], 16) == APPROX((1504 * 2 - 3000) / (1504 * 2))


def test_paged_waste_is_zero_on_block_aligned_lengths():
    assert paged_waste([16, 32], 16) == APPROX(0.0)


def test_paged_waste_stays_under_the_four_percent_the_lesson_claims():
    """Потеря ограничена хвостом одного блока — при любых длинах, длинных и коротких."""
    assert paged_waste([1500] * 8, 16) < 0.04
    assert paged_waste([137, 4001, 60, 8192], 16) < 0.04


def test_paged_waste_is_hundreds_of_times_below_contiguous_on_the_same_data():
    """Те самые «60-80% против <4%» из урока — на одних и тех же длинах."""
    lens = [1500] * 8
    assert paged_waste(lens, KV_BLOCK_SIZE) * 100 < contiguous_waste(lens, 8192)


# ---------------------------------------------------------------- BlockPool
def test_a_new_pool_has_everything_free():
    pool = BlockPool(10, 16)
    assert (pool.used_blocks(), pool.free_blocks()) == (0, 10)


def test_allocate_takes_whole_blocks():
    pool = BlockPool(10, 16)
    pool.allocate("a", 20)
    assert (pool.used_blocks(), pool.free_blocks()) == (2, 8)


def test_free_returns_the_blocks_to_the_pool():
    """Возврат в общий пул — то, что позволяет принять новый запрос сразу."""
    pool = BlockPool(10, 16)
    pool.allocate("a", 20)
    pool.free("a")
    assert pool.free_blocks() == 10


def test_append_token_does_not_take_a_block_until_the_current_one_is_full():
    pool = BlockPool(10, 16)
    pool.allocate("a", 20)
    pool.append_token("a")
    assert pool.used_blocks() == 2


def test_append_token_takes_a_new_block_when_the_current_one_fills_up():
    pool = BlockPool(10, 16)
    pool.allocate("a", 32)
    pool.append_token("a")
    assert pool.used_blocks() == 3


def test_allocate_refuses_what_does_not_fit():
    pool = BlockPool(2, 16)
    with pytest.raises(OutOfKVBlocks):
        pool.allocate("a", 100)


def test_allocate_rejects_a_duplicate_sequence_id():
    pool = BlockPool(10, 16)
    pool.allocate("a", 20)
    with pytest.raises(ValueError):
        pool.allocate("a", 20)


def test_the_paged_pool_survives_a_load_that_contiguous_reservation_cannot():
    """Тот же бюджет блоков: непрерывная резервация тянет 3 запроса, блочная — 19."""
    contiguous_capacity = POOL_BLOCKS // blocks_for(8192, KV_BLOCK_SIZE)
    pool = BlockPool(POOL_BLOCKS, KV_BLOCK_SIZE)
    admitted = 0
    while True:
        try:
            pool.allocate(admitted, 1500)
        except OutOfKVBlocks:
            break
        admitted += 1
    assert contiguous_capacity == 3
    assert admitted == 19


# --------------------------------------------------------------- chunk_plan
def test_chunk_plan_slices_a_long_prompt():
    assert chunk_plan(1200, 512) == [512, 512, 176]


def test_chunk_plan_without_a_chunk_size_is_one_piece():
    assert chunk_plan(1200, None) == [1200]


def test_chunk_plan_never_changes_the_amount_of_work():
    """Куски перераспределяют prefill по шагам, но не добавляют и не убавляют его."""
    for size in (None, 128, 512, 4096):
        assert sum(chunk_plan(8192, size)) == 8192


# --------------------------------------------------------------- percentile
def test_percentile_takes_the_nearest_rank():
    assert percentile([1, 2, 3, 4], 50) == 2


def test_p99_ignores_a_single_outlier_that_p100_shows():
    values = [10] * 99 + [1000]
    assert percentile(values, 99) == 10
    assert percentile(values, 100) == 1000


def test_percentile_refuses_an_empty_sample():
    with pytest.raises(ValueError):
        percentile([], 99)


# ------------------------------------------- static vs continuous batching
def test_both_schedulers_produce_the_same_real_tokens():
    """Сравнение честное: работа одна и та же, отличается только планировщик."""
    static = schedule_static(MIXED_WORKLOAD, 8)
    continuous = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    assert static["output_tokens"] == REAL_OUTPUT_TOKENS
    assert continuous["output_tokens"] == REAL_OUTPUT_TOKENS


def test_continuous_batching_beats_static_on_throughput():
    static = schedule_static(MIXED_WORKLOAD, 8)
    continuous = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    assert continuous["throughput"] > static["throughput"]


def test_continuous_batching_beats_static_on_the_ttft_tail():
    """Хвост TTFT — то, ради чего всё затевалось: не ждать сбора и слива батча."""
    static = schedule_static(MIXED_WORKLOAD, 8)
    continuous = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    assert continuous["ttft_p99"] < static["ttft_p99"]


def test_continuous_batching_beats_static_on_the_end_to_end_tail():
    static = schedule_static(MIXED_WORKLOAD, 8)
    continuous = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    assert continuous["e2e_p99"] < static["e2e_p99"]


def test_static_burns_wall_clock_on_padding():
    """Те же токены за большее время — разница и есть оплаченный паддинг."""
    static = schedule_static(MIXED_WORKLOAD, 8)
    continuous = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    assert static["makespan"] > continuous["makespan"]


# --------------------------------------------------------- chunked prefill
def test_chunked_prefill_cuts_the_worst_inter_token_stall():
    """Промпт на 8192 токена перестаёт замораживать decode всем остальным."""
    plain = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    chunked = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS, CHUNK_SIZE)
    assert chunked["itl_max"] < plain["itl_max"] / 4


def test_smaller_chunks_cut_the_stall_further():
    coarse = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS, 512)
    fine = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS, 128)
    assert fine["itl_max"] < coarse["itl_max"]


def test_chunked_prefill_does_not_buy_throughput_by_itself():
    """Урок говорит прямо: chunked prefill лечит хвост, а не пропускную способность."""
    plain = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    chunked = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS, CHUNK_SIZE)
    assert chunked["throughput"] <= plain["throughput"]
    assert chunked["output_tokens"] == plain["output_tokens"]


def test_a_tighter_kv_budget_slows_the_whole_run_down():
    """Бюджет KV-блоков и есть предел параллелизма — не HBM и не размер батча."""
    roomy = schedule_continuous(MIXED_WORKLOAD, POOL_BLOCKS)
    tight = schedule_continuous(MIXED_WORKLOAD, 600)
    assert tight["makespan"] > roomy["makespan"]
    assert tight["output_tokens"] == roomy["output_tokens"]


def test_a_request_that_cannot_fit_the_pool_at_all_is_refused():
    with pytest.raises(OutOfKVBlocks):
        schedule_continuous(MIXED_WORKLOAD, 100)


def test_admitted_requests_reserve_the_full_kv_horizon():
    """Три промпта не должны пройти admission, а затем словить OOM в decode."""
    requests = [(0.0, 16, 2)] * 3
    report = schedule_continuous(requests, total_blocks=4, block_size=16)
    assert report["output_tokens"] == 6


def test_static_and_continuous_ttft_end_at_the_first_generated_token():
    request = [(0.0, 16, 2)]
    static = schedule_static(request, batch_size=1)
    continuous = schedule_continuous(request, total_blocks=2, block_size=16)
    assert static["ttft_mean"] == APPROX(continuous["ttft_mean"])
