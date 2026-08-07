"""Тесты к уроку «SGLang и RadixAttention». Правь exercise.py."""

import pytest

from exercise import (
    CACHE_CAPACITY,
    CANONICAL_ORDER,
    MIXED_WORKLOAD,
    ORDERED_WORKLOAD,
    PARTS,
    QUESTIONS,
    SCRAMBLED_ORDER,
    SCRAMBLED_WORKLOAD,
    CacheTooSmall,
    RadixCache,
    cache_aware_order,
    common_prefix_len,
    prefill_speedup,
    render_prompt,
    run_workload,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SYSTEM = tuple(range(10))          # короткий «системный промпт» для тестов дерева
ROOMY = 100000                     # кэш, в который влезает всё: вытеснения нет


# ------------------------------------------------------- common_prefix_len
def test_common_prefix_len_stops_at_the_first_mismatch():
    assert common_prefix_len((1, 2, 3), (1, 2, 9)) == 2


def test_a_mismatch_in_the_very_first_token_wipes_the_prefix_out():
    """Сколько бы одинакового ни шло дальше, дерево туда уже не дойдёт."""
    assert common_prefix_len((9, 2, 3, 4), (1, 2, 3, 4)) == 0


# ------------------------------------------------------------ render_prompt
def test_render_prompt_concatenates_components_in_the_given_order():
    assert render_prompt(("system", "tools"), {"system": (1, 2), "tools": (3,)}) == (1, 2, 3)


def test_the_same_component_order_shares_the_whole_prefix():
    a = render_prompt(CANONICAL_ORDER, PARTS)
    b = render_prompt(CANONICAL_ORDER, PARTS)
    assert common_prefix_len(a, b) == len(a) == 400


def test_swapping_two_components_cuts_the_shared_prefix_to_the_first_of_them():
    """Ловушка урока: человек видит тот же промпт, дерево — другую последовательность."""
    canonical = render_prompt(CANONICAL_ORDER, PARTS)
    scrambled = render_prompt(SCRAMBLED_ORDER, PARTS)
    assert common_prefix_len(canonical, scrambled) == len(PARTS["system"])
    lost = len(PARTS["tools"]) + len(PARTS["context_a"])
    assert common_prefix_len(canonical, scrambled) == len(canonical) - lost


def test_render_prompt_refuses_an_unknown_component():
    with pytest.raises(KeyError):
        render_prompt(("system", "nope"), PARTS)


def test_render_prompt_refuses_an_empty_order():
    with pytest.raises(ValueError):
        render_prompt((), PARTS)


# ---------------------------------------------------------- prefill_speedup
def test_no_reuse_means_no_speedup():
    assert prefill_speedup(0.0) == APPROX(1.0)


def test_the_lesson_six_point_four_x_needs_a_hit_rate_above_eighty_percent():
    assert prefill_speedup(0.844) == pytest.approx(6.41, abs=0.01)


def test_the_last_percents_of_hit_rate_are_worth_the_most():
    """Зависимость нелинейная — поэтому дисциплина промпта окупается в конце."""
    late = prefill_speedup(0.95) - prefill_speedup(0.90)
    early = prefill_speedup(0.50) - prefill_speedup(0.45)
    assert late > early


def test_prefill_speedup_refuses_a_hit_rate_of_one():
    with pytest.raises(ValueError):
        prefill_speedup(1.0)


# ---------------------------------------------------------------- RadixCache
def test_a_new_cache_holds_nothing_and_matches_nothing():
    cache = RadixCache(ROOMY)
    assert cache.used_tokens() == 0
    assert cache.match(SYSTEM) == 0


def test_the_first_prompt_reuses_nothing_and_lands_whole():
    cache = RadixCache(ROOMY)
    assert cache.insert(SYSTEM + (100, 101), 1.0) == {"reused": 0, "new": 12}
    assert cache.used_tokens() == 12


def test_a_shared_system_prompt_saves_exactly_the_common_prefix():
    """Ровно длина общего префикса — ни токеном больше, ни токеном меньше."""
    cache = RadixCache(ROOMY)
    a = render_prompt(CANONICAL_ORDER, PARTS) + QUESTIONS[0]
    b = render_prompt(CANONICAL_ORDER, PARTS) + QUESTIONS[1]
    cache.insert(a, 1.0)
    stats = cache.insert(b, 2.0)
    assert stats["reused"] == common_prefix_len(a, b) == 400
    assert stats["new"] == len(QUESTIONS[1])


def test_a_prompt_that_differs_in_the_first_token_reuses_nothing():
    cache = RadixCache(ROOMY)
    cache.insert(SYSTEM + (100, 101), 1.0)
    assert cache.insert((999,) + SYSTEM[1:] + (100, 101), 2.0)["reused"] == 0


def test_reinserting_the_same_prompt_costs_no_new_tokens():
    cache = RadixCache(ROOMY)
    prompt = SYSTEM + (100, 101)
    cache.insert(prompt, 1.0)
    assert cache.insert(prompt, 2.0) == {"reused": 12, "new": 0}
    assert cache.used_tokens() == 12


def test_match_counts_a_partial_segment_too():
    """Совпадение может оборваться в СЕРЕДИНЕ отрезка узла — это тоже экономия."""
    cache = RadixCache(ROOMY)
    cache.insert((1, 2, 3, 4, 5), 1.0)
    assert cache.match((1, 2, 3)) == 3
    assert cache.match((1, 2, 9)) == 2
    assert cache.match((9, 2, 3)) == 0


def test_match_leaves_the_cache_untouched():
    cache = RadixCache(ROOMY)
    cache.insert((1, 2, 3, 4, 5), 1.0)
    cache.match((1, 2, 9))
    assert cache.used_tokens() == 5


def test_splitting_a_node_does_not_duplicate_its_tokens():
    """Два промпта с общим началом занимают объединение, а не сумму."""
    cache = RadixCache(ROOMY)
    cache.insert(SYSTEM + (100, 101, 102), 1.0)
    cache.insert(SYSTEM + (200, 201, 202), 2.0)
    assert cache.used_tokens() == 16
    assert cache.match(SYSTEM + (100, 101, 102)) == 13
    assert cache.match(SYSTEM + (200, 201, 202)) == 13


def test_eviction_never_removes_a_node_that_still_has_children():
    """Снос узла с потомками оборвал бы путь к их KV — горячая ветка обязана уцелеть."""
    cache = RadixCache(18)
    cache.insert(SYSTEM + (100, 101, 102), 1.0)
    cache.insert(SYSTEM + (200, 201, 202), 2.0)
    assert cache.insert(SYSTEM + (300, 301, 302), 3.0) == {"reused": 10, "new": 3}
    assert cache.match(SYSTEM) == 10                      # родитель на месте
    assert cache.match(SYSTEM + (100, 101, 102)) == 10    # самый старый лист ушёл
    assert cache.match(SYSTEM + (200, 201, 202)) == 13    # свежий сосед остался


def test_a_branch_is_peeled_leaf_by_leaf_from_the_bottom_up():
    cache = RadixCache(ROOMY)
    cache.insert(SYSTEM + (100, 101, 102), 1.0)
    cache.insert(SYSTEM + (200, 201, 202), 5.0)
    assert cache.evict(3, 9.0) == 3
    assert cache.match(SYSTEM) == 10          # ещё есть потомок — узел живёт
    assert cache.evict(3, 9.0) == 3
    assert cache.match(SYSTEM) == 10          # потомков нет, но за этот виток не тронут
    assert cache.evict(1, 9.0) == 10          # только теперь он сам стал листом
    assert cache.match(SYSTEM) == 0


def test_eviction_spares_the_branch_the_current_request_just_touched():
    cache = RadixCache(ROOMY)
    cache.insert(SYSTEM + (100, 101, 102), 4.0)
    with pytest.raises(CacheTooSmall):
        cache.evict(1, 4.0)


def test_a_prompt_longer_than_the_whole_cache_is_refused():
    cache = RadixCache(5)
    with pytest.raises(CacheTooSmall):
        cache.insert(SYSTEM, 1.0)


def test_insert_refuses_an_empty_prompt():
    with pytest.raises(ValueError):
        RadixCache(ROOMY).insert((), 1.0)


# ------------------------------------------------------- cache_aware_order
def test_an_empty_cache_leaves_the_queue_in_arrival_order():
    cache = RadixCache(ROOMY)
    assert cache_aware_order(cache, MIXED_WORKLOAD, [0, 1, 2, 3]) == [0, 1, 2, 3]


def test_the_longest_shared_prefix_is_served_first():
    """Тот самый обход в глубину: горячая ветка не успевает остыть."""
    cache = RadixCache(ROOMY)
    cache.insert(MIXED_WORKLOAD[2], 1.0)
    assert cache_aware_order(cache, MIXED_WORKLOAD, [1, 3, 4])[0] == 4
    assert cache.match(MIXED_WORKLOAD[4]) > cache.match(MIXED_WORKLOAD[1])


def test_equal_matches_keep_the_arrival_order():
    cache = RadixCache(ROOMY)
    cache.insert(MIXED_WORKLOAD[0], 1.0)
    assert cache_aware_order(cache, MIXED_WORKLOAD, [5, 3, 1]) == [1, 3, 5]


# ------------------------------------------------------------- run_workload
def test_a_shared_rag_prefix_gives_the_speedup_the_lesson_advertises():
    result = run_workload(ORDERED_WORKLOAD, CACHE_CAPACITY, "fcfs")
    assert result["hit_rate"] == APPROX(2800 / 3360)
    assert result["speedup"] == APPROX(6.0)


def test_reuse_moves_tokens_from_prefill_to_free_but_never_invents_them():
    result = run_workload(ORDERED_WORKLOAD, CACHE_CAPACITY, "fcfs")
    assert result["prompt_tokens"] == sum(len(p) for p in ORDERED_WORKLOAD)
    assert result["reused_tokens"] + result["prefill_tokens"] == result["prompt_tokens"]
    assert result["speedup"] == APPROX(prefill_speedup(result["hit_rate"]))


def test_scrambled_component_order_halves_the_hit_rate():
    """Тот же трафик, те же компоненты — только порядок сборки другой."""
    ordered = run_workload(ORDERED_WORKLOAD, CACHE_CAPACITY, "fcfs")
    scrambled = run_workload(SCRAMBLED_WORKLOAD, CACHE_CAPACITY, "fcfs")
    assert scrambled["hit_rate"] == APPROX(ordered["hit_rate"] / 2)
    assert scrambled["prefill_tokens"] > ordered["prefill_tokens"] * 3


def test_cache_aware_beats_fcfs_when_two_branches_do_not_fit_together():
    fcfs = run_workload(MIXED_WORKLOAD, CACHE_CAPACITY, "fcfs")
    aware = run_workload(MIXED_WORKLOAD, CACHE_CAPACITY, "cache_aware")
    assert aware["hit_rate"] > fcfs["hit_rate"]
    assert aware["prefill_tokens"] < fcfs["prefill_tokens"]


def test_the_scheduler_only_matters_because_the_cache_is_finite():
    """В кэше, куда влезает всё, обе политики дают ровно одно и то же."""
    fcfs = run_workload(MIXED_WORKLOAD, 100000, "fcfs")
    aware = run_workload(MIXED_WORKLOAD, 100000, "cache_aware")
    assert fcfs["hit_rate"] == APPROX(aware["hit_rate"])
    assert fcfs["order"] != aware["order"]


def test_cache_aware_groups_requests_of_one_branch_together():
    aware = run_workload(MIXED_WORKLOAD, CACHE_CAPACITY, "cache_aware")
    assert aware["order"] == [0, 2, 4, 6, 1, 3, 5, 7]


def test_every_request_is_served_exactly_once_under_both_policies():
    for policy in ("fcfs", "cache_aware"):
        order = run_workload(MIXED_WORKLOAD, CACHE_CAPACITY, policy)["order"]
        assert sorted(order) == list(range(len(MIXED_WORKLOAD)))


def test_run_workload_refuses_an_unknown_policy():
    with pytest.raises(ValueError):
        run_workload(MIXED_WORKLOAD, CACHE_CAPACITY, "lru")
