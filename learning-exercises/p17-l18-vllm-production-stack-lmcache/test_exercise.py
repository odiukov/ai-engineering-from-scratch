"""Тесты к уроку «Многоуровневый KV-кэш: GPU, CPU, диск и цена попадания». Правь exercise.py."""

import pytest

from exercise import (
    TIERS,
    CacheError,
    cache_lookup,
    cache_put,
    effective_hit_ms,
    make_cache,
    recompute_ms,
    restore_ms,
    run_workload,
    serve_request,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, rel=1e-9)


def where(cache):
    """Плоская карта «уровень -> ключи по возрастанию давности»."""
    return {tier: list(cache["tiers"][tier]) for tier in TIERS}


def two_passes(docs, tokens):
    """Один и тот же набор префиксов дважды подряд."""
    return [(d, tokens) for d in docs] * 2


DOCS = [f"doc{i}" for i in range(10)]


# --------------------------------------------------------------- recompute_ms
def test_recomputing_a_four_k_prefix():
    assert recompute_ms(4000) == APPROX(500.0)


def test_recompute_is_linear_in_context_length():
    assert recompute_ms(8000) == APPROX(2 * recompute_ms(4000))


def test_negative_context_is_rejected():
    with pytest.raises(CacheError):
        recompute_ms(-1)


# ----------------------------------------------------------------- restore_ms
def test_kv_already_in_hbm_costs_nothing_to_restore():
    assert restore_ms(4000, "gpu") == APPROX(0.0)


def test_restoring_from_cpu_dram():
    """0.5 мс на обращение плюс 655.36 МБ по 50 ГБ/с."""
    assert restore_ms(4000, "cpu") == APPROX(13.6072)


def test_restoring_from_disk():
    assert restore_ms(4000, "disk") == ROUGH(40.0 + 655_360_000 / 1.5e9 * 1000)


def test_the_disk_hop_dominates_short_contexts():
    """Сорок миллисекунд хопа до Ceph — дороже, чем целый подъём 4К с CPU."""
    assert restore_ms(10, "disk") > restore_ms(4000, "cpu")


def test_unknown_tier_is_rejected():
    with pytest.raises(CacheError):
        restore_ms(4000, "tape")


# ------------------------------------------------------------ effective_hit_ms
def test_a_long_context_is_worth_pulling_off_disk():
    assert effective_hit_ms(4000, "disk") == ROUGH(restore_ms(4000, "disk"))
    assert effective_hit_ms(4000, "disk") < recompute_ms(4000)


def test_a_short_context_is_cheaper_to_recompute_than_to_pull_off_disk():
    """Попадание на диске есть, выигрыша нет — движок считает заново."""
    assert effective_hit_ms(500, "disk") == APPROX(recompute_ms(500))


def test_the_disk_crossover_reflects_the_full_kv_geometry():
    """40 / (0.125 - 0.109227) = 2535.93: с 2536-го токена диск обгоняет."""
    assert effective_hit_ms(2535, "disk") == APPROX(recompute_ms(2535))
    assert effective_hit_ms(2536, "disk") < recompute_ms(2536)


def test_cpu_dram_beats_recompute_at_any_useful_length():
    for tokens in (100, 1000, 32_000):
        assert effective_hit_ms(tokens, "cpu") == ROUGH(restore_ms(tokens, "cpu"))


# ----------------------------------------------------------------- make_cache
def test_a_fresh_cache_holds_nothing():
    assert where(make_cache(1000, 1000, 1000)) == {"gpu": [], "cpu": [], "disk": []}


def test_a_tier_can_be_switched_off_with_zero_capacity():
    """Кэш из одного GPU-уровня — это обычный vLLM без LMCache."""
    cache = cache_put(make_cache(4000, 0, 0), "a", 1000)
    assert where(cache) == {"gpu": ["a"], "cpu": [], "disk": []}


def test_negative_capacity_is_rejected():
    with pytest.raises(CacheError):
        make_cache(1000, -1, 1000)


# --------------------------------------------------------------- cache_lookup
def test_lookup_of_a_missing_prefix_is_none():
    assert cache_lookup(make_cache(1000, 1000, 1000), "nope") is None


def test_lookup_names_the_tier():
    cache = cache_put(make_cache(0, 0, 5000), "a", 1000)
    assert cache_lookup(cache, "a") == "disk"


def test_lookup_does_not_disturb_the_lru_order():
    """Роутер спрашивает «где лежит» до маршрутизации — сдвигать ничего нельзя."""
    cache = cache_put(cache_put(make_cache(2000, 2000, 0), "a", 1000), "b", 1000)
    cache_lookup(cache, "a")
    assert where(cache_put(cache, "c", 1000))["gpu"] == ["b", "c"]


# ------------------------------------------------------------------ cache_put
def test_a_new_prefix_lands_in_hbm():
    assert where(cache_put(make_cache(2000, 0, 0), "a", 1000))["gpu"] == ["a"]


def test_a_prefix_skips_tiers_that_are_switched_off():
    assert where(cache_put(make_cache(0, 0, 5000), "a", 1000))["disk"] == ["a"]


def test_eviction_cascades_all_the_way_down():
    """Ловушка: вытолкнутое с GPU не пропадает, оно ложится на CPU и толкает дальше."""
    cache = make_cache(2000, 2000, 2000)
    for key in "ABCDE":
        cache = cache_put(cache, key, 1000)
    assert where(cache) == {"gpu": ["D", "E"], "cpu": ["B", "C"], "disk": ["A"]}


def test_an_oversized_entry_does_not_wash_out_the_tier():
    """Ловушка: запись длиннее ёмкости уровня не влезет туда и пустой."""
    cache = cache_put(cache_put(make_cache(2000, 3000, 10_000), "a", 1000), "b", 1000)
    cache = cache_put(cache, "huge", 5000)
    assert where(cache) == {"gpu": ["a", "b"], "cpu": [], "disk": ["huge"]}


def test_an_entry_too_big_for_every_tier_is_dropped():
    assert cache_lookup(cache_put(make_cache(100, 100, 100), "huge", 9999), "huge") is None


def test_writing_the_same_key_twice_promotes_instead_of_duplicating():
    cache = cache_put(make_cache(0, 0, 5000), "a", 1000)
    cache = cache_put(cache, "a", 1000)
    assert where(cache) == {"gpu": [], "cpu": [], "disk": ["a"]}


def test_eviction_is_lru_and_not_fifo():
    """Обращение к 'a' делает её свежей, поэтому вылетает 'b', а не 'a'."""
    cache = cache_put(cache_put(make_cache(2000, 2000, 0), "a", 1000), "b", 1000)
    cache, _ = serve_request(cache, "a", 1000)
    cache = cache_put(cache, "c", 1000)
    assert cache_lookup(cache, "a") == "gpu"
    assert cache_lookup(cache, "b") == "cpu"


def test_an_empty_entry_is_rejected():
    with pytest.raises(CacheError):
        cache_put(make_cache(1000, 0, 0), "a", 0)


# -------------------------------------------------------------- serve_request
def test_a_miss_costs_a_full_recompute():
    _, report = serve_request(make_cache(100_000, 0, 0), "a", 4000)
    assert report["hit"] is False
    assert report["cost_ms"] == APPROX(recompute_ms(4000))
    assert report["saved_ms"] == APPROX(0.0)


def test_a_miss_still_warms_the_cache():
    cache, _ = serve_request(make_cache(100_000, 0, 0), "a", 4000)
    assert cache_lookup(cache, "a") == "gpu"


def test_a_hit_in_hbm_is_free():
    cache, _ = serve_request(make_cache(100_000, 0, 0), "a", 4000)
    _, report = serve_request(cache, "a", 4000)
    assert report["tier"] == "gpu"
    assert report["cost_ms"] == APPROX(0.0)
    assert report["saved_ms"] == APPROX(500.0)


def test_a_hit_from_disk_promotes_the_prefix_back_to_hbm():
    cache = cache_put(make_cache(100_000, 0, 100_000), "a", 4000)
    cache["tiers"]["gpu"].pop("a", None)
    cache["tiers"]["disk"]["a"] = 4000
    cache, report = serve_request(cache, "a", 4000)
    assert report["tier"] == "disk"
    assert cache_lookup(cache, "a") == "gpu"


def test_a_disk_hit_on_a_short_prefix_saves_nothing():
    """Попадание есть, экономии ноль: поднять дороже, чем посчитать заново."""
    cache, _ = serve_request(make_cache(0, 0, 100_000), "a", 500)
    _, report = serve_request(cache, "a", 500)
    assert report["hit"] is True
    assert report["saved_ms"] == APPROX(0.0)


# --------------------------------------------------------------- run_workload
def test_while_the_working_set_fits_hbm_the_lower_tiers_stay_idle():
    _, report = run_workload(make_cache(100_000, 500_000, 500_000),
                             two_passes(DOCS, 4000))
    assert report["hits_by_tier"] == {"gpu": 10, "cpu": 0, "disk": 0}
    assert report["saved_pct"] == APPROX(50.0)


def test_adding_tiers_to_a_roomy_cache_changes_nothing():
    """Малое давление на HBM — включать LMCache не за чем."""
    workload = two_passes(DOCS, 4000)
    _, plain = run_workload(make_cache(100_000, 0, 0), workload)
    _, tiered = run_workload(make_cache(100_000, 500_000, 500_000), workload)
    assert plain["saved_ms"] == APPROX(tiered["saved_ms"])


def test_a_cramped_hbm_alone_saves_nothing_on_the_second_pass():
    """Рабочее множество не влезает: каждый запрос вытесняет следующего."""
    _, report = run_workload(make_cache(8000, 0, 0), two_passes(DOCS, 4000))
    assert report["misses"] == 20
    assert report["saved_ms"] == APPROX(0.0)


def test_a_cpu_tier_turns_the_same_thrashing_into_savings():
    """Вот здесь LMCache и окупается: KV перестал влезать в HBM."""
    workload = two_passes(DOCS, 4000)
    _, without = run_workload(make_cache(8000, 0, 0), workload)
    _, with_cpu = run_workload(make_cache(8000, 500_000, 0), workload)
    assert with_cpu["hits_by_tier"]["cpu"] > 0
    assert with_cpu["saved_ms"] > without["saved_ms"]
    assert with_cpu["saved_pct"] > 45.0


def test_short_prefixes_lift_the_hit_rate_without_lifting_the_savings():
    """Диск даёт попадания, а не миллисекунды: вот для чего wasted_hits."""
    _, report = run_workload(make_cache(0, 0, 500_000), two_passes(DOCS, 500))
    assert report["hit_rate"] == APPROX(0.5)
    assert report["wasted_hits"] == 10
    assert report["saved_ms"] == APPROX(0.0)


def test_a_context_longer_than_the_disk_tier_never_gets_cached():
    """Верхняя граница пользы: то, что не влезло, пересчитывается всегда."""
    _, report = run_workload(make_cache(1000, 1000, 10_000), two_passes(DOCS, 20_000))
    assert report["hit_rate"] == APPROX(0.0)
    assert report["saved_pct"] == APPROX(0.0)


def test_an_empty_workload_does_not_divide_by_zero():
    _, report = run_workload(make_cache(0, 0, 0), [])
    assert report["saved_pct"] == APPROX(0.0)
    assert report["hit_rate"] == APPROX(0.0)
