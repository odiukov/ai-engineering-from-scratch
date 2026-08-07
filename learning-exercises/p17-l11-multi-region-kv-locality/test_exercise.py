"""Тесты к уроку «Мультирегиональный инференс и локальность KV-кэша». Правь exercise.py."""

import random

import pytest

from exercise import (
    CACHE_HIT_MS,
    CACHE_MISS_MS,
    REQUIRED_DR_FILES,
    NoEligibleReplicaError,
    UnknownRegionError,
    dr_manifest_gaps,
    expected_ttft_ms,
    percentile,
    prefix_key,
    route_cache_aware,
    route_round_robin,
    rtt_ms,
    simulate,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Две реплики в разных регионах — минимальная конфигурация, на которой
# видно, что «ближе» и «быстрее» это не одно и то же.
EAST = {"name": "east-0", "region": "us-east-1"}
EU = {"name": "eu-0", "region": "eu-west-1"}
TWO_REGIONS = [EAST, EU]

FLEET = [
    {"name": "east-0", "region": "us-east-1"},
    {"name": "east-1", "region": "us-east-1"},
    {"name": "eu-0", "region": "eu-west-1"},
    {"name": "eu-1", "region": "eu-west-1"},
]

# Четыре реплики в одном регионе: на них видно чистый эффект кэша,
# без примеси сетевой задержки.
EAST_FLEET = [{"name": f"east-{i}", "region": "us-east-1"} for i in range(4)]


def request(origin, tenant, tail=0):
    """Запрос арендатора tenant: у каждого свой системный промпт на 600 токенов."""
    return {"origin": origin, "tokens": [tenant] * 600 + [tail]}


def tenant_stream(origin, tenants, count, seed=0):
    """Поток запросов от случайных арендаторов — воспроизводимо, по seed."""
    rng = random.Random(seed)
    return [request(origin, rng.choice(tenants), i) for i in range(count)]


# -------------------------------------------------------------- prefix_key
def test_same_prefix_gives_the_same_key():
    assert prefix_key([1, 2, 3], 2) == prefix_key([1, 2, 9, 9], 2)


def test_different_prefix_gives_a_different_key():
    assert prefix_key([1, 2, 3], 2) != prefix_key([1, 5], 2)


def test_key_is_stable_across_calls():
    """Роутер обязан класть один и тот же префикс на одну и ту же реплику."""
    assert prefix_key([7, 7, 7], 3) == prefix_key([7, 7, 7], 3)


def test_short_prompt_is_hashed_whole():
    assert prefix_key([1, 2], 512) == prefix_key([1, 2], 2)


def test_zero_length_prefix_is_a_call_error():
    with pytest.raises(ValueError):
        prefix_key([1, 2, 3], 0)


# ------------------------------------------------------------------ rtt_ms
def test_same_region_costs_no_network_time():
    assert rtt_ms("us-east-1", "us-east-1") == APPROX(0.0)


def test_transatlantic_rtt_is_the_published_75_ms():
    assert rtt_ms("us-east-1", "eu-west-1") == APPROX(75.0)


def test_rtt_table_is_symmetric():
    assert rtt_ms("eu-west-1", "us-east-1") == rtt_ms("us-east-1", "eu-west-1")


def test_unknown_region_pair_is_not_guessed():
    with pytest.raises(UnknownRegionError):
        rtt_ms("us-east-1", "moon-base-1")


# --------------------------------------------------------- expected_ttft_ms
def test_local_cache_hit_is_the_fast_path():
    assert expected_ttft_ms(True, "us-east-1", "us-east-1") == APPROX(CACHE_HIT_MS)


def test_local_cache_miss_pays_full_prefill():
    assert expected_ttft_ms(False, "us-east-1", "us-east-1") == APPROX(CACHE_MISS_MS)


def test_a_distant_cache_hit_beats_a_local_miss():
    """Кэш экономит 720 мс, Атлантика стоит 75 — считать надо сумму."""
    far_hit = expected_ttft_ms(True, "us-east-1", "eu-west-1")
    local_miss = expected_ttft_ms(False, "us-east-1", "us-east-1")
    assert far_hit == APPROX(155.0)
    assert far_hit < local_miss


def test_even_the_apac_hop_stays_cheaper_than_a_cold_prefill():
    """220 мс RTT — это всё ещё меньше, чем 720 мс сэкономленного prefill."""
    assert expected_ttft_ms(True, "us-east-1", "ap-southeast-1") < CACHE_MISS_MS


def test_the_break_even_rtt_is_the_prefill_saving_itself():
    """Выше 720 мс RTT удалённый кэш уже не окупается."""
    slow = {("us-east-1", "mars-1"): 900.0}
    assert expected_ttft_ms(True, "us-east-1", "mars-1", slow) > CACHE_MISS_MS


# ------------------------------------------------------- route_round_robin
def test_round_robin_walks_the_fleet_in_order():
    assert [route_round_robin(i, FLEET) for i in range(5)] == [0, 1, 2, 3, 0]


def test_round_robin_without_replicas_has_nowhere_to_go():
    with pytest.raises(NoEligibleReplicaError):
        route_round_robin(0, [])


# ------------------------------------------------------- route_cache_aware
def test_cache_aware_prefers_the_replica_that_holds_the_prefix():
    caches = [[], ["abc"]]
    assert route_cache_aware("abc", "us-east-1", TWO_REGIONS, caches) == 1


def test_nearest_region_loses_to_a_distant_cache_hit():
    """Локальная реплика пуста, европейская держит префикс — летим в Европу."""
    caches = [[], ["abc"]]
    chosen = route_cache_aware("abc", "us-east-1", TWO_REGIONS, caches)
    assert TWO_REGIONS[chosen]["region"] == "eu-west-1"


def test_a_hit_too_far_away_loses_to_a_local_miss():
    """RTT 900 мс дороже, чем сэкономленные 720 мс prefill — остаёмся дома."""
    replicas = [{"name": "east-0", "region": "us-east-1"},
                {"name": "mars-0", "region": "mars-1"}]
    slow = {("us-east-1", "mars-1"): 900.0}
    caches = [[], ["abc"]]
    assert route_cache_aware("abc", "us-east-1", replicas, caches, slow) == 0


def test_without_any_cache_the_router_stays_local():
    caches = [[], []]
    assert route_cache_aware("abc", "us-east-1", TWO_REGIONS, caches) == 0


def test_residency_bound_keeps_the_request_inside_its_region():
    """GDPR сильнее TTFT: попадание в кэш за океаном не повод везти туда PHI."""
    caches = [[], ["abc"]]
    chosen = route_cache_aware("abc", "us-east-1", TWO_REGIONS, caches,
                               residency_bound=True)
    assert TWO_REGIONS[chosen]["region"] == "us-east-1"


def test_residency_bound_with_no_local_replica_is_an_error():
    with pytest.raises(NoEligibleReplicaError):
        route_cache_aware("abc", "ap-southeast-1", TWO_REGIONS, [[], []],
                          residency_bound=True)


# -------------------------------------------------------------- percentile
def test_percentile_uses_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_100_is_the_worst_observation():
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_percentile_of_an_empty_sample_is_undefined():
    with pytest.raises(ValueError):
        percentile([], 99)


# ---------------------------------------------------------------- simulate
def test_round_robin_wastes_the_cache_when_prefixes_outnumber_the_slots():
    """Восемь арендаторов, кэш на четыре — слепой роутер промахивается через раз."""
    reqs = tenant_stream("us-east-1", list(range(8)), 200)
    result = simulate(reqs, EAST_FLEET, "round_robin", cache_capacity=4)
    assert result["hit_rate"] < 0.7


def test_cache_aware_routing_keeps_each_tenant_on_its_own_replica():
    reqs = tenant_stream("us-east-1", list(range(8)), 200)
    rr = simulate(reqs, EAST_FLEET, "round_robin", cache_capacity=4)
    aware = simulate(reqs, EAST_FLEET, "cache_aware", cache_capacity=4)
    assert aware["hit_rate"] > rr["hit_rate"] + 0.2
    assert aware["mean_ttft"] < rr["mean_ttft"]


def test_simulation_is_deterministic():
    """Никакого множества с солёными хешами внутри: два прогона — одни числа."""
    reqs = tenant_stream("eu-west-1", list(range(7)), 50)
    assert simulate(reqs, FLEET, "cache_aware") == simulate(reqs, FLEET, "cache_aware")


def test_simulation_does_not_leak_state_into_the_replica_list():
    reqs = tenant_stream("us-east-1", list(range(4)), 20)
    simulate(reqs, FLEET, "cache_aware")
    assert FLEET == [
        {"name": "east-0", "region": "us-east-1"},
        {"name": "east-1", "region": "us-east-1"},
        {"name": "eu-0", "region": "eu-west-1"},
        {"name": "eu-1", "region": "eu-west-1"},
    ]


def test_regional_strategy_never_leaves_the_origin_region():
    reqs = tenant_stream("eu-west-1", list(range(3)), 30)
    assert simulate(reqs, FLEET, "cache_aware_regional")["cross_region"] == 0


def test_global_routing_buys_hit_rate_with_cross_region_traffic():
    """Глобальный роутер быстрее, но часть запросов уезжает за границу — и это счёт."""
    reqs = tenant_stream("us-east-1", list(range(3)), 10)
    reqs += tenant_stream("eu-west-1", list(range(3)), 10, seed=1)
    glob = simulate(reqs, FLEET, "cache_aware", cache_capacity=2)
    regional = simulate(reqs, FLEET, "cache_aware_regional", cache_capacity=2)
    assert glob["cross_region"] > regional["cross_region"]
    assert glob["mean_ttft"] < regional["mean_ttft"]


def test_a_tiny_cache_evicts_the_prefix_before_it_is_reused():
    """Ёмкость кэша меньше числа горячих префиксов — попадания исчезают."""
    reqs = tenant_stream("us-east-1", list(range(6)), 60)
    roomy = simulate(reqs, EAST_FLEET, "cache_aware", cache_capacity=8)
    tiny = simulate(reqs, EAST_FLEET, "cache_aware", cache_capacity=1)
    assert tiny["hit_rate"] < roomy["hit_rate"]


def test_p99_is_never_better_than_the_median():
    reqs = tenant_stream("us-east-1", list(range(5)), 50)
    result = simulate(reqs, FLEET, "cache_aware")
    assert result["p99_ttft"] >= result["p50_ttft"]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        simulate([request("us-east-1", 1)], FLEET, "least-connections")


# -------------------------------------------------------- dr_manifest_gaps
def test_weights_alone_are_not_a_backup():
    """Именно так выглядят те самые 32% провалов: веса есть, стартовать нечем."""
    gaps = dr_manifest_gaps(["model.safetensors"])
    assert "tokenizer.json" in gaps
    assert "quantize_config.json" in gaps


def test_a_complete_manifest_has_no_gaps():
    assert dr_manifest_gaps(REQUIRED_DR_FILES) == []


def test_extra_files_in_the_backup_do_not_hide_a_missing_one():
    backup = [f for f in REQUIRED_DR_FILES if f != "tokenizer.json"] + ["README.md"]
    assert dr_manifest_gaps(backup) == ["tokenizer.json"]
