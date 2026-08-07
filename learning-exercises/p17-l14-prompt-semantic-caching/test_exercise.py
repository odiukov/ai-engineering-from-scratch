"""Тесты к уроку «Кэш промптов и семантический кэш: два слоя и их цена». Правь exercise.py."""

import math

import pytest

from exercise import (
    PRICE_CACHED_READ,
    PRICE_INPUT,
    WRITE_MULTIPLIER,
    CacheError,
    cache_stats,
    common_prefix_tokens,
    cosine,
    l2_request_cost,
    nearest_entry,
    parallel_wave_cost,
    run_semantic_cache,
    semantic_lookup,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def vec(degrees):
    """Единичный вектор под заданным углом: косинус между ними = cos разности."""
    rad = math.radians(degrees)
    return [math.cos(rad), math.sin(rad)]


# Три вопроса о возврате денег. Соседние по смыслу, ответы разные.
Q_SUB = {"vector": vec(0), "answer": "вернуть за подписку"}
Q_GOODS = {"vector": vec(10), "answer": "вернуть за товар"}      # cos = 0.9848
Q_SHIP = {"vector": vec(40), "answer": "сроки доставки"}         # cos = 0.766
Q_SUB_AGAIN = {"vector": vec(0), "answer": "вернуть за подписку"}

WORKLOAD = [Q_SUB, Q_GOODS, Q_SUB_AGAIN, Q_SHIP]

# Промпт из токенов: статичная инструкция плюс изменчивый хвост.
STATIC = ["Ты", "поддержка", "магазина", ".", "Отвечай", "коротко", "."]


# ------------------------------------------------------------------- cosine
def test_perpendicular_vectors_have_zero_similarity():
    assert cosine([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    assert cosine([3, 4], [6, 8]) == APPROX(1.0)


def test_opposite_vectors_are_minus_one():
    assert cosine([1, 0], [-1, 0]) == APPROX(-1.0)


def test_ten_degrees_apart_is_still_very_similar():
    assert cosine(vec(0), vec(10)) == pytest.approx(0.9848, abs=1e-4)


def test_zero_vector_has_no_direction():
    """Деление на ноль отравило бы порог тихо, поэтому это CacheError."""
    with pytest.raises(CacheError):
        cosine([0, 0], [1, 0])


def test_vectors_from_different_models_are_rejected():
    with pytest.raises(CacheError):
        cosine([1, 0], [1, 0, 0])


# ------------------------------------------------------------ nearest_entry
def test_empty_cache_returns_nothing_and_minus_one():
    assert nearest_entry([], vec(0)) == (None, -1.0)


def test_nearest_entry_finds_the_closest_vector():
    entries = [dict(Q_SHIP), dict(Q_SUB)]
    entry, sim = nearest_entry(entries, vec(2))
    assert entry["answer"] == "вернуть за подписку"
    assert sim == pytest.approx(math.cos(math.radians(2)), abs=1e-9)


def test_empty_cache_similarity_is_below_any_threshold():
    """-1.0, а не 0.0: ноль это «перпендикулярно», осмысленная близость."""
    assert nearest_entry([], vec(0))[1] < 0.0


def test_ties_go_to_the_earlier_entry():
    first = {"vector": vec(0), "answer": "первый"}
    second = {"vector": vec(0), "answer": "второй"}
    assert nearest_entry([first, second], vec(0))[0]["answer"] == "первый"


# ---------------------------------------------------------- semantic_lookup
def test_exact_repeat_is_a_hit():
    entry, sim = semantic_lookup([dict(Q_SUB)], vec(0), 0.95)
    assert entry is not None
    assert sim == APPROX(1.0)


def test_distant_query_is_a_miss_but_similarity_is_still_reported():
    entry, sim = semantic_lookup([dict(Q_SUB)], vec(40), 0.95)
    assert entry is None
    assert sim == pytest.approx(0.766, abs=1e-3)


def test_threshold_is_inclusive():
    entries = [dict(Q_SUB)]
    sim = cosine(Q_SUB["vector"], vec(10))
    assert semantic_lookup(entries, vec(10), sim)[0] is not None


def test_high_threshold_turns_a_near_neighbour_into_a_miss():
    assert semantic_lookup([dict(Q_SUB)], vec(10), 0.99)[0] is None


# -------------------------------------------------------- run_semantic_cache
def test_first_query_always_goes_to_the_llm():
    records = run_semantic_cache(WORKLOAD, 0.95)
    assert records[0]["served"] == "llm"


def test_low_threshold_serves_a_neighbour_answer():
    """Порог 0.95: вопрос про товар получает ответ про подписку."""
    records = run_semantic_cache(WORKLOAD, 0.95)
    assert records[1]["served"] == "cache"
    assert records[1]["answer"] == "вернуть за подписку"
    assert records[1]["correct"] is False


def test_high_threshold_keeps_the_neighbour_out():
    records = run_semantic_cache(WORKLOAD, 0.99)
    assert records[1]["served"] == "llm"
    assert records[1]["correct"] is True


def test_exact_repeat_is_served_from_cache_correctly():
    records = run_semantic_cache(WORKLOAD, 0.99)
    assert records[2]["served"] == "cache"
    assert records[2]["correct"] is True


def test_only_misses_are_written_to_the_cache():
    """Четыре запроса, порог 0.99: три промаха, значит три записи в кэше.

    Если писать и попадания тоже, четвёртый запрос сравнивался бы с четырьмя
    записями и близость к дублю vec(0) была бы найдена дважды.
    """
    records = run_semantic_cache(WORKLOAD, 0.99)
    assert sum(1 for r in records if r["served"] == "llm") == 3


def test_every_record_carries_the_similarity_it_was_judged_by():
    records = run_semantic_cache(WORKLOAD, 0.95)
    assert records[0]["similarity"] == -1.0
    assert records[1]["similarity"] == pytest.approx(0.9848, abs=1e-4)


# --------------------------------------------------------------- cache_stats
def test_empty_run_has_zero_hit_rate_and_does_not_divide_by_zero():
    stats = cache_stats([])
    assert stats["hit_rate"] == APPROX(0.0)
    assert stats["false_hit_rate"] == APPROX(0.0)


def test_lower_threshold_raises_the_hit_rate():
    loose = cache_stats(run_semantic_cache(WORKLOAD, 0.95))
    strict = cache_stats(run_semantic_cache(WORKLOAD, 0.99))
    assert loose["hit_rate"] > strict["hit_rate"]


def test_the_extra_hits_of_a_low_threshold_are_wrong_answers():
    """Главное свойство урока: дешевле — значит иногда неверно."""
    loose = cache_stats(run_semantic_cache(WORKLOAD, 0.95))
    strict = cache_stats(run_semantic_cache(WORKLOAD, 0.99))
    assert loose["false_hits"] == 1
    assert strict["false_hits"] == 0


def test_false_hit_rate_is_measured_against_hits_not_requests():
    """Половина попаданий врёт, хотя от всех запросов это только четверть."""
    stats = cache_stats(run_semantic_cache(WORKLOAD, 0.95))
    assert stats["hits"] == 2
    assert stats["false_hit_rate"] == APPROX(0.5)
    assert stats["false_hits"] / stats["total"] == APPROX(0.25)


def test_llm_calls_equal_misses():
    stats = cache_stats(run_semantic_cache(WORKLOAD, 0.95))
    assert stats["llm_calls"] == stats["misses"]
    assert stats["hits"] + stats["misses"] == stats["total"]


# --------------------------------------------------- common_prefix_tokens
def test_identical_prompts_share_everything():
    assert common_prefix_tokens(STATIC, list(STATIC)) == len(STATIC)


def test_prefix_stops_at_the_first_difference():
    assert common_prefix_tokens(["Ты", "ассистент", "."], ["Ты", "ассистент", "!"]) == 2


def test_timestamp_in_front_destroys_the_shared_prefix():
    """Анти-паттерн динамического содержимого: время в начале обнуляет кэш."""
    a = ["14:32"] + STATIC
    b = ["14:33"] + STATIC
    assert common_prefix_tokens(a, b) == 0


def test_timestamp_at_the_end_keeps_the_prefix_cacheable():
    """Тот же текст, время перенесено за границу кэша — префикс целиком общий."""
    a = STATIC + ["14:32"]
    b = STATIC + ["14:33"]
    assert common_prefix_tokens(a, b) == len(STATIC)


def test_no_common_prefix_at_all():
    assert common_prefix_tokens(["a"], ["b"]) == 0


# --------------------------------------------------------- l2_request_cost
def test_cold_request_pays_the_write_premium():
    assert l2_request_cost(4000, 200, 200, "5min", False) == APPROX(0.0186)


def test_warm_request_pays_the_read_price():
    assert l2_request_cost(4000, 200, 200, "5min", True) == APPROX(0.0048)


def test_hour_ttl_write_is_more_expensive_than_five_minute_ttl():
    hour = l2_request_cost(4000, 0, 0, "1hr", False)
    five = l2_request_cost(4000, 0, 0, "5min", False)
    assert hour / five == APPROX(WRITE_MULTIPLIER["1hr"] / WRITE_MULTIPLIER["5min"])


def test_cached_read_is_ten_times_cheaper_than_fresh_input():
    """Только префикс, ничего больше: тёплый запрос ровно в 10 раз дешевле входа."""
    warm = l2_request_cost(4000, 0, 0, "5min", True)
    assert warm == APPROX(4000 / 1e6 * PRICE_CACHED_READ)
    assert 4000 / 1e6 * PRICE_INPUT / warm == APPROX(10.0)


def test_unknown_ttl_is_rejected():
    """Молчаливый множитель 1.0 занизил бы счёт ровно на премию за запись."""
    with pytest.raises(CacheError):
        l2_request_cost(4000, 0, 0, "1day", False)


def test_dynamic_tail_is_never_cached():
    warm_short = l2_request_cost(4000, 0, 0, "5min", True)
    warm_long = l2_request_cost(4000, 1000, 0, "5min", True)
    assert warm_long - warm_short == APPROX(1000 / 1e6 * PRICE_INPUT)


# ------------------------------------------------------ parallel_wave_cost
def test_parallel_wave_pays_the_write_premium_n_times():
    assert parallel_wave_cost(10, 4000, 50, 50, "1hr", False) == APPROX(0.249)


def test_serializing_the_first_call_collapses_the_bill():
    assert parallel_wave_cost(10, 4000, 50, 50, "1hr", True) == APPROX(0.0438)


def test_the_anti_pattern_costs_several_times_more():
    """Урок обещает 5-10x на длинном префиксе и коротком выходе — вот они."""
    bad = parallel_wave_cost(10, 4000, 50, 50, "1hr", False)
    good = parallel_wave_cost(10, 4000, 50, 50, "1hr", True)
    assert bad / good > 5.0


def test_a_wave_of_one_costs_the_same_either_way():
    bad = parallel_wave_cost(1, 4000, 50, 50, "5min", False)
    good = parallel_wave_cost(1, 4000, 50, 50, "5min", True)
    assert bad == APPROX(good)


def test_short_prefix_makes_serialization_pointless():
    """Нечего кэшировать — нечего и экономить: выигрыш почти исчезает."""
    bad = parallel_wave_cost(10, 20, 50, 500, "5min", False)
    good = parallel_wave_cost(10, 20, 50, 500, "5min", True)
    assert bad / good < 1.02


def test_empty_wave_is_a_caller_error():
    with pytest.raises(CacheError):
        parallel_wave_cost(0, 4000, 50, 50, "5min", True)
