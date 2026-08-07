"""Тесты к уроку «Кэширование промптов». Правь exercise.py."""

import pytest

from exercise import (
    MIN_CACHEABLE_TOKENS,
    READ_MULTIPLIER,
    WRITE_MULTIPLIER,
    break_even_reads,
    cache_friendly_layout,
    cache_lookup,
    common_prefix_len,
    request_cost,
    simulate_session,
    split_tokens,
)

APPROX = lambda x: pytest.approx(x, rel=1e-9)

# Учебные промпты короткие, поэтому порог кэширования в тестах опускаем до 4.
# В бою он равен MIN_CACHEABLE_TOKENS = 1024.
SMALL = 4
PROMPT = ["s1", "s2", "s3", "s4", "t1", "t2", "u1", "u2", "u3", "u4"]


# -------------------------------------------------------- common_prefix_len
def test_prefix_stops_at_the_first_difference():
    assert common_prefix_len(["a", "b", "c"], ["a", "b", "d"]) == 2


def test_prefix_of_zero_when_the_very_first_token_differs():
    """Даже если дальше всё совпадает — кэшу это безразлично."""
    assert common_prefix_len(["x", "b", "c"], ["a", "b", "c"]) == 0


def test_prefix_handles_different_lengths():
    assert common_prefix_len(["a", "b"], ["a", "b", "c", "d"]) == 2


def test_prefix_of_empty_is_zero():
    assert common_prefix_len([], ["a"]) == 0
    assert common_prefix_len(["a"], []) == 0


def test_prefix_is_not_a_similarity_measure():
    """Девяносто процентов общих токенов не в начале дают ровно ноль."""
    a = ["ts=10:00"] + [f"w{i}" for i in range(100)]
    b = ["ts=10:01"] + [f"w{i}" for i in range(100)]
    assert common_prefix_len(a, b) == 0


# ---------------------------------------------------- cache_friendly_layout
def test_layout_lifts_stable_sections_above_volatile_ones():
    sections = [
        ("user", ["q"], False),
        ("system", ["s1", "s2"], True),
        ("tools", ["t"], True),
    ]
    ordered, prefix = cache_friendly_layout(sections)
    assert [s[0] for s in ordered] == ["system", "tools", "user"]
    assert prefix == ["s1", "s2", "t"]


def test_layout_keeps_the_relative_order_inside_each_group():
    sections = [
        ("system", ["s"], True),
        ("fewshot", ["f"], True),
        ("history", ["h"], False),
        ("user", ["u"], False),
    ]
    ordered, _ = cache_friendly_layout(sections)
    assert [s[0] for s in ordered] == ["system", "fewshot", "history", "user"]


def test_layout_prefix_stops_at_the_first_volatile_section():
    sections = [("system", ["s"], True), ("clock", ["now"], False)]
    _, prefix = cache_friendly_layout(sections)
    assert prefix == ["s"]


def test_layout_with_nothing_stable_gives_an_empty_prefix():
    _, prefix = cache_friendly_layout([("user", ["q"], False)])
    assert prefix == []


def test_layout_recovers_the_prefix_that_a_bad_order_destroyed():
    """Динамическая строка сверху — и кэшируемого префикса нет вовсе."""
    bad = [("clock", ["now"], False), ("system", ["s1", "s2"], True)]
    _, before = cache_friendly_layout([bad[0]])
    _, after = cache_friendly_layout(bad)
    assert before == []
    assert after == ["s1", "s2"]


# -------------------------------------------------------------- cache_lookup
def test_lookup_hits_a_stored_prefix():
    assert cache_lookup([PROMPT[:6]], PROMPT, min_cacheable=SMALL) == 6


def test_lookup_misses_when_the_first_token_differs():
    """Расхождение в первом токене обнуляет выгоду целиком."""
    other = ["ts"] + PROMPT[1:]
    assert cache_lookup([PROMPT[:6]], other, min_cacheable=SMALL) == 0


def test_lookup_has_no_partial_hits():
    """Совпало 5 токенов из 6 — провайдер не засчитает ни одного."""
    almost = PROMPT[:5] + ["OTHER"] + PROMPT[6:]
    assert cache_lookup([PROMPT[:6]], almost, min_cacheable=SMALL) == 0


def test_lookup_ignores_blocks_below_the_threshold():
    """Тот самый порог в 1024 токена: короткий блок не кэшируется молча."""
    assert cache_lookup([PROMPT[:6]], PROMPT, min_cacheable=8) == 0


def test_lookup_picks_the_longest_matching_entry():
    cache = [PROMPT[:4], PROMPT[:6], PROMPT[:5]]
    assert cache_lookup(cache, PROMPT, min_cacheable=SMALL) == 6


def test_lookup_of_an_empty_cache_is_a_miss():
    assert cache_lookup([], PROMPT, min_cacheable=SMALL) == 0


# --------------------------------------------------------------- split_tokens
def test_first_request_writes_the_whole_cacheable_block():
    assert split_tokens(PROMPT, [], 6, min_cacheable=SMALL) == {
        "read": 0,
        "write": 6,
        "fresh": 4,
    }


def test_second_request_reads_instead_of_writing():
    assert split_tokens(PROMPT, [PROMPT[:6]], 6, min_cacheable=SMALL) == {
        "read": 6,
        "write": 0,
        "fresh": 4,
    }


def test_tokens_below_the_breakpoint_are_never_free():
    """Всё, что после точки останова, всегда идёт по полной ставке."""
    split = split_tokens(PROMPT, [PROMPT[:6]], 6, min_cacheable=SMALL)
    assert split["fresh"] == len(PROMPT) - 6


def test_split_always_accounts_for_every_token():
    for cache in ([], [PROMPT[:6]]):
        split = split_tokens(PROMPT, cache, 6, min_cacheable=SMALL)
        assert split["read"] + split["write"] + split["fresh"] == len(PROMPT)


def test_a_block_below_the_threshold_disables_the_cache_entirely():
    split = split_tokens(PROMPT, [PROMPT[:6]], 6, min_cacheable=999)
    assert split == {"read": 0, "write": 0, "fresh": len(PROMPT)}


def test_breakpoint_past_the_end_of_the_prompt_is_an_error():
    with pytest.raises(ValueError):
        split_tokens(PROMPT, [], len(PROMPT) + 1, min_cacheable=SMALL)


# --------------------------------------------------------------- request_cost
def test_fresh_tokens_cost_the_list_price():
    assert request_cost({"read": 0, "write": 0, "fresh": 1000}, 3.0) == APPROX(0.003)


def test_a_cache_write_costs_more_than_a_plain_token():
    """Вот почему кэш на одном-единственном вызове убыточен."""
    plain = request_cost({"read": 0, "write": 0, "fresh": 1000}, 3.0)
    written = request_cost({"read": 0, "write": 1000, "fresh": 0}, 3.0)
    assert written == APPROX(plain * WRITE_MULTIPLIER)
    assert written > plain


def test_a_cache_read_costs_a_tenth_of_a_plain_token():
    plain = request_cost({"read": 0, "write": 0, "fresh": 1000}, 3.0)
    read = request_cost({"read": 1000, "write": 0, "fresh": 0}, 3.0)
    assert read == APPROX(plain * READ_MULTIPLIER)


def test_cost_scales_with_the_price_of_the_model():
    split = {"read": 500, "write": 500, "fresh": 500}
    assert request_cost(split, 6.0) == APPROX(2 * request_cost(split, 3.0))


# ------------------------------------------------------------ simulate_session
def test_a_single_request_loses_money_on_the_cache():
    """Одна запись без единого чтения — плюс 25% к счёту, а не минус."""
    got = simulate_session([PROMPT], 6, 3.0, min_cacheable=SMALL)
    assert got["writes"] == 1
    assert got["reads"] == 0
    assert got["total_cost_usd"] > got["no_cache_cost_usd"]
    assert got["saving_pct"] < 0


def test_repeated_requests_turn_the_cache_profitable():
    got = simulate_session([PROMPT] * 10, 6, 3.0, min_cacheable=SMALL)
    assert got["writes"] == 1
    assert got["reads"] == 9
    assert got["total_cost_usd"] < got["no_cache_cost_usd"]
    assert got["saving_pct"] > 40


def test_saving_grows_with_the_number_of_reads():
    two = simulate_session([PROMPT] * 2, 6, 3.0, min_cacheable=SMALL)["saving_pct"]
    ten = simulate_session([PROMPT] * 10, 6, 3.0, min_cacheable=SMALL)["saving_pct"]
    assert ten > two


def test_a_drifting_first_token_kills_every_hit():
    """Динамический штамп времени сверху — и каждый запрос пишет заново."""
    prompts = [[f"ts={i}"] + PROMPT[1:] for i in range(10)]
    got = simulate_session(prompts, 6, 3.0, min_cacheable=SMALL)
    assert got["reads"] == 0
    assert got["writes"] == 10
    assert got["total_cost_usd"] > got["no_cache_cost_usd"]


def test_three_identical_requests_match_the_worked_example():
    got = simulate_session([PROMPT] * 3, 6, 3.0, min_cacheable=SMALL)
    assert got["saving_pct"] == APPROX(31.0)


def test_session_does_not_mutate_the_prompts():
    prompts = [list(PROMPT) for _ in range(3)]
    simulate_session(prompts, 6, 3.0, min_cacheable=SMALL)
    assert prompts == [PROMPT, PROMPT, PROMPT]


def test_a_block_below_the_threshold_saves_nothing_at_any_volume():
    got = simulate_session([PROMPT] * 50, 6, 3.0, min_cacheable=MIN_CACHEABLE_TOKENS)
    assert (got["reads"], got["writes"]) == (0, 0)
    assert got["total_cost_usd"] == APPROX(got["no_cache_cost_usd"])


# ----------------------------------------------------------- break_even_reads
def test_anthropic_five_minute_cache_pays_back_after_one_read():
    assert break_even_reads() == 1


def test_extended_one_hour_ttl_needs_a_second_read():
    """Премия за запись удваивается — и порог окупаемости растёт."""
    assert break_even_reads(2.0) == 2


def test_without_a_write_premium_the_cache_pays_back_immediately():
    assert break_even_reads(1.0) == 0


def test_break_even_matches_the_simulation():
    """Столько чтений, сколько обещает формула, — и сессия уже в плюсе."""
    n = break_even_reads()
    got = simulate_session([PROMPT] * (n + 1), 6, 3.0, min_cacheable=SMALL)
    assert got["total_cost_usd"] < got["no_cache_cost_usd"]


def test_one_read_short_of_break_even_is_still_a_loss():
    got = simulate_session([PROMPT] * break_even_reads(), 6, 3.0, min_cacheable=SMALL)
    assert got["total_cost_usd"] > got["no_cache_cost_usd"]


def test_a_read_that_is_not_cheaper_never_pays_back():
    with pytest.raises(ValueError):
        break_even_reads(1.25, 1.0)
