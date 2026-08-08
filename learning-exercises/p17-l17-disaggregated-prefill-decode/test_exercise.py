"""Тесты к уроку «Разделение prefill и decode: два пула и цена передачи KV». Правь exercise.py."""

import pytest

from exercise import (
    COLOCATION_PENALTY,
    LINK_RDMA_GBPS,
    LINK_TCP_GBPS,
    TRANSFER_SETUP_MS,
    DisaggError,
    colocated_ms,
    crossover_prompt_tokens,
    disagg_gain_ms,
    disaggregated_ms,
    fleet_report,
    kv_bytes,
    phase_ms,
    transfer_ms,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, rel=1e-9)


# ------------------------------------------------------------------ kv_bytes
def test_four_thousand_tokens_use_the_full_llama70b_fp8_kv_geometry():
    assert kv_bytes(4000) == 655_360_000


def test_kv_size_is_linear_in_prompt_length():
    assert kv_bytes(8000) == 2 * kv_bytes(4000)


def test_empty_prompt_has_no_kv():
    assert kv_bytes(0) == 0


def test_negative_prompt_length_is_rejected():
    with pytest.raises(DisaggError):
        kv_bytes(-1)


# --------------------------------------------------------------- transfer_ms
def test_rdma_transfer_of_a_four_k_prompt():
    """655.36 МБ по 100 ГБ/с = 6.5536 мс, плюс 20 мс рукопожатия."""
    assert transfer_ms(4000, LINK_RDMA_GBPS) == APPROX(26.5536)


def test_tcp_transfer_of_the_same_prompt_is_ten_times_the_bytes_time():
    assert transfer_ms(4000, LINK_TCP_GBPS) == APPROX(85.536)


def test_handshake_is_paid_even_when_there_is_nothing_to_ship():
    """Постоянное слагаемое — то, из-за чего короткие промпты не окупаются."""
    assert transfer_ms(0, LINK_RDMA_GBPS) == APPROX(TRANSFER_SETUP_MS)


def test_a_dead_link_is_rejected():
    with pytest.raises(DisaggError):
        transfer_ms(4000, 0.0)


# ------------------------------------------------------------------ phase_ms
def test_phase_times_of_a_typical_rag_request():
    assert phase_ms(4000, 300) == {"prefill_ms": APPROX(500.0), "decode_ms": APPROX(7500.0)}


def test_decode_dominates_prefill_by_an_order_of_magnitude():
    """Prefill глотает промпт одним forward, decode идёт по шагу на токен."""
    phases = phase_ms(4000, 300)
    assert phases["decode_ms"] > 10 * phases["prefill_ms"]


def test_zero_rate_is_rejected():
    with pytest.raises(DisaggError):
        phase_ms(4000, 300, prefill_tps=0.0)


# -------------------------------------------------------------- colocated_ms
def test_colocated_request_pays_the_prefill_penalty():
    assert colocated_ms(4000, 300) == ROUGH(500.0 / 0.7 + 7500.0)


def test_without_a_penalty_colocation_is_just_the_sum_of_the_phases():
    assert colocated_ms(4000, 300, penalty=0.0) == APPROX(8000.0)


def test_penalty_divides_and_does_not_multiply():
    """Ловушка: 1/(1-0.3) = 1.4286, а не 1.30 — разница в 10% времени prefill."""
    penalised = colocated_ms(4000, 0, penalty=0.3)
    assert penalised == ROUGH(500.0 / 0.7)
    assert penalised > 500.0 * 1.3


def test_decode_is_not_penalised_by_colocation():
    """Decode упирается в полосу памяти — соседний prefill ему почти не мешает."""
    with_decode = colocated_ms(4000, 300)
    without_decode = colocated_ms(4000, 0)
    assert with_decode - without_decode == ROUGH(phase_ms(0, 300)["decode_ms"])


def test_penalty_of_one_is_rejected():
    with pytest.raises(DisaggError):
        colocated_ms(4000, 300, penalty=1.0)


# ---------------------------------------------------------- disaggregated_ms
def test_disaggregated_request_is_prefill_plus_transfer_plus_decode():
    assert disaggregated_ms(4000, 300, LINK_TCP_GBPS) == APPROX(8085.536)


def test_a_faster_link_makes_the_split_path_shorter():
    assert (disaggregated_ms(4000, 300, LINK_RDMA_GBPS)
            < disaggregated_ms(4000, 300, LINK_TCP_GBPS))


def test_the_split_path_carries_no_colocation_penalty():
    """В этом весь смысл: prefill идёт на своей GPU и не ждёт чужой decode."""
    phases = phase_ms(4000, 300)
    assert disaggregated_ms(4000, 300, LINK_TCP_GBPS) == APPROX(
        phases["prefill_ms"] + phases["decode_ms"] + transfer_ms(4000, LINK_TCP_GBPS))


# ------------------------------------------------------------ disagg_gain_ms
def test_a_long_prompt_pays_for_the_transfer():
    assert disagg_gain_ms(4000, 300, LINK_TCP_GBPS) > 0


def test_a_short_prompt_does_not_pay_for_the_transfer():
    """На 200 токенах штраф отбирает 8.6 мс, а рукопожатие стоит 20."""
    assert disagg_gain_ms(200, 300, LINK_TCP_GBPS) < 0


def test_the_gain_does_not_depend_on_the_answer_length():
    """Decode считается одинаково в обеих схемах и в разности сокращается."""
    assert (disagg_gain_ms(4000, 999, LINK_TCP_GBPS)
            == ROUGH(disagg_gain_ms(4000, 1, LINK_TCP_GBPS)))


def test_without_a_colocation_penalty_splitting_is_pure_loss():
    """Выигрывать нечего, а налог на передачу никуда не делся."""
    assert disagg_gain_ms(4000, 300, LINK_TCP_GBPS, penalty=0.0) == APPROX(-85.536)


def test_rdma_wins_more_than_tcp_on_the_same_request():
    assert (disagg_gain_ms(4000, 300, LINK_RDMA_GBPS)
            > disagg_gain_ms(4000, 300, LINK_TCP_GBPS))


# --------------------------------------------------- crossover_prompt_tokens
def test_tcp_crossover_is_close_to_the_five_hundred_tokens_of_the_lesson():
    assert crossover_prompt_tokens(LINK_TCP_GBPS) == 538


def test_a_faster_link_pays_off_on_shorter_prompts():
    assert crossover_prompt_tokens(LINK_RDMA_GBPS) < crossover_prompt_tokens(LINK_TCP_GBPS)


def test_the_crossover_is_the_first_prompt_length_that_wins():
    cross = crossover_prompt_tokens(LINK_TCP_GBPS)
    assert disagg_gain_ms(cross, 0, LINK_TCP_GBPS) > 0
    assert disagg_gain_ms(cross - 1, 0, LINK_TCP_GBPS) <= 0


def test_a_link_slower_than_the_break_even_never_pays_off():
    """Ниже ~3.06 ГБ/с налог на токен растёт быстрее выигрыша: длина не спасёт."""
    assert crossover_prompt_tokens(3.0) is None
    assert crossover_prompt_tokens(4.0) is not None


def test_a_heavier_colocation_penalty_lowers_the_crossover():
    assert (crossover_prompt_tokens(LINK_TCP_GBPS, penalty=0.5)
            < crossover_prompt_tokens(LINK_TCP_GBPS, penalty=COLOCATION_PENALTY))


# --------------------------------------------------------------- fleet_report
def test_a_fleet_of_short_chats_loses_from_splitting():
    fleet = [(150, 120)] * 200
    report = fleet_report(fleet, LINK_TCP_GBPS)
    assert report["gain_ms"] < 0
    assert report["hurt"] == 200
    assert report["helped"] == 0


def test_a_fleet_of_long_rag_prefixes_wins():
    fleet = [(8000, 300)] * 200
    report = fleet_report(fleet, LINK_TCP_GBPS)
    assert report["gain_ms"] > 0
    assert report["helped"] == 200


def test_a_mixed_fleet_splits_into_helped_and_hurt():
    fleet = [(8000, 300)] * 100 + [(150, 120)] * 100
    report = fleet_report(fleet, LINK_TCP_GBPS)
    assert report["helped"] == 100
    assert report["hurt"] == 100
    assert report["requests"] == 200


def test_the_same_fleet_gains_more_over_rdma():
    fleet = [(8000, 300)] * 50
    assert (fleet_report(fleet, LINK_RDMA_GBPS)["gain_ms"]
            > fleet_report(fleet, LINK_TCP_GBPS)["gain_ms"])


def test_an_empty_fleet_does_not_divide_by_zero():
    assert fleet_report([], LINK_TCP_GBPS)["gain_pct"] == APPROX(0.0)
