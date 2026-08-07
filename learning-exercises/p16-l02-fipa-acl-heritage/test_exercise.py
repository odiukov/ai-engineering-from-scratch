"""Тесты к уроку «Наследие FIPA-ACL и речевые акты». Правь exercise.py."""

import pytest

from exercise import (
    PERFORMATIVES,
    ACLError,
    MissingFieldError,
    UnknownPerformativeError,
    award,
    cfp,
    collect_proposals,
    make_message,
    mcp_to_acl,
    render,
    reply_to,
    run_contract_net,
)

MCP_CALL = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": 42,
    "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}},
}
MCP_READ = {
    "jsonrpc": "2.0",
    "method": "resources/read",
    "id": 43,
    "params": {"uri": "file:///etc/hosts"},
}


# ------------------------------------------------------------- make_message
def test_make_message_keeps_the_performative():
    assert make_message("inform", "a1", "a2", "x")["performative"] == "inform"


def test_make_message_fills_fipa_defaults():
    msg = make_message("inform", "a1", "a2", "x")
    assert (msg["language"], msg["ontology"]) == ("SL0", "default")


def test_unknown_performative_is_rejected():
    """Каталог performatives закрыт: чего нет в FIPA, того нет и здесь."""
    with pytest.raises(UnknownPerformativeError):
        make_message("yell", "a1", "a2", "x")


def test_every_catalogued_performative_is_accepted():
    for p in PERFORMATIVES:
        assert make_message(p, "a1", "a2", "x")["performative"] == p


def test_empty_sender_is_a_missing_field():
    """Ловушка: пустая строка — это отсутствующее поле, а не короткое."""
    with pytest.raises(MissingFieldError):
        make_message("inform", "", "a2", "x")


def test_empty_receiver_is_a_missing_field():
    with pytest.raises(MissingFieldError):
        make_message("inform", "a1", "", "x")


def test_optional_envelope_fields_default_to_none():
    msg = make_message("inform", "a1", "a2", "x")
    assert msg["protocol"] is None and msg["conversation_id"] is None


# ------------------------------------------------------------------- render
def test_render_produces_the_canonical_envelope():
    text = render(make_message("inform", "a1", "a2", "((price IBM 83))"))
    assert text == "\n".join([
        "(inform",
        "  :sender a1",
        "  :receiver a2",
        "  :content '((price IBM 83))'",
        "  :language SL0",
        "  :ontology default",
        ")",
    ])


def test_render_skips_empty_fields():
    """Конверт без protocol не должен печатать пустую строку ':protocol'."""
    assert ":protocol" not in render(make_message("inform", "a1", "a2", "x"))


def test_render_shows_conversation_id_when_present():
    text = render(make_message("inform", "a1", "a2", "x", conversation_id="c-1"))
    assert "  :conversation-id c-1" in text


def test_render_survives_structured_content():
    """content бывает dict — repr делает его однозначным."""
    text = render(make_message("propose", "w1", "mgr", {"price": 3}))
    assert "  :content {'price': 3}" in text


# ----------------------------------------------------------------- reply_to
def test_reply_swaps_the_addresses():
    m = make_message("cfp", "mgr", "w1", "task", conversation_id="cn-1")
    r = reply_to(m, "propose", {"price": 3})
    assert (r["sender"], r["receiver"]) == ("w1", "mgr")


def test_reply_carries_the_same_conversation_id():
    """Нить разговора — это conversation_id. Новый id рвёт переговоры."""
    m = make_message("cfp", "mgr", "w1", "task", conversation_id="cn-1")
    assert reply_to(m, "propose", {"price": 3})["conversation_id"] == "cn-1"


def test_reply_points_back_at_the_original_message():
    m = make_message("cfp", "mgr", "w1", "task", conversation_id="cn-1",
                     reply_with="cfp-w1")
    assert reply_to(m, "propose", {"price": 3})["in_reply_to"] == "cfp-w1"


def test_reply_inherits_the_interaction_protocol():
    m = make_message("cfp", "mgr", "w1", "t", protocol="fipa-contract-net",
                     conversation_id="cn-1")
    assert reply_to(m, "propose", {"price": 1})["protocol"] == "fipa-contract-net"


def test_reply_still_validates_the_performative():
    m = make_message("cfp", "mgr", "w1", "task", conversation_id="cn-1")
    with pytest.raises(UnknownPerformativeError):
        reply_to(m, "shrug", "x")


# --------------------------------------------------------------- mcp_to_acl
def test_tools_call_maps_to_request():
    assert mcp_to_acl(MCP_CALL)["performative"] == "request"


def test_resources_read_maps_to_query_ref():
    assert mcp_to_acl(MCP_READ)["performative"] == "query-ref"


def test_jsonrpc_id_becomes_the_correlation_id():
    """id из JSON-RPC — тот самый correlation id, ради которого жил FIPA."""
    assert mcp_to_acl(MCP_CALL)["conversation_id"] == "jsonrpc-42"


def test_tool_name_becomes_the_ontology():
    assert mcp_to_acl(MCP_CALL)["ontology"] == "lookup_stock"


def test_unmapped_method_is_refused():
    with pytest.raises(ACLError):
        mcp_to_acl({"jsonrpc": "2.0", "method": "sampling/createMessage", "id": 1,
                    "params": {}})


# --------------------------------------------------------------------- cfp
def test_cfp_reaches_every_bidder():
    msgs = cfp("mgr", ["w1", "w2", "w3"], "compress logs", "cn-1")
    assert [m["receiver"] for m in msgs] == ["w1", "w2", "w3"]


def test_cfp_gives_each_invitation_its_own_reply_with():
    msgs = cfp("mgr", ["w1", "w2"], "t", "cn-1")
    assert len({m["reply_with"] for m in msgs}) == 2


def test_cfp_without_bidders_sends_nothing():
    assert cfp("mgr", [], "t", "cn-1") == []


# -------------------------------------------------------- collect_proposals
def test_collect_proposals_ignores_other_performatives():
    log = cfp("mgr", ["w1"], "t", "cn-1")
    assert collect_proposals(log, "cn-1") == []


def test_collect_proposals_ignores_another_auction():
    """Журнал общий: без фильтра по нити задача уйдёт в чужой аукцион."""
    log = run_contract_net("mgr", ["w1"], "t", "cn-1", {"w1": {"price": 1}})
    assert collect_proposals(log, "cn-2") == []


# ------------------------------------------------------------------- award
def test_award_without_proposals_gives_nothing():
    """Никто не откликнулся — присуждать нечего, и это не ошибка."""
    assert award([]) == (None, [])


def test_award_picks_the_cheapest_bid():
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1",
                           {"w1": {"price": 3}, "w2": {"price": 2}})
    accept, _ = award(collect_proposals(log, "cn-1"))
    assert accept["receiver"] == "w2"


def test_award_rejects_everyone_else():
    log = run_contract_net("mgr", ["w1", "w2", "w3"], "t", "cn-1",
                           {"w1": {"price": 3}, "w2": {"price": 2}, "w3": {"price": 9}})
    _, rejects = award(collect_proposals(log, "cn-1"))
    assert sorted(m["receiver"] for m in rejects) == ["w1", "w3"]


def test_award_respects_a_custom_score():
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1",
                           {"w1": {"price": 3, "eta": 1}, "w2": {"price": 2, "eta": 40}})
    accept, _ = award(collect_proposals(log, "cn-1"), score=lambda c: c["eta"])
    assert accept["receiver"] == "w1"


def test_award_keeps_the_conversation_thread():
    log = run_contract_net("mgr", ["w1"], "t", "cn-1", {"w1": {"price": 3}})
    accept, _ = award(collect_proposals(log, "cn-1"))
    assert accept["conversation_id"] == "cn-1"


# --------------------------------------------------------- run_contract_net
def test_contract_net_runs_the_full_sequence():
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1",
                           {"w1": {"price": 3}, "w2": {"price": 2}})
    assert [m["performative"] for m in log] == [
        "cfp", "cfp", "propose", "propose", "accept-proposal", "reject-proposal",
    ]


def test_contract_net_without_bids_awards_nothing():
    """Приглашения ушли, заявок нет — задача не присуждается никому."""
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1", {})
    assert [m["performative"] for m in log] == ["cfp", "cfp"]
    assert not any(m["performative"] == "accept-proposal" for m in log)


def test_contract_net_awards_exactly_one_winner():
    log = run_contract_net("mgr", ["w1", "w2", "w3"], "t", "cn-1",
                           {"w1": {"price": 3}, "w2": {"price": 2}, "w3": {"price": 5}})
    assert sum(m["performative"] == "accept-proposal" for m in log) == 1


def test_a_silent_bidder_still_gets_invited():
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1", {"w1": {"price": 3}})
    invited = [m["receiver"] for m in log if m["performative"] == "cfp"]
    assert invited == ["w1", "w2"]


def test_uninvited_bid_is_ignored():
    """Заявка от того, кого не звали, в аукцион не попадает."""
    log = run_contract_net("mgr", ["w1"], "t", "cn-1",
                           {"w1": {"price": 9}, "stranger": {"price": 1}})
    accept, _ = award(collect_proposals(log, "cn-1"))
    assert accept["receiver"] == "w1"


def test_every_message_shares_one_conversation_id():
    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-7",
                           {"w1": {"price": 3}, "w2": {"price": 2}})
    assert {m["conversation_id"] for m in log} == {"cn-7"}
