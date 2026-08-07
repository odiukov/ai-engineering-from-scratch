"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_bidders = [f"w{i}" for i in range(400)]
_bids = {b: {"price": random.randint(1, 100)} for b in _bidders}

_msg = {
    "performative": "propose",
    "sender": "w1",
    "receiver": "mgr",
    "content": {"price": 3},
    "language": "SL0",
    "ontology": "contract-net",
    "protocol": "fipa-contract-net",
    "conversation_id": "cn-1",
    "reply_with": "propose-w1",
    "in_reply_to": "cfp-w1",
}
_log = [dict(_msg) for _ in range(4000)]

_mcp_call = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": 42,
    "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}},
}

BENCH = {
    "make_message": ("inform", "a1", "a2", "((price IBM 83))"),
    "render": (_msg,),
    "reply_to": (_msg, "inform", "ok"),
    "mcp_to_acl": (_mcp_call,),
    "cfp": ("mgr", _bidders, "compress logs", "cn-1"),
    "collect_proposals": (_log, "cn-1"),
    "award": (_log,),
    "run_contract_net": ("mgr", _bidders, "t", "cn-1", _bids),
}
