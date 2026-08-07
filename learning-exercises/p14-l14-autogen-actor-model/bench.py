"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _enqueue(runtime, sender, recipient, body):
    """Свой мини-send, чтобы замер не зависел от того, чей exercise загружен."""
    runtime["counter"] += 1
    runtime["actors"][recipient]["inbox"].append(
        {"sender": sender, "recipient": recipient, "topic": "tick",
         "body": body, "mid": runtime["counter"]}
    )


def _self_loop(state, message, runtime):
    """Отвечает сам себе: ящик не пустеет, но и не растёт."""
    state["n"] = state.get("n", 0) + 1
    me = message["recipient"]
    _enqueue(runtime, me, me, 0)


def _pong(state, message, runtime):
    """Пинг-понг двух акторов: работа на вызов остаётся постоянной."""
    me = message["recipient"]
    _enqueue(runtime, me, "b" if me == "a" else "a", message["body"] + 1)


def _busiest(runtime):
    ranked = sorted(runtime["actors"].items(),
                    key=lambda kv: (-len(kv[1]["inbox"]), kv[0]))
    name, actor = ranked[0]
    return name if actor["inbox"] else None


def _runtime(names, handler):
    rt = {"actors": {}, "dead_letters": [], "counter": 0}
    for name in names:
        rt["actors"][name] = {"handler": handler, "state": {}, "inbox": []}
    return rt


def _message(mid, recipient):
    return {"sender": "__user__", "recipient": recipient,
            "topic": "tick", "body": mid, "mid": mid}


_send_rt = _runtime([f"a{i}" for i in range(50)], _self_loop)

_publish_rt = _runtime([f"a{i}" for i in range(8)], _self_loop)
_subscribers = [f"a{i}" for i in range(8)]

_deliver_rt = _runtime(["solo"], _self_loop)
_deliver_rt["actors"]["solo"]["inbox"] = [_message(i, "solo") for i in range(200)]

_rr_rt = _runtime(["a", "b"], _pong)
_rr_rt["actors"]["a"]["inbox"].append(_message(1, "a"))

_sel_rt = _runtime(["a", "b"], _pong)
_sel_rt["actors"]["a"]["inbox"].append(_message(1, "a"))

_dlq_rt = _runtime([], _self_loop)
_dlq_rt["dead_letters"] = [
    (_message(i, "ghost"), f"no actor 'ghost{i % 7}'") for i in range(2000)
]

BENCH = {
    "send": (_send_rt, "__user__", "a7", "tick", 0),
    "publish": (_publish_rt, "__user__", "tick", _subscribers, 0),
    "deliver_one": (_deliver_rt, "solo"),
    "run_round_robin": (_rr_rt, ["a", "b"], 50, 100),
    "run_selector": (_sel_rt, _busiest, 100),
    "dead_letter_report": (_dlq_rt,),
}
