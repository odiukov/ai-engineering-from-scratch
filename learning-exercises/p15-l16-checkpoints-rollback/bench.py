"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# Состояние из 300 счетов и журнала на 300 записей: снапшот такого словаря уже
# заметно дороже поверхностной копии, и разница видна на замере.
_STATE = {f"balance_{i}": random.randint(0, 10_000) for i in range(300)}
_STATE["limits"] = {f"acct_{i}": random.randint(1, 100) for i in range(100)}
_STATE["sent"] = [f"tx-{i:04d}" for i in range(300)]

_SNAP = {key: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
         for key, v in _STATE.items()}

# Журнал из 400 чекпоинтов: find_checkpoint ищет с конца, и на длинном журнале
# видно, кто пошёл с начала.
_LOG = [
    {"name": f"tx-{i:04d}:verified", "state": {"balance_A": i}, "at": float(i)}
    for i in range(400)
]

_LEASE = {"worker": "w1", "until": 30.0}


def _true(state):
    return True


def _touch(state):
    state["balance_0"] = state.get("balance_0", 0)


# Шаг уже закрыт терминальным чекпоинтом tx-0000:verified — значит замер идёт по
# идемпотентной ветке, не мутирует ни состояние, ни журнал, и упирается ровно в
# поиск по журналу.
_STEP = {
    "id": "tx-0000",
    "precondition": _true,
    "apply": _touch,
    "verify": _true,
}

# Для checkpoint состояние маленькое: журнал в этом замере растёт с каждым
# вызовом, и большие снапшоты просто съели бы память.
_SMALL_STATE = {"balance_A": 1500, "sent": ["tx-0001"]}

BENCH = {
    "snapshot": (_STATE,),
    "restore": (dict(_STATE), _SNAP),
    "checkpoint": ([], "tx:before", _SMALL_STATE, 0.0),
    "find_checkpoint": (_LOG, "tx-0000:verified"),
    "rollback_to": ({"balance_A": 0}, _LOG, "tx-0000:verified"),
    "lease_expired": (_LEASE, 10.0),
    "claim_lease": (_LEASE, "w2", 10.0, 30.0),
    "run_step": ({"balance_0": 1}, _LOG, _STEP, 1.0),
}
