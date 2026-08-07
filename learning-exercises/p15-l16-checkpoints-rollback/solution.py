"""
Чекпоинты и откат — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь руками собрано то, что в проде дают LangGraph checkpointing, Cloudflare
Durable Objects и Checkpoint-примитивы Microsoft Agent Framework: снапшот
состояния на каждом переходе, lease recovery при падении воркера и связка
«идемпотентность + precondition + verify + rollback».

Что проверяют тесты: откат к снапшоту восстанавливает состояние ПОЛНОСТЬЮ и не
оставляет побочных эффектов, precondition не даёт выполнить действие на
изменившемся состоянии, повтор после успеха ничего не делает второй раз.

Только защита. Кода, который обходит проверки, здесь нет и быть не должно.

Журнал — список словарей, состояние — словарь, время — параметр now.
Ни файлов, ни сети, ни LLM.
"""


def snapshot(state):
    """Глубокая копия состояния: словари и списки копируются рекурсивно.

    snapshot({"a": 1})                    ->  {'a': 1}
    snapshot({"sent": ["x"]})["sent"]     ->  ['x']   (новый список)

    Ловушка, на которой ломается откат: dict(state) — копия поверхностная.
    Вложенный список sent останется тем же объектом, apply допишет в него
    элемент, и «снапшот» изменится вместе с живым состоянием. Откатывать будет
    некуда.

    Скаляры и строки неизменяемы, их копировать не нужно. set и произвольные
    объекты эта функция не разбирает — держи состояние из dict, list и скаляров,
    как и делают настоящие чекпоинт-бэкенды (они умеют только JSON).
    """
    if isinstance(state, dict):
        return {key: snapshot(value) for key, value in state.items()}
    if isinstance(state, list):
        return [snapshot(item) for item in state]
    return state


def restore(state, snap):
    """Возвращает state к снапшоту НА МЕСТЕ, не создавая новый объект.

    live = {"a": 1, "tmp": 2}
    restore(live, {"a": 1})   ->  {'a': 1}      ключ 'tmp' исчез
    live["a"]                 ->  1             тот же объект live

    Два требования, и оба важны:
      * ключи, появившиеся после снапшота, обязаны исчезнуть — иначе «полного»
        восстановления не будет, останется мусор от неудачного действия;
      * менять надо тот же самый объект: на него держат ссылки и воркер, и
        функция шага. Вернуть новый словарь недостаточно.

    И ещё: в state кладётся копия снапшота, а не сам снапшот. Иначе следующая
    же запись в state испортит запись в журнале.
    """
    state.clear()
    state.update(snapshot(snap))
    return state


def checkpoint(log, name, state, now):
    """Пишет в журнал именованный снапшот состояния и возвращает запись.

    log = []
    checkpoint(log, "tx:before", {"a": 1}, now=5.0)
        ->  {'name': 'tx:before', 'state': {'a': 1}, 'at': 5.0}
    len(log)  ->  1

    Продакшен-реализации пишут КАЖДЫЙ переход, а не только точки коммита: пара
    лишних записей стоит дешевле, чем восстановление «примерно туда».

    В журнал уезжает снапшот, а не ссылка на состояние: иначе через минуту в
    журнале будет лежать текущее состояние под видом старого.
    """
    entry = {"name": name, "state": snapshot(state), "at": now}
    log.append(entry)
    return entry


def find_checkpoint(log, name=None):
    """Последний чекпоинт с таким именем (или последний вообще). None, если нет.

    find_checkpoint([])                       ->  None
    find_checkpoint(log, "tx:before")["at"]   ->  5.0
    find_checkpoint(log)["name"]              ->  'tx:verified'   последний

    Именно «последний», а не первый: чекпоинты latest-wins, и после повторного
    прохода по тому же шагу актуален свежий.

    Ищи с конца — так дешевле и так честнее по смыслу.
    """
    for entry in reversed(log):
        if name is None or entry["name"] == name:
            return entry
    return None


def rollback_to(state, log, name):
    """Откатывает состояние к названному чекпоинту. Возвращает эту запись.

    rollback_to(state, log, "tx:before")["name"]  ->  'tx:before'
    rollback_to(state, log, "нет такого")         ->  LookupError

    Неизвестное имя — это не «откатим куда получится», это ошибка: бросай
    LookupError. Молчаливый откат в непонятное состояние хуже падения.

    Само состояние правится на месте (см. restore), поэтому все, кто держит
    ссылку на state, увидят откат. Побочные эффекты, записанные в state после
    снапшота, исчезают вместе с ним — это и есть in-band rollback.
    """
    entry = find_checkpoint(log, name)
    if entry is None:
        raise LookupError(f"no checkpoint named {name!r}")
    restore(state, entry["state"])
    return entry


def lease_expired(lease, now):
    """Истекла ли аренда (lease) воркера к моменту now.

    lease_expired(None, now=0.0)                     ->  True
    lease_expired({"worker": "w1", "until": 10.0}, 9.0)   ->  False
    lease_expired({"worker": "w1", "until": 10.0}, 10.0)  ->  True

    Аренды нет — считаем истёкшей: работу надо кому-то забрать. Ровно в момент
    until тоже истекла, граница закрыта.

    Это механизм, из-за которого rolling deploy не теряет незавершённые
    прогоны: упавший воркер просто перестаёт продлевать аренду.
    """
    if lease is None:
        return True
    return now >= lease["until"]


def claim_lease(lease, worker, now, duration):
    """Пытается взять аренду. Возвращает новую аренду или None, если занято.

    claim_lease(None, "w1", now=0.0, duration=30.0)
        ->  {'worker': 'w1', 'until': 30.0}
    claim_lease({"worker": "w1", "until": 30.0}, "w2", 10.0, 30.0)  ->  None
    claim_lease({"worker": "w1", "until": 30.0}, "w2", 30.0, 30.0)
        ->  {'worker': 'w2', 'until': 60.0}

    Правила: живую аренду чужого воркера забирать нельзя; истёкшую — можно;
    свою собственную можно продлить в любой момент (это и есть heartbeat).

    Возврат None, а не исключение: «не смог забрать» — нормальный исход в гонке
    двух воркеров, а не авария.
    """
    if lease is not None and not lease_expired(lease, now) and lease["worker"] != worker:
        return None
    return {"worker": worker, "until": now + duration}


def run_step(state, log, step, now):
    """Выполняет шаг по схеме идемпотентность → precondition → apply → verify → rollback.

    step — словарь: id, precondition(state), apply(state), verify(state).
    Возвращает одну из строк:
      'already-done'         — шаг уже проверен раньше, apply не вызывался;
      'precondition-failed'  — состояние разошлось с одобренным, apply не вызывался;
      'rolled-back'          — apply сработал, verify нет, состояние откачено;
      'ok'                   — применено и подтверждено чтением.

    run_step(state, log, transfer, now=1.0)   ->  'ok'
    run_step(state, log, transfer, now=2.0)   ->  'already-done'

    Все четыре части нужны, и каждая закрывает свой класс отказа:
      идемпотентность — повтор после сбоя;
      precondition    — состояние изменилось между одобрением и применением
                        (одобрили перевод при балансе 1500, к моменту
                        применения там 500);
      verify          — «инструмент вернул 200», а эффекта нет;
      rollback        — известное плохое состояние надо вернуть назад.

    Снапшот ":before" пишется ДО apply — иначе откатывать будет не к чему.
    """
    sid = step["id"]

    # Идемпотентность: терминальный чекпоинт означает, что шаг закрыт.
    if find_checkpoint(log, f"{sid}:verified") is not None:
        return "already-done"

    if not step["precondition"](state):
        checkpoint(log, f"{sid}:aborted", state, now)
        return "precondition-failed"

    checkpoint(log, f"{sid}:before", state, now)
    step["apply"](state)
    checkpoint(log, f"{sid}:applied", state, now)

    if not step["verify"](state):
        rollback_to(state, log, f"{sid}:before")
        checkpoint(log, f"{sid}:rolled-back", state, now)
        return "rolled-back"

    checkpoint(log, f"{sid}:verified", state, now)
    return "ok"
