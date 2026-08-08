"""
Human-in-the-loop: propose-then-commit

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l15-propose-then-commit
Разбор:  /check-code p15-l15-propose-then-commit
"""

import hashlib
import json

REQUIRED_METADATA = ("intent", "lineage", "blast_radius", "rollback")
CHECKLIST = ("understood_resource", "verified_blast_radius", "rollback_ready")
TTL_SECONDS = 900.0


def idempotency_key(thread_id, action, payload):
    """Ключ идемпотентности предложения: 16 hex-символов от (тред, действие, данные).

    len(idempotency_key("t-1", "email.send", {"to": "a"}))    ->  16
    idempotency_key("t-1", "x", {"a": 1, "b": 2})
        ==  idempotency_key("t-1", "x", {"b": 2, "a": 1})     ->  True
    idempotency_key("t-1", "x", {"a": 1})
        ==  idempotency_key("t-2", "x", {"a": 1})             ->  False

    Порядок ключей в payload не должен влиять на результат: иначе один и тот же
    перевод денег получит два разных ключа и выполнится дважды.

    Главная ловушка урока: НЕ добавляй сюда время. С временем в подписи каждый
    повтор после сетевого сбоя даёт новый ключ, и одобренное однажды действие
    выполняется столько раз, сколько было повторов.
    """
    raise NotImplementedError


def missing_metadata(proposal):
    """Каких обязательных метаданных не хватает предложению.

    missing_metadata({"intent": "i", "lineage": "l",
                      "blast_radius": "b", "rollback": "r"})   ->  ()
    missing_metadata({"intent": "i"})
        ->  ('lineage', 'blast_radius', 'rollback')

    Пустая строка считается отсутствующим полем: rollback="" — это не план
    откатa, это галочка ради галочки.

    Порядок в ответе — как в REQUIRED_METADATA, чтобы сообщение ревьюеру было
    предсказуемым.
    """
    raise NotImplementedError


def is_expired(record, now):
    """Истёк ли срок предложения к моменту now.

    is_expired({"expires_at": 100.0}, now=99.0)    ->  False
    is_expired({"expires_at": 100.0}, now=100.0)   ->  True

    Ровно в момент expires_at считаем истёкшим: граница закрыта в сторону
    безопасного отказа. Время приходит параметром, а не из time.time(), иначе
    тест на просрочку пришлось бы писать через sleep.
    """
    raise NotImplementedError


def propose(store, proposal, now, ttl=TTL_SECONDS):
    """Сохраняет предложение в store со статусом "pending" и возвращает его ключ.

    store = {}
    key = propose(store, full_proposal, now=0.0)
    store[key]["status"]      ->  'pending'
    store[key]["expires_at"]  ->  900.0
    propose(store, full_proposal, now=500.0) == key   ->  True

    Повторный propose того же действия НЕ создаёт вторую запись и не сбрасывает
    статус: это и есть смысл ключа идемпотентности. Уже одобренное предложение
    после повтора остаётся одобренным.

    Предложение без обязательных метаданных — не предложение: бросай
    ValueError. Ревьюер, которому не показали blast radius и план откатa,
    физически не может провести настоящий review.
    """
    raise NotImplementedError


def approve(store, key, answers, now):
    """Одобряет предложение только при полностью заполненном чеклисте.

    approve(store, key, {}, now=1.0)                        ->  False
    approve(store, key, {"understood_resource": True}, 1.0)  ->  False
    approve(store, key, dict.fromkeys(CHECKLIST, True), 1.0) ->  True

    answers — словарь ответов на вопросы CHECKLIST. Требуется именно True:
    "yes", 1 и прочая правдоподобная ерунда не проходит, потому что галочка,
    поставленная не глядя, — это ровно тот rubber-stamp, от которого чеклист и
    защищает.

    Просроченное предложение одобрить нельзя: его статус переводится в
    "expired". Одобрять можно только "pending" — повторное одобрение уже
    одобренного вернёт False.
    """
    raise NotImplementedError


def commit(store, key, execute, now):
    """Передаёт одобренное действие идемпотентному исполнителю.

    commit(store, key, execute, now=1.0)   ->  'refused'   (ещё не одобрено)
    commit(store, key, execute, now=1.0)   ->  'committed' (после approve)
    commit(store, key, execute, now=2.0)   ->  'already-committed'

    execute — функция (key, action, payload). Она ОБЯЗАНА атомарно хранить key
    рядом с побочным эффектом и при повторе ключа возвращать прежний результат,
    не повторяя эффект. Локальная запись статуса сама по себе не даёт exactly
    once: процесс может упасть после эффекта, но до status="committed".

    Возможные исходы:
      'refused'           — статус не "approved", execute не вызывался;
      'expired'           — срок вышел между одобрением и применением;
      'committed'         — выполнено сейчас;
      'already-committed' — выполнено раньше, execute снова не вызывался.

    Статус пишется ПОСЛЕ execute: запись до вызова оставила бы «выполнено» без
    эффекта. Обратный crash gap закрывает не этот store, а исполнитель,
    получающий тот же key при повторе. Если целевая система не поддерживает
    атомарную дедупликацию по ключу, честная гарантия здесь — at least once,
    поэтому такой execute не соответствует контракту. За commit всё равно
    идёт verify: идемпотентность предотвращает дубль, но не доказывает эффект.
    """
    raise NotImplementedError


def verify(store, key, read_back):
    """Перечитывает цель после commit и фиксирует, случился ли побочный эффект.

    verify(store, key, lambda a, p: True)    ->  True    статус 'verified'
    verify(store, key, lambda a, p: False)   ->  False   статус 'verify-failed'

    read_back — функция (action, payload), которая ходит в целевую систему и
    отвечает, видно ли там результат. Возврат 200 от инструмента — это не
    проверка; проверка — это чтение.

    Если статус ещё не "committed", проверять нечего: возвращаем False, не
    трогая запись. Статус "verify-failed" — известное плохое состояние, дальше
    по нему включается откат (урок 16).
    """
    raise NotImplementedError
