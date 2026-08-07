"""
Память репозитория и durable state

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l34-repo-memory-and-state
Разбор:  /check-code p14-l34-repo-memory-and-state
"""

import json
import re

TEMP_SUFFIX = ".tmp"
SCHEMA_VERSION = 2
STATE_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "active_task_id", "touched_files", "risks", "next_action"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [SCHEMA_VERSION]},
        "active_task_id": {"type": ["string", "null"], "pattern": r"^T-\d{3,}$"},
        "touched_files": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
    },
}


class SchemaError(ValueError):
    """Схема отвергла значение. Наследуется от ValueError, а не от Exception:
    плохая запись в состояние — это ошибка значения, и ловить её надо адресно.
    """
    pass


def validate(value, schema, path="$"):
    """Проверить значение по подмножеству JSON Schema. Вернуть None или бросить SchemaError.

    Поддержано: type (в т.ч. список типов), enum, pattern, required,
    properties, items. Лишние поля в объекте запрещены.

    validate(5, {"type": "integer"})            ->  None
    validate("5", {"type": "integer"})          ->  SchemaError
    validate({"a": 1}, {"type": "object",
                        "properties": {}})      ->  SchemaError (лишнее поле a)

    Ловушка: в Python bool — подкласс int, поэтому True обязан проваливать
    проверку на "integer", иначе schema_version=True проедет в файл.
    pattern применяется только к строкам: у None паттерна нет.
    """
    raise NotImplementedError


def atomic_write(fs, path, content, crash_after_temp=False):
    """Атомарная запись в модель файловой системы. Вернуть НОВЫЙ словарь.

    Порядок: пишем во временный ключ path + TEMP_SUFFIX, потом переименовываем
    его в path. crash_after_temp=True изображает падение между этими шагами.

    atomic_write({}, "s.json", "{}")               ->  {'s.json': '{}'}
    atomic_write({"s.json": "old"}, "s.json", "new")["s.json"]  ->  'new'

    Смысл упражнения: после падения на входном fs обязано остаться СТАРОЕ
    содержимое, а временного ключа не должно остаться нигде. Наполовину
    записанный файл состояния хуже, чем отсутствие файла: сессия поднимется
    и продолжит работу по мусору.
    """
    raise NotImplementedError


def remember(memory, key, value, now):
    """Записать факт в память репозитория без дублей. Вернуть НОВЫЙ список.

    Запись: {"key", "value", "first_seen", "last_seen"}.

    first = remember([], "python", "3.12", 100)
        ->  [{'key': 'python', 'value': '3.12', 'first_seen': 100, 'last_seen': 100}]
    remember(first, "python", "3.12", 200)
        ->  список из ОДНОЙ записи: first_seen 100, last_seen 200

    Ключ уже есть — обновляем value и last_seen, first_seen оставляем, позицию
    в списке не меняем. Второй записи о том же факте появиться не должно:
    именно из дублей вырастает память, которой агент перестаёт доверять.
    Время приходит параметром: функция, зовущая time.time(), непроверяема.
    """
    raise NotImplementedError


def forget_stale(memory, now, ttl):
    """Выкинуть факты, которые не подтверждались дольше ttl. Вернуть НОВЫЙ список.

    memory = [{"key": "a", "value": "1", "first_seen": 0, "last_seen": 0}]
    forget_stale(memory, 100, ttl=200)  ->  тот же список (100 - 0 <= 200)
    forget_stale(memory, 100, ttl=50)   ->  []

    Ровно на границе (now - last_seen == ttl) факт ещё живой. Порядок
    оставшихся записей сохраняется.
    Без устаревания память превращается в свалку: старый факт про версию
    Python переживает апгрейд и уводит следующую сессию не туда.
    """
    raise NotImplementedError


def commit_state(fs, path, state, schema=STATE_SCHEMA):
    """Проверить состояние по схеме и записать атомарно. Вернуть НОВЫЙ fs.

    fs = commit_state({}, "agent_state.json", good_state)
    commit_state({}, "s.json", {"schema_version": 1})  ->  SchemaError

    Сначала валидация, потом запись — не наоборот. Плохая запись обязана быть
    отклонённой записью: файл состояния — источник истины, и чинить его
    руками потом дороже, чем отказать сейчас.
    """
    raise NotImplementedError


def load_state(fs, path, schema=STATE_SCHEMA):
    """Прочитать состояние из fs и проверить по схеме.

    load_state({"s.json": '{"schema_version": 1}'}, "s.json")  ->  SchemaError
    load_state({}, "s.json")                                   ->  KeyError

    Файл чужой версии не читают «как получится»: менеджер отказывается его
    загружать и отправляет через migrate_state. Отсутствующий файл — KeyError,
    а не пустое состояние по умолчанию: тихая подстановка пустышки стирает
    работу предыдущей сессии.
    """
    raise NotImplementedError


def migrate_state(state):
    """Поднять состояние до SCHEMA_VERSION. Вернуть НОВЫЙ словарь.

    Миграция v1 -> v2: поле blockers переименовано в risks.

    migrate_state({"schema_version": 1, "blockers": ["x"]})
        ->  {'schema_version': 2, 'risks': ['x']}
    migrate_state({"schema_version": 2, "risks": []})  ->  вход без изменений

    Миграция идемпотентна: второй прогон на уже поднятом состоянии ничего не
    меняет, иначе её нельзя вешать на старт каждой сессии.
    Неизвестная версия — SchemaError: лучше отказаться читать, чем угадать.
    """
    raise NotImplementedError
