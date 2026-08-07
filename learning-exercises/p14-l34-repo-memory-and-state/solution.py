"""
Память репозитория и durable state — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json
import re

# Файловую систему моделируем словарём путь -> содержимое. Ничего на диск не
# пишем: тест, который трогает диск, перестаёт быть воспроизводимым.
TEMP_SUFFIX = ".tmp"

# Текущая версия схемы состояния. Файл более старой версии не читают, а мигрируют.
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
    if "type" in schema:
        wanted = schema["type"]
        types = [wanted] if isinstance(wanted, str) else list(wanted)
        ok = False
        for name in types:
            if name == "object" and isinstance(value, dict):
                ok = True
            elif name == "array" and isinstance(value, list):
                ok = True
            elif name == "string" and isinstance(value, str):
                ok = True
            # bool отсекаем явно: isinstance(True, int) == True
            elif name == "integer" and isinstance(value, int) and not isinstance(value, bool):
                ok = True
            elif name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                ok = True
            elif name == "null" and value is None:
                ok = True
        if not ok:
            raise SchemaError(f"{path}: ожидался {wanted}, получен {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} не входит в {schema['enum']}")

    if "pattern" in schema and isinstance(value, str):
        if not re.match(schema["pattern"], value):
            raise SchemaError(f"{path}: {value!r} не подходит под /{schema['pattern']}/")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: нет обязательного поля {key!r}")
        properties = schema.get("properties", {})
        if properties:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise SchemaError(f"{path}: лишние поля {extra}")
            for key, sub in properties.items():
                if key in value:
                    validate(value[key], sub, f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")


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
    new_fs = dict(fs)
    temp = path + TEMP_SUFFIX
    new_fs[temp] = content
    if crash_after_temp:
        # ключевая часть: fs, который отдали на вход, не тронут вообще,
        # а новый словарь наружу не уходит — падение не оставляет следов
        raise OSError(f"падение после записи {temp}")
    new_fs[path] = new_fs.pop(temp)
    return new_fs


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
    new_memory = [dict(item) for item in memory]
    for item in new_memory:
        if item["key"] == key:
            item["value"] = value
            item["last_seen"] = now
            return new_memory
    new_memory.append(
        {"key": key, "value": value, "first_seen": now, "last_seen": now}
    )
    return new_memory


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
    return [dict(item) for item in memory if now - item["last_seen"] <= ttl]


def commit_state(fs, path, state, schema=STATE_SCHEMA):
    """Проверить состояние по схеме и записать атомарно. Вернуть НОВЫЙ fs.

    fs = commit_state({}, "agent_state.json", good_state)
    commit_state({}, "s.json", {"schema_version": 1})  ->  SchemaError

    Сначала валидация, потом запись — не наоборот. Плохая запись обязана быть
    отклонённой записью: файл состояния — источник истины, и чинить его
    руками потом дороже, чем отказать сейчас.
    """
    validate(state, schema)
    return atomic_write(fs, path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def load_state(fs, path, schema=STATE_SCHEMA):
    """Прочитать состояние из fs и проверить по схеме.

    load_state({"s.json": '{"schema_version": 1}'}, "s.json")  ->  SchemaError
    load_state({}, "s.json")                                   ->  KeyError

    Файл чужой версии не читают «как получится»: менеджер отказывается его
    загружать и отправляет через migrate_state. Отсутствующий файл — KeyError,
    а не пустое состояние по умолчанию: тихая подстановка пустышки стирает
    работу предыдущей сессии.
    """
    if path not in fs:
        raise KeyError(path)
    state = json.loads(fs[path])
    validate(state, schema)
    return state


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
    version = state.get("schema_version")
    if version == SCHEMA_VERSION:
        return dict(state)
    if version != 1:
        raise SchemaError(f"нет миграции для schema_version={version!r}")
    migrated = dict(state)
    migrated["risks"] = migrated.pop("blockers", [])
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated
