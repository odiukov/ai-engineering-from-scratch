"""
Скрипты инициализации агента — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import json

# Требования рабочего места. В настоящем init-скрипте они лежат рядом с кодом,
# здесь — константами модуля.
REQUIRED_PYTHON = (3, 10)
REQUIRED_DEPS = ("pytest", "ruff")
REQUIRED_ENV = ("AGENT_WORKDIR",)

# Пороги. Возраст состояния и TTL замка — сутки, как в уроке.
STATE_MAX_AGE_SECONDS = 24 * 60 * 60
LOCK_TTL_SECONDS = 24 * 60 * 60
LKG_FILE_BUDGET = 50

# Файловую систему моделируем словарём путь -> содержимое. Никакого диска:
# init-скрипт обязан быть проверяемым, а тест, пишущий в файлы, — не тест.
STATE_PATH = "workdir/agent_state.json"
REPORT_PATH = "workdir/init_report.json"
LOCK_PATH = "workdir/prereqs.lock"

# Порядок проб в отчёте фиксирован: отчёт читают глазами и диффают между
# запусками, а диффать перетасованные строки невозможно.
PROBE_ORDER = ("runtime", "dependencies", "env", "state_freshness", "lkg_diff")

# Поля манифеста, из которых считается отпечаток.
CONFIG_FIELDS = ("python", "deps", "env", "test_command")


def probe_runtime(version, required=REQUIRED_PYTHON):
    """Проба версии рантайма. Вернуть {"name", "status", "detail"}.

    status — "pass", "warn" или "fail".

    probe_runtime((3, 12))["status"]  ->  "pass"
    probe_runtime((3, 9))["status"]   ->  "fail"

    version и required — кортежи ЧИСЕЛ, и сравниваются как кортежи. Соблазн
    сравнить строки "3.9" >= "3.10" даёт True: лексикографически девятка
    больше единицы. Именно так рождается «у меня всё работало» на неверной
    версии Python.
    """
    ok = tuple(version) >= tuple(required)
    return {
        "name": "runtime",
        "status": "pass" if ok else "fail",
        "detail": f"нужен python >= {tuple(required)}, есть {tuple(version)}",
    }


def probe_dependencies(installed, required=REQUIRED_DEPS):
    """Проба зависимостей: все ли required есть среди installed.

    probe_dependencies(["pytest", "ruff"])["status"]  ->  "pass"
    probe_dependencies(["pytest"])["status"]          ->  "fail"

    В detail перечисли ИМЕНА недостающих пакетов, отсортированные. Отчёт
    «зависимости не в порядке» без имён заставляет человека запускать пробу
    заново руками — а весь смысл init в том, чтобы одно место говорило всё
    сразу.

    Лишние установленные пакеты пробу не ломают: init проверяет достаточность,
    а не чистоту окружения.
    """
    have = set(installed)
    missing = sorted(dep for dep in required if dep not in have)
    return {
        "name": "dependencies",
        "status": "fail" if missing else "pass",
        "detail": f"нет пакетов: {missing}" if missing else f"на месте: {sorted(required)}",
    }


def probe_env(environ, required=REQUIRED_ENV):
    """Проба переменных окружения. Пустое значение считается отсутствующим.

    probe_env({"AGENT_WORKDIR": "/work"})["status"]  ->  "pass"
    probe_env({"AGENT_WORKDIR": ""})["status"]       ->  "fail"
    probe_env({})["status"]                          ->  "fail"

    Переменная, объявленная пустой строкой, — самая злая из возможных: код
    видит, что ключ есть, и падает уже в середине сессии с непонятным
    сообщением. Пробел тоже пустота.

    В detail попадают только ИМЕНА. Значение переменной в отчёт писать нельзя:
    init_report.json уходит в логи CI, а там лежат ключи.
    """
    missing = sorted(name for name in required if not str(environ.get(name, "")).strip())
    return {
        "name": "env",
        "status": "fail" if missing else "pass",
        "detail": f"нет переменных: {missing}" if missing else f"на месте: {sorted(required)}",
    }


def probe_state_freshness(state, now, max_age=STATE_MAX_AGE_SECONDS):
    """Проба свежести состояния прошлой сессии.

    state — словарь с полем "written_at" или None, если файла состояния нет.
    now передаётся параметром: проба, смотрящая на настоящие часы, не
    воспроизводится.

    probe_state_freshness(None, now=100)["status"]                  ->  "warn"
    probe_state_freshness({"written_at": 90}, now=100)["status"]    ->  "pass"
    probe_state_freshness({"written_at": 0}, now=10 ** 6)["status"] ->  "warn"

    Оба сомнительных случая — предупреждение, а не отказ. Отсутствие состояния
    это просто первый запуск, а старое состояние после упавшей сессии — повод
    подтвердить у человека, но не повод не стартовать. Отказом должно быть
    только то, что делает работу невозможной.

    state из будущего (now < written_at) -> ValueError: сломанные часы надо
    заметить, а не тихо посчитать возраст отрицательным и назвать свежим.
    """
    if state is None:
        return {"name": "state_freshness", "status": "warn", "detail": "состояния нет, первый запуск"}
    written_at = state["written_at"]
    if now < written_at:
        raise ValueError(f"состояние из будущего: written_at={written_at}, now={now}")
    age = now - written_at
    stale = age > max_age
    return {
        "name": "state_freshness",
        "status": "warn" if stale else "pass",
        "detail": f"состоянию {age} с (порог {max_age})",
    }


def probe_lkg_diff(changed_files, budget=LKG_FILE_BUDGET):
    """Проба расхождения с last-known-good: сколько файлов разошлось.

    changed_files — список путей или None, если базовая линия не закреплена.

    probe_lkg_diff(["a.py", "b.py"])["status"]      ->  "pass"
    probe_lkg_diff([f"f{i}.py" for i in range(51)])["status"]  ->  "fail"
    probe_lkg_diff(None)["status"]                  ->  "warn"

    Ровно budget файлов — ещё pass; отказ начинается с budget + 1. Смысл
    пробы в том, чтобы дрейф не накапливался между сессиями: каждая сессия
    сверяется с ОДНОЙ базовой линией, а не с результатом предыдущей.

    budget < 0 -> ValueError: такой бюджет запрещал бы любой diff, включая
    пустой, и стартовать стало бы невозможно.
    """
    if budget < 0:
        raise ValueError(f"бюджет расхождения не может быть отрицательным: {budget}")
    if changed_files is None:
        return {"name": "lkg_diff", "status": "warn", "detail": "базовая линия не закреплена"}
    count = len(changed_files)
    return {
        "name": "lkg_diff",
        "status": "fail" if count > budget else "pass",
        "detail": f"разошлось файлов: {count} (бюджет {budget})",
    }


def deps_fingerprint(config):
    """Отпечаток манифеста: 16 hex-символов sha256 от канонического вида.

    config — {"python": (3, 10), "deps": [...], "env": [...], "test_command": ...}.

    deps_fingerprint({"python": (3, 10), "deps": ["b", "a"], "env": [],
                      "test_command": "pytest"})
      ==  deps_fingerprint({"python": (3, 10), "deps": ["a", "b"], "env": [],
                            "test_command": "pytest"})

    Списки перед хешированием ОБЯЗАНЫ сортироваться: порядок строк в
    requirements не меняет окружение, а меняющийся от перестановки отпечаток
    сбрасывал бы кэш проб на каждом ровно-таком-же запуске.

    Добавили или убрали зависимость — отпечаток обязан измениться, иначе
    замок будет прикрывать устаревший результат.

    Нет любого поля из CONFIG_FIELDS -> ValueError.
    """
    missing = [field for field in CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"в манифесте нет полей: {missing}")
    canonical = json.dumps(
        {
            "python": list(config["python"]),
            "deps": sorted(config["deps"]),
            "env": sorted(config["env"]),
            "test_command": config["test_command"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def lock_is_fresh(lock, config, now, ttl=LOCK_TTL_SECONDS):
    """Можно ли доверять prereqs.lock и пропустить дорогие пробы.

    lock — {"fingerprint": ..., "written_at": ...} или None.

    lock_is_fresh(None, cfg, now=0)                                  ->  False
    lock_is_fresh({"fingerprint": f, "written_at": 0}, cfg, now=10)   ->  True
    lock_is_fresh({"fingerprint": f, "written_at": 0}, cfg, now=10**6) ->  False

    Три причины не доверять: замка нет, отпечаток манифеста не совпал,
    возраст дошёл до ttl. Ровно ttl — уже не свежий: полагаться на границу
    удобнее, когда она строгая.

    Замок из будущего (now < written_at) тоже не свежий. Лишний прогон проб
    стоит секунды, а доверие к записи со сломанных часов — потерянную сессию.

    Это тот же приём, что кэш слоёв в Docker: идемпотентная проба плюс хеш
    содержимого равно право пропустить работу.
    """
    if not lock:
        return False
    if lock.get("fingerprint") != deps_fingerprint(config):
        return False
    written_at = lock.get("written_at")
    if written_at is None or now < written_at:
        return False
    return now - written_at < ttl


def run_init(fs, snapshot, config, now, use_cache=True):
    """Весь init-прогон над файловой системой-словарём.

    snapshot — {"version", "installed", "environ", "changed_files"}: то, что
    скрипт узнаёт про машину. Состояние и замок читаются из fs.

    Вернуть {"fs", "report", "started", "skipped"}.
      report — {"timestamp": now, "probes": [...], "ok": bool, "blocking": [имена]};
      started — можно ли запускать агента;
      skipped — сработал ли короткий путь по свежему замку.

    Правила:
      * свежий замок и use_cache -> ничего не запускаем и НИЧЕГО не пишем:
        skipped=True, report=None, fs возвращается как есть;
      * есть проба со статусом "fail" -> started=False, замок НЕ обновляем;
        падать надо громко и в одном месте, иначе агент стартует на битом
        рабочем месте и выяснит это на сороковом шаге;
      * все пробы кроме warn зелёные -> пишем отчёт и замок, started=True.

    Идемпотентность: два прогона подряд дают одинаковый отчёт с точностью до
    timestamp. Именно она позволяет повесить скрипт на pre-task хук и в CI,
    не боясь, что второй запуск что-то испортит.

    Входной fs не мутировать — вернуть новый словарь.
    """
    fresh_fs = dict(fs)
    lock = json.loads(fresh_fs[LOCK_PATH]) if LOCK_PATH in fresh_fs else None
    if use_cache and lock_is_fresh(lock, config, now):
        return {"fs": fresh_fs, "report": None, "started": True, "skipped": True}

    state = json.loads(fresh_fs[STATE_PATH]) if STATE_PATH in fresh_fs else None
    probes = [
        probe_runtime(snapshot["version"], config["python"]),
        probe_dependencies(snapshot["installed"], config["deps"]),
        probe_env(snapshot["environ"], config["env"]),
        probe_state_freshness(state, now),
        probe_lkg_diff(snapshot["changed_files"]),
    ]
    blocking = [p["name"] for p in probes if p["status"] == "fail"]
    report = {
        "timestamp": now,
        "probes": probes,
        "ok": not blocking,
        "blocking": blocking,
    }
    # Отчёт пишем всегда: человеку нужно место, куда смотреть, и при падении
    # особенно. Замок — только на зелёном прогоне.
    fresh_fs[REPORT_PATH] = json.dumps(report, sort_keys=True, ensure_ascii=False)
    if blocking:
        return {"fs": fresh_fs, "report": report, "started": False, "skipped": False}
    fresh_fs[LOCK_PATH] = json.dumps(
        {"fingerprint": deps_fingerprint(config), "written_at": now},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {"fs": fresh_fs, "report": report, "started": True, "skipped": False}
