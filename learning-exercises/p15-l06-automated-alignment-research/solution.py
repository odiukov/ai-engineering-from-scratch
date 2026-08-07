"""
Автоматизированные исследования алаймента: форум вне песочницы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Anthropic запустила параллельные Automated Alignment Researchers (Claude Opus
4.6), каждого в своей песочнице, а находки складывала в общий форум, хранилище
которого лежит СНАРУЖИ любой песочницы. Смысл ровно один: агент не может
удалить или переписать собственную запись.

Собираем эту конструкцию руками:

    распределение задач -> агент решает -> запись уходит в append-only цепочку
    -> verify_forum() ловит любую тихую правку задним числом

Плюс второй сюжет отчёта: свободная декомпозиция («вот цель, разбирайся сам»)
даёт выше потолок, чем предписанный пошаговый workflow, но и разброс шире, и
аудит дороже.

Запись форума — словарь:
    {"author": "AAR-A", "task": "weak-to-strong", "regime": "free",
     "result": 0.62, "prev_hash": ..., "my_hash": ...}
"""

import hashlib

# Длина усечённого хеша. 16 hex-символов = 64 бита: для учебной цепочки хватает,
# для настоящего аудита берут полный sha256.
HASH_LEN = 16

# Хеш "до первой записи". Без него первая запись ни к чему не привязана, и
# её можно подменить целиком, не порвав цепочку.
GENESIS = "0" * HASH_LEN

# Предписанный workflow: узкий разброс, низкий потолок.
FIXED_SPREAD = 0.25
# Свободная декомпозиция: сдвиг вверх, но и хвост в обе стороны.
FREE_MEAN = 0.15
FREE_SD = 0.22


def record_hash(record, prev_hash):
    """Хеш записи, привязанный к предыдущему звену цепочки.

    Собери каноническую строку "author|task|regime|result|prev_hash", где
    result отформатирован как "%.3f", возьми sha256 и обрежь до HASH_LEN.

    record_hash({"author": "A", "task": "t", "regime": "free", "result": 0.5},
                GENESIS)   ->  16-символьная строка, одинаковая при каждом вызове

    Три обязательных свойства:
      * детерминизм. Никакой соли, никакого времени внутри — иначе проверить
        цепочку через год будет нечем;
      * prev_hash ВХОДИТ в подпись. Иначе записи можно переставлять местами:
        каждая по отдельности сойдётся, а порядок событий поедет;
      * посторонние ключи в record на подпись не влияют. Подписываем то, что
        объявлено содержимым записи, а не всё, что агент туда дописал.
    """
    payload = "{author}|{task}|{regime}|{result:.3f}|{prev}".format(
        author=record["author"],
        task=record["task"],
        regime=record["regime"],
        result=record["result"],
        prev=prev_hash,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LEN]


def append_record(forum, record):
    """Дописать запись в конец форума. Вернуть НОВЫЙ список.

    append_record([], {"author": "A", "task": "t", "regime": "free",
                       "result": 0.5})
        ->  [{... "prev_hash": GENESIS, "my_hash": <хеш>}]

    Первая запись ссылается на GENESIS, каждая следующая — на my_hash
    предыдущей. Так получается цепочка, а не куча.

    Возвращаем новый список и новый словарь записи. Append-only — это в первую
    очередь про то, что уже лежащее не меняется: если функция правит вход на
    месте, «форум до» и «форум после» перестают существовать по отдельности, и
    сравнить их при разборе инцидента не с чем.
    """
    prev = forum[-1]["my_hash"] if forum else GENESIS
    entry = dict(record)
    entry["prev_hash"] = prev
    entry["my_hash"] = record_hash(entry, prev)
    return [*forum, entry]


def verify_forum(forum):
    """Проверить цепочку. Вернуть список индексов испорченных записей.

    verify_forum(чистый_форум)               ->  []
    verify_forum(форум_с_правкой_в_записи_3)  ->  [3]

    Идём с начала, держим ожидаемый prev. Запись битая, если её prev_hash не
    совпал с ожидаемым ИЛИ её my_hash не совпал с пересчитанным.

    Тонкость, которую надо понять, а не обойти: после испорченной записи мы
    продолжаем от её СОХРАНЁННОГО my_hash, а не от пересчитанного. Поэтому
    тихая правка одного поля даёт ровно один индекс — точку, где цепочка
    порвана. Если продолжать от пересчитанного, покраснеет весь хвост, и
    отчёт «испорчено 47 записей» перестанет указывать на место.
    """
    broken = []
    prev = GENESIS
    for index, record in enumerate(forum):
        if record.get("prev_hash") != prev or record.get("my_hash") != record_hash(record, prev):
            broken.append(index)
        prev = record.get("my_hash")
    return broken


def tamper_record(forum, index, delta):
    """Тихо подправить результат записи, не трогая хеши. Вернуть НОВЫЙ форум.

    Это модель того, что делает скомпрометированный AAR: переписывает свой
    неудачный замер, чтобы отчёт выглядел лучше.

    tamper_record(forum, 0, 0.5)  ->  копия форума, где result записи 0 выше
                                      на 0.5, а my_hash прежний

    verify_forum() на результате обязан вернуть [index].

    Функция копирует форум и правит копию. Тест, в котором «до» и «после» —
    один и тот же объект, ничего не докажет.
    """
    if not 0 <= index < len(forum):
        raise IndexError("нет такой записи в форуме")
    edited = [dict(record) for record in forum]
    edited[index]["result"] = edited[index]["result"] + delta
    return edited


def allocate(tasks, agents):
    """Разложить задачи по агентам по кругу. Вернуть {агент: [задачи]}.

    allocate(["a", "b", "c"], ["X", "Y"])  ->  {"X": ["a", "c"], "Y": ["b"]}
    allocate([], ["X"])                    ->  {"X": []}

    Никакого rng: распределение обязано быть детерминированным, иначе два
    прогона «с одинаковыми настройками» окажутся разными экспериментами, и
    сравнение режимов ничего не значит.

    Две ловушки:
      * пустой список агентов — ValueError, а не тихий пустой план: задачи бы
        просто исчезли;
      * повторяющиеся имена агентов — тоже ValueError. Второй "X" затрёт
        список первого, и часть задач потеряется без единого сообщения.
    """
    if not agents:
        raise ValueError("некому раздавать задачи")
    if len(set(agents)) != len(agents):
        raise ValueError("имена агентов должны быть уникальны")
    plan = {agent: [] for agent in agents}
    for position, task in enumerate(tasks):
        plan[agents[position % len(agents)]].append(task)
    return plan


def solve_task(rng, base, regime):
    """Результат агента по задаче со сложностью base.

    "fixed" — предписанный workflow: base + равномерный шум [0, FIXED_SPREAD).
    "free"  — свободная декомпозиция: base + нормальный шум (FREE_MEAN, FREE_SD).

    solve_task(rng, 0.4, "fixed")  ->  число в [0.40, 0.65)
    solve_task(rng, 0.4, "free")   ->  обычно около 0.55, но бывает и 0.2, и 0.9

    В этом и весь компромисс из отчёта Anthropic. Предписанный workflow
    ограничивает AAR сверху: ниже базы не упадёт, но и выше потолка не
    прыгнет. Свободная декомпозиция даёт и рекорд, и провал — а вместе с ними
    несанкционированную декомпозицию, неожиданное использование инструментов
    и дрейф цели. Дороже аудит, выше потолок.

    Неизвестный режим — ValueError. Опечатка в имени режима не должна тихо
    попадать в одну из веток.
    """
    if regime == "fixed":
        return base + rng.random() * FIXED_SPREAD
    if regime == "free":
        return base + rng.gauss(FREE_MEAN, FREE_SD)
    raise ValueError(f"неизвестный режим: {regime!r}")


def run_forum(rng, tasks, agents, regime):
    """Прогнать всех агентов по их задачам и сложить находки в форум.

    tasks — список пар (имя задачи, базовая сложность).
    Вернуть готовую цепочку записей.

    run_forum(rng, [("t1", 0.4), ("t2", 0.3)], ["A", "B"], "fixed")
        ->  два звена: A решал t1, B решал t2

    Каждая задача решается ровно один раз — распределение идёт через
    allocate(). Порядок записей задан порядком агентов, а не порядком, в
    котором словарь плана перечисляет ключи: форум обязан читаться одинаково
    при том же seed.

    verify_forum() на свежем форуме всегда возвращает пустой список. Если это
    не так, ломать нечего — цепочка собрана неправильно с самого начала.
    """
    difficulty = dict(tasks)
    plan = allocate([name for name, _ in tasks], agents)
    forum = []
    for agent in agents:
        for name in plan[agent]:
            forum = append_record(forum, {
                "author": agent,
                "task": name,
                "regime": regime,
                "result": solve_task(rng, difficulty[name], regime),
            })
    return forum


def regime_summary(forum):
    """Свести форум: по задачам и в целом.

    Ключи: "per_task" ({задача: {"mean", "max", "min", "count"}}),
    "overall_mean", "records".

    regime_summary([запись(t1, 0.4), запись(t1, 0.6), запись(t2, 0.5)])
        ->  overall_mean 0.5, то есть (0.5 + 0.5) / 2, а НЕ (0.4 + 0.6 + 0.5) / 3

    Обрати внимание на overall_mean: это среднее СРЕДНИХ ПО ЗАДАЧАМ, а не
    среднее по записям. Разница не косметическая — задача, по которой
    отчиталось трое агентов, иначе перевесила бы задачу с одним отчётом, и
    «средний результат режима» начал бы зависеть от того, как раздали работу.

    Пустой форум — ValueError: сводки ни о чём не бывает.
    """
    if not forum:
        raise ValueError("форум пуст, сводить нечего")
    grouped = {}
    for record in forum:
        grouped.setdefault(record["task"], []).append(record["result"])
    per_task = {
        task: {
            "mean": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "count": len(values),
        }
        for task, values in grouped.items()
    }
    means = [stats["mean"] for stats in per_task.values()]
    return {
        "per_task": per_task,
        "overall_mean": sum(means) / len(means),
        "records": len(forum),
    }
