"""
Reflexion: обучение словами вместо градиентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Задача-игрушка из урока: подобрать три числа из [LOW..HIGH] с нужной суммой.
LOW = 1
HIGH = 9

# Сколько рефлексий помещается в эпизодическую память. Урок называет
# переполнение "memory rot": старые записи вытесняют полезные.
MEMORY_LIMIT = 6

# Порог для эвристического оценщика: длиннее — трактуем как неэффективный ход.
MAX_STEPS = 5


def binary_evaluator(attempt, target):
    """Скалярный оценщик: (успех, отклонение суммы от цели).

    binary_evaluator([9, 9, 2], 20)  ->  (True, 0)
    binary_evaluator([1, 1, 1], 20)  ->  (False, -17)

    Знак отклонения несёт смысл: минус — недобрали, плюс — перебрали. Именно
    он потом превращается в словесную рефлексию. Голое "не получилось"
    рефлексии не даёт, из него нечего сказать следующей попытке.
    """
    total = sum(attempt)
    return total == target, total - target


def heuristic_evaluator(actions, max_steps=MAX_STEPS):
    """Эвристический оценщик: список сигнатур провала в траектории.

    heuristic_evaluator(["a", "b"])            ->  []
    heuristic_evaluator(["a", "a"])            ->  ["stuck: действие 'a' повторилось"]
    heuristic_evaluator(list("abcdef"))        ->  ['inefficient: 6 шагов > 5']

    Второй тип оценщика из урока: ground truth не нужен, нужны заранее
    известные признаки беды. Дёшево, работает без разметки, ловит ровно то,
    что успели формализовать — и ничего сверх.
    """
    flags = []
    for a, b in zip(actions, actions[1:]):
        if a == b:
            flags.append(f"stuck: действие {a!r} повторилось")
            break
    if len(actions) > max_steps:
        flags.append(f"inefficient: {len(actions)} шагов > {max_steps}")
    return flags


def reflect(attempt, delta):
    """Само-рефлексия: одна строка о том, ЧТО именно пошло не так.

    reflect([1, 1, 1], -17)  ->  'сумма 3 меньше цели на 17: бери числа крупнее'
    reflect([9, 9, 9], 7)    ->  'сумма 27 больше цели на 7: бери числа мельче'
    reflect([9, 9, 2], 0)    ->  'получилось'

    Разница между полезной и бесполезной рефлексией — в конкретике.
    "Будь внимательнее" не меняет следующую попытку; "недобрал 17, бери
    крупнее" меняет. Это и есть весь verbal reinforcement.
    """
    total = sum(attempt)
    if delta < 0:
        return f"сумма {total} меньше цели на {-delta}: бери числа крупнее"
    if delta > 0:
        return f"сумма {total} больше цели на {delta}: бери числа мельче"
    return "получилось"


def add_reflection(memory, reflection, max_len=MEMORY_LIMIT):
    """Добавить рефлексию в эпизодическую память, вернув НОВЫЙ список.

    add_reflection([], {"trial": 1})            ->  [{'trial': 1}]
    add_reflection([{"trial": i} for i in range(6)], {"trial": 9}, max_len=6)
        ->  6 элементов, самый старый вытеснен

    Память ограничена сверху: иначе двадцатый прогон тащит в промпт
    девятнадцать чужих рефлексий и стоит дороже, чем даёт.

    Ловушка: не меняй входной список. Тесты сравнивают ДО и ПОСЛЕ, а в бою
    на общую память смотрят сразу несколько попыток.
    """
    updated = list(memory) + [reflection]
    if len(updated) > max_len:
        # вытесняем с головы: самая старая рефлексия наименее актуальна
        updated = updated[len(updated) - max_len:]
    return updated


def expire_reflections(memory, now, ttl):
    """Выбросить рефлексии старше ttl попыток относительно now.

    expire_reflections([{"trial": 1}, {"trial": 9}], now=10, ttl=3)
        ->  [{'trial': 9}]
    expire_reflections([{"trial": 9}], now=10, ttl=3)
        ->  [{'trial': 9}]

    Момент времени приходит параметром, а не берётся из time.time(): иначе
    поведение зависит от того, когда запустили, и тест перестаёт быть тестом.

    Это лечение memory rot из урока: разовая неудача сети не должна
    висеть в промпте вечно.
    """
    return [r for r in memory if now - r["trial"] <= ttl]


def memory_prompt(memory):
    """Собрать эпизодическую память в кусок промпта для следующей попытки.

    memory_prompt([])  ->  '(нет прошлых рефлексий)'
    memory_prompt([{"trial": 1, "text": "недобрал"}])
        ->  '- попытка 1: недобрал'

    Явная заглушка для пустой памяти важнее, чем кажется: пустая строка
    в промпте сливается с соседним блоком, и модель считает чужой текст
    своей рефлексией.
    """
    if not memory:
        return "(нет прошлых рефлексий)"
    return "\n".join(f"- попытка {r['trial']}: {r['text']}" for r in memory)


def actor(memory, low=LOW, high=HIGH):
    """Актёр: следующая попытка по эпизодической памяти.

    actor([])  ->  [1, 1, 1]           (памяти нет — самая наивная догадка)
    actor([{"attempt": [1, 1, 1], "delta": -17}])  ->  [9, 9, 2]

    Актёр читает ПОСЛЕДНЮЮ рефлексию и раскидывает нужную поправку по
    позициям, упираясь в границы [low, high]. Без памяти он навсегда
    залипает на первой догадке — в этом и смысл базового прогона из урока:
    сравнить агента с памятью и без.

    Это буквально "градиентный шаг словами": направление берётся из текста
    прошлой неудачи, а не из производной.
    """
    if not memory:
        return [low, low, low]
    last = memory[-1]
    attempt = list(last["attempt"])
    need = -last["delta"]  # на столько надо изменить сумму
    for i in range(len(attempt)):
        if need == 0:
            break
        # сколько ещё можно сдвинуть эту позицию в нужную сторону
        room = (high - attempt[i]) if need > 0 else (low - attempt[i])
        change = room if abs(room) < abs(need) else need
        attempt[i] += change
        need -= change
    return attempt


def run_reflexion(target, max_trials=4, use_memory=True):
    """Полный цикл Reflexion: актёр -> оценщик -> рефлексия -> память -> ...

    Возвращает список попыток
    [{'trial', 'attempt', 'success', 'delta', 'reflection'}, ...].

    run_reflexion(20, use_memory=True)   ->  2 попытки, последняя успешная
    run_reflexion(20, use_memory=False)  ->  4 одинаковые неудачные попытки

    Успех обрывает цикл: лишний прогон после победы — просто сожжённые токены.
    С выключенной памятью актёр видит пустой буфер на каждой попытке и
    повторяет одно и то же — ровно то, что урок называет базовой линией.
    """
    memory = []
    trials = []
    for t in range(1, max_trials + 1):
        attempt = actor(memory if use_memory else [])
        success, delta = binary_evaluator(attempt, target)
        text = reflect(attempt, delta)
        trials.append({"trial": t, "attempt": attempt, "success": success,
                       "delta": delta, "reflection": text})
        if success:
            break
        memory = add_reflection(
            memory, {"trial": t, "attempt": attempt, "delta": delta, "text": text}
        )
    return trials
