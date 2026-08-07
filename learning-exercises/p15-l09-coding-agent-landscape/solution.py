"""
Ландшафт кодовых агентов: сравнение по осям возможностей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def pass_rate(tasks):
    """Доля решённых задач. tasks — список dict с ключом "solved".

    pass_rate([{"solved": True}, {"solved": False}])  ->  0.5
    pass_rate([])                                     ->  0.0

    Пустой список даёт 0.0, а не ZeroDivisionError: «агент не решил ничего»
    — осмысленный ответ, падение — нет.
    """
    if not tasks:
        return 0.0
    return sum(1 for t in tasks if t["solved"]) / len(tasks)


def score_excluding_easy(tasks, min_lines):
    """Доля решённых среди задач, требующих не меньше min_lines строк правки.

    tasks — список dict с ключами "lines" и "solved".

    score_excluding_easy([{"lines": 1, "solved": True},
                          {"lines": 20, "solved": False}], 10)  ->  0.0

    Зачем: в SWE-bench Verified 161 задача из 500 требует правки в 1-2
    строки. Общий процент тянут вверх именно они, а продакшн-распределение
    ближе к SWE-bench Pro (10+ строк).

    Если после фильтра не осталось ни одной задачи — вернуть 0.0.
    Считай через pass_rate, не переписывай деление заново.
    """
    hard = [t for t in tasks if t["lines"] >= min_lines]
    return pass_rate(hard)


def rank_agents(scores):
    """Лидерборд: список (имя, оценка), от лучшей оценки к худшей.

    rank_agents({"aider": 0.5, "cline": 0.6})  ->  [("cline", 0.6), ("aider", 0.5)]
    rank_agents({"b": 0.5, "a": 0.5})          ->  [("a", 0.5), ("b", 0.5)]

    Ничьи разбиваются по имени по алфавиту. Это не украшательство: без
    детерминированного tie-break два запуска на одних и тех же числах дают
    разный порядок, и «агент поднялся на позицию» перестаёт что-то значить.
    """
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rank_changes(before, after):
    """Как перетасовался лидерборд: {имя: на сколько позиций поднялся}.

    before и after — то, что вернул rank_agents.

    rank_changes([("a", 1), ("b", 0)], [("b", 1), ("a", 0)])  ->  {"a": -1, "b": 1}

    Плюс — поднялся вверх (индекс стал меньше), минус — опустился. Имена,
    которых нет в обоих списках, пропускаются: сравнивать агента с самим
    собой из другого набора нельзя.
    """
    pos_before = {name: i for i, (name, _score) in enumerate(before)}
    pos_after = {name: i for i, (name, _score) in enumerate(after)}
    return {
        name: pos_before[name] - pos_after[name]
        for name in pos_before
        if name in pos_after
    }


def scaffold_delta(baseline, scaffolded):
    """Прибавка от scaffold-а в процентных пунктах (знак сохраняется).

    scaffold_delta(43.2, 59.8)  ->  16.6
    scaffold_delta(59.8, 43.2)  ->  -16.6

    Это и есть заголовок урока: одна и та же Claude Sonnet 4.5 даёт 43.2 в
    SWE-agent v1 и 59.8 в автономном scaffold-е Cline. Веса те же, продукт —
    это петля вокруг них.

    Разность именно процентных пунктов, а не отношение: «выросло на 38%»
    и «выросло на 16.6 пункта» — разные утверждения, путать нельзя.
    """
    return scaffolded - baseline


def simulate_scaffold(bug_count, files_per_action):
    """Сколько ходов и какой blast radius у scaffold-а с заданной ёмкостью действия.

    files_per_action = 1 — это JSON tool call: одно действие правит один
    файл. Больше единицы — это CodeAct: один сниппет правит несколько.

    Вернуть dict {"turns": ..., "blast": ...}.

    simulate_scaffold(3, 1)  ->  {"turns": 4, "blast": 1}
    simulate_scaffold(3, 3)  ->  {"turns": 2, "blast": 3}
    simulate_scaffold(0, 5)  ->  {"turns": 1, "blast": 0}

    Последний ход всегда есть — это завершающее "done", даже если чинить
    было нечего. Отсюда +1.

    blast — фактически задетые файлы худшего действия, а не заявленная
    ёмкость: на двух багах CodeAct с ёмкостью 10 трогает два файла, и врать
    про десять в отчёте о риске нельзя.

    files_per_action <= 0 — это не scaffold, а ошибка конфигурации:
    подними ValueError.
    """
    if files_per_action <= 0:
        raise ValueError("files_per_action must be positive")
    turns = math.ceil(bug_count / files_per_action) + 1
    return {"turns": turns, "blast": min(files_per_action, bug_count)}


def blast_radius(actions):
    """Худший blast radius трассы: максимум файлов, задетых ОДНИМ действием.

    actions — список dict с ключом "files".

    blast_radius([{"files": ["a.py"]}, {"files": ["a.py", "b.py"]}])  ->  2
    blast_radius([])                                                  ->  0

    Считать надо максимум, а не сумму: аудит отвечает на вопрос «что
    успеет натворить одно неудачное действие до того, как его заметят»,
    а не «сколько файлов тронуто за сессию».
    """
    if not actions:
        return 0
    return max(len(a["files"]) for a in actions)


def compare_axes(profile_a, profile_b):
    """Сравнение двух scaffold-ов по осям: {ось: "a" | "b" | "tie" | "unknown"}.

    profile_a и profile_b — dict {ось: число, больше — лучше}.

    compare_axes({"retrieval": 3}, {"retrieval": 1})  ->  {"retrieval": "a"}
    compare_axes({"retrieval": 1}, {"verifier": 1})
        ->  {"retrieval": "unknown", "verifier": "unknown"}

    Ось, измеренная только у одного из двух, даёт "unknown", а не победу.
    Это главный вывод урока про бенчмарки: неизмеренная ось — не нулевая
    ось, и подставлять туда ноль значит выдумывать результат.

    Ключи возвращаются в алфавитном порядке — чтобы отчёт был стабилен.
    """
    out = {}
    for axis in sorted(set(profile_a) | set(profile_b)):
        if axis not in profile_a or axis not in profile_b:
            out[axis] = "unknown"
        elif profile_a[axis] > profile_b[axis]:
            out[axis] = "a"
        elif profile_a[axis] < profile_b[axis]:
            out[axis] = "b"
        else:
            out[axis] = "tie"
    return out
