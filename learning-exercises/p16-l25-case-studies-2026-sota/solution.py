"""
Разборы кейсов и состояние дел 2026 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Правило «масштабируй усилие под сложность запроса» из инженерного разбора
# системы Research у Anthropic: простой запрос — один агент, средний — три,
# сложное исследование — десять и больше.
SUBAGENTS = {"simple": 1, "medium": 3, "complex": 10}


class CyclicRouting(Exception):
    """В графе ролей нашёлся цикл — маршрут не построить.

    СВОЙ класс, а не RuntimeError: NotImplementedError наследуется от
    RuntimeError, и проверка «падает на цикле» зеленела бы на заготовке.
    """


def linear_fit(xs, ys):
    """Метод наименьших квадратов для прямой. Вернуть (наклон, свободный член).

    linear_fit([0, 1, 2], [1, 3, 5])  ->  (2.0, 1.0)
    linear_fit([0, 1], [5, 5])        ->  (0.0, 5.0)

    slope = sum((x-mx)(y-my)) / sum((x-mx)^2), intercept = my - slope*mx.

    Нужно, чтобы посчитать долю объяснённой дисперсии — ту самую, из-за
    которой Anthropic пишет, что 80% разброса на BrowseComp объясняется
    одним лишь объёмом потраченных токенов.

    Ловушка: если все x одинаковы, знаменатель нулевой. Это ValueError, а не
    «наклон ноль»: по вертикальному облаку точек прямую не провести.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("для прямой нужно минимум две точки")
    mx = sum(xs) / n
    my = sum(ys) / n
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("все x совпали: наклон не определён")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    return slope, my - slope * mx


def r_squared(xs, ys):
    """Доля дисперсии y, объяснённая линейной зависимостью от x.

    r_squared([0, 1, 2], [1, 3, 5])        ->  1.0   (точки лежат на прямой)
    r_squared([0, 1, 2, 3], [0, 3, 1, 2])  ->  0.16  (прямая почти ничего не ловит)

    R^2 = 1 - SS_res / SS_tot, где SS_res — сумма квадратов остатков от
    подобранной прямой, SS_tot — от среднего.

    Ровно эта величина стоит за фразой «80% разброса объясняется объёмом
    токенов»: не «токены — причина», а «одной переменной хватает, чтобы
    предсказать четыре пятых разброса». Причинность отсюда не следует.

    Постоянный y — ValueError: объяснять нечего, SS_tot равен нулю.
    """
    slope, intercept = linear_fit(xs, ys)
    my = sum(ys) / len(ys)
    ss_total = sum((y - my) ** 2 for y in ys)
    if ss_total == 0:
        raise ValueError("y постоянен: дисперсии нет, объяснять нечего")
    ss_residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_residual / ss_total


def relative_improvement(baseline, system):
    """Относительный прирост системы над базой.

    relative_improvement(1.0, 1.902)  ->  0.902
    relative_improvement(0.5, 0.25)   ->  -0.5
    relative_improvement(0.0, 0.3)    ->  ValueError

    Именно так читается «+90.2% над одноагентным Opus 4»: это отношение к
    базе, а не разница в процентных пунктах. Путать их — самый частый
    способ раздуть результат вдвое.

    Нулевая база — ValueError: делить не на что, и любой прирост от нуля
    формально бесконечен.
    """
    if baseline == 0:
        raise ValueError("нулевая база: относительный прирост не определён")
    return (system - baseline) / baseline


def verification_budget(total_tokens, tax):
    """Разбить токенный бюджет на работу и на верификацию.

    verification_budget(100000, 0.25)  ->  (75000.0, 25000.0)
    verification_budget(100000, 0.0)   ->  (100000.0, 0.0)

    Независимый верификатор стоит 20-30% бюджета — и Anthropic, и MetaGPT
    платят этот налог осознанно: без выделенной роли проверяющего система
    наблюдаемо галлюцинирует.

    tax вне [0, 1) — ValueError. Единица означала бы систему, которая
    только проверяет и ничего не делает.
    """
    if not 0.0 <= tax < 1.0:
        raise ValueError("доля на верификацию должна быть в [0, 1)")
    verification = total_tokens * tax
    return (total_tokens - verification, verification)


def subagent_budget(complexity):
    """Сколько субагентов поднимать под запрос данной сложности.

    subagent_budget("simple")   ->  1
    subagent_budget("medium")   ->  3
    subagent_budget("complex")  ->  10
    subagent_budget("huge")     ->  ValueError

    Правило из системы Research: усилие масштабируется под сложность, а не
    «всегда десять агентов». Мультиагентность стоит 15x токенов — платить
    их за простой факт-чек бессмысленно.

    Неизвестная сложность — ValueError. Молчаливый дефолт «пусть будет
    один» превращает сложные запросы в тихо плохие ответы.
    """
    if complexity not in SUBAGENTS:
        raise ValueError("неизвестная сложность: %r" % (complexity,))
    return SUBAGENTS[complexity]


def topological_order(dag):
    """Порядок ролей в DAG маршрутизации. Вернуть список узлов.

    dag — {узел: список преемников}. Алгоритм Кана, при выборе из нескольких
    готовых узлов берём лексикографически меньший: порядок обязан быть
    воспроизводимым.

    topological_order({"pm": ["arch"], "arch": ["eng"], "eng": []})
        ->  ["pm", "arch", "eng"]
    topological_order({"a": ["b"], "b": ["a"]})  ->  CyclicRouting

    Так MacNet (arXiv:2406.07155) разгоняет ChatDev до 1000+ агентов:
    маршрут известен заранее и считается офлайн, а не выясняется в чате.

    Ловушка: узел может встречаться только в списках преемников и не иметь
    своего ключа. Такие узлы тоже участвуют.
    """
    nodes = set(dag)
    for successors in dag.values():
        nodes.update(successors)
    indegree = {n: 0 for n in nodes}
    for successors in dag.values():
        for s in successors:
            indegree[s] += 1

    ready = sorted(n for n in nodes if indegree[n] == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for s in sorted(dag.get(node, [])):
            indegree[s] -= 1
            if indegree[s] == 0:
                # вставка с сохранением сортировки: узлов мало, bisect излишен
                ready.append(s)
        ready.sort()
    if len(order) != len(nodes):
        raise CyclicRouting("в графе ролей есть цикл")
    return order


def critical_path(dag):
    """Длина самой длинной цепочки ролей в узлах.

    critical_path({"pm": ["arch"], "arch": ["eng"], "eng": []})  ->  3
    critical_path({"pm": ["a", "b", "c"], "a": [], "b": [], "c": []})  ->  2
    critical_path({})  ->  0

    Считается за один проход по топологическому порядку: к моменту, когда
    доходим до узла, все его предшественники уже посчитаны.

    Это и есть аргумент MacNet за DAG вместо чата: ширина растёт до тысяч
    узлов, а глубина — а с ней и число последовательных LLM-вызовов —
    почти нет. Чат же последователен по определению.
    """
    order = topological_order(dag)
    depth = {n: 1 for n in order}
    for node in order:
        for s in dag.get(node, []):
            depth[s] = max(depth[s], depth[node] + 1)
    return max(depth.values()) if depth else 0


def retirable_versions(runs, current):
    """Версии рантайма, которые можно погасить. Вернуть отсортированный список.

    runs — список {"version": ..., "done": ...}. Версия гасится, если она не
    текущая и у неё не осталось незавершённых прогонов.

    retirable_versions([{"version": "v1", "done": True},
                        {"version": "v1", "done": False},
                        {"version": "v2", "done": True}], "v3")  ->  ["v2"]

    Rainbow deployment: агент живёт часами, и убивать его на каждом деплое
    нельзя. Старые версии рантайма доживают вместе с новыми, пока их
    прогоны не досчитают.

    Ловушка: текущую версию гасить нельзя никогда, даже если прямо сейчас
    у неё нет ни одного активного прогона — новые придут через секунду.
    """
    versions = {run["version"] for run in runs}
    busy = {run["version"] for run in runs if not run["done"]}
    return sorted(v for v in versions if v != current and v not in busy)
