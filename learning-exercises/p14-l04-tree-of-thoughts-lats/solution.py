"""
Tree of Thoughts и LATS: рассуждение как поиск — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import itertools
import math

# Игра 24 из статьи Yao et al.: четырьмя числами и знаками + - * / получить 24.
TARGET = 24
OPS = ("+", "-", "*", "/")


def make_node(state, trace=()):
    """Узел дерева поиска: состояние, путь к нему и счётчики MCTS.

    make_node((6.0, 4.0))
        ->  {'state': (6.0, 4.0), 'trace': (), 'visits': 0,
             'value_sum': 0.0, 'children': []}

    Узел — это "мысль" из статьи: связный промежуточный шаг, а не токен.
    visits и value_sum нужны только LATS, ToT их не трогает; держим в одном
    типе, чтобы дерево у обоих алгоритмов было одно и то же.

    Ловушка: children обязан быть СВОИМ списком у каждого узла. Общий
    список по умолчанию склеит всё дерево в один узел.
    """
    return {
        # Корень нормализуем так же, как дочерние состояния: иначе порядок
        # входных чисел меняет порядок ничьих в beam search.
        "state": tuple(sorted(state, reverse=True)),
        "trace": tuple(trace),
        "visits": 0,
        "value_sum": 0.0,
        "children": [],
    }


def expand(node):
    """Все дочерние мысли: взять пару чисел, применить операцию, свернуть.

    len(expand(make_node((6.0, 4.0))))   ->  6
    expand(make_node((5.0,)))            ->  []   (сворачивать нечего)

    Деление на ноль пропускаем — это не "мысль", а исключение.
    Состояние храним отсортированным по убыванию: (24.0, 1.0) и (1.0, 24.0)
    для игры 24 — одно и то же, и без нормализации дерево раздувается вдвое.
    """
    children = []
    state = node["state"]
    if len(state) < 2:
        return children
    for i, j in itertools.combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        candidates = [
            (a, "+", b, a + b),
            (a, "-", b, a - b),
            (a, "*", b, a * b),
            (b, "-", a, b - a),
        ]
        if b != 0:
            candidates.append((a, "/", b, a / b))
        if a != 0:
            candidates.append((b, "/", a, b / a))
        for left, op, right, v in candidates:
            rest = [s for k, s in enumerate(state) if k not in (i, j)]
            new_state = tuple(sorted(rest + [v], reverse=True))
            step = f"{left}{op}{right}={v}"
            children.append(make_node(new_state, node["trace"] + (step,)))
    return children


def _closest(state, target):
    """Расстояние от ближайшего числа состояния до target."""
    return min(abs(v - target) for v in state)


def value(node, target=TARGET):
    """Self-evaluation: точный лист либо промах после одного шага вперёд.

    value(make_node((24.0,)))       ->  1.0
    value(make_node((20.0,)))       ->  -0.04
    value(make_node((23.0, 5.0)))   ->  -0.04

    Оценка неполного состояния смотрит, чего можно достичь ЕЩЁ ОДНОЙ
    операцией. Наивная близость текущих чисел ошибочно считает (24, 4)
    идеальным состоянием, хотя после обязательного последнего действия
    получатся 96, 28, 20, 6, 1/6 или -20, но не 24.

    В статье оценку выдаёт промпт ("sure / likely / impossible"), здесь она
    символьная — так тест проверяет логику поиска, а не качество подсказки.
    """
    state = node["state"]
    if len(state) == 1:
        gap = abs(state[0] - target)
        return 1.0 if gap < 1e-6 else -gap / 100.0
    reachable = [child["state"] for child in expand(node)]
    best_gap = min((_closest(s, target) for s in reachable),
                   default=_closest(state, target))
    return -best_gap / 100.0


def beam_search(root, target=TARGET, width=5, depth=3):
    """ToT в виде beam search: расширяем, оцениваем, оставляем лучших width.

    Возвращает (лучший узел, сколько узлов раскрыли).

    beam_search(make_node((8.0, 3.0, 1.0, 1.0)))[0]['state']  ->  (24.0,)

    Результат зависит только от МНОЖЕСТВА чисел, а не от их порядка на
    входе: поиск обязан находить одну и ту же ветку при любом обходе.

    Точное решение обрывает поиск сразу — платить за оставшиеся уровни
    незачем, ToT и без того жжёт в 100–1000 раз больше токенов, чем CoT.
    """
    frontier = [root]
    expansions = 0
    best = root
    for _ in range(depth):
        scored = []
        for node in frontier:
            for child in expand(node):
                expansions += 1
                scored.append((value(child, target), child))
        if not scored:
            break
        scored.sort(key=lambda pair: -pair[0])
        if scored[0][0] > value(best, target):
            best = scored[0][1]
        if scored[0][0] >= 1.0:
            return scored[0][1], expansions
        frontier = [n for _, n in scored[:width]]
    return best, expansions


def uct(parent_visits, child, c=1.4):
    """Формула выбора в MCTS: Q + c * sqrt(ln N / n).

    uct(10, {"visits": 0, "value_sum": 0.0})              ->  inf
    uct(1, {"visits": 1, "value_sum": 0.5}, c=0.0)        ->  0.5

    Непосещённый ребёнок получает inf: MCTS обязан хотя бы раз заглянуть
    в каждую ветку, прежде чем судить о ней.

    Первое слагаемое — эксплуатация (что уже показало себя),
    второе — исследование (что редко пробовали). c крутит баланс.
    """
    if child["visits"] == 0:
        return float("inf")
    q = child["value_sum"] / child["visits"]
    # max(...,1): ln(0) не существует, а первый заход в корень бывает при N=0
    return q + c * math.sqrt(math.log(max(parent_visits, 1)) / child["visits"])


def select_path(root, c=1.4):
    """Фаза Select: спуск от корня до листа по максимуму UCT.

    select_path(node_without_children)  ->  [node]

    Возвращается ВЕСЬ путь, а не только лист: по нему потом поедет
    backpropagate. Хранить у узла ссылку на родителя тоже можно, но путь
    списком проще и не создаёт циклов в структуре.
    """
    path = [root]
    node = root
    while node["children"]:
        node = max(node["children"], key=lambda ch: uct(node["visits"], ch, c))
        path.append(node)
    return path


def backpropagate(path, reward):
    """Фаза Backpropagate: разослать награду вверх по пути.

    node = make_node((1.0,)); backpropagate([node], 1.0)
        ->  node['visits'] == 1, node['value_sum'] == 1.0

    После двух наград 1.0 и 0.0 средняя оценка узла (value_sum / visits)
    равна 0.5 — MCTS хранит именно среднее, а не последнее значение.
    Функция меняет узлы на месте и ничего не возвращает.
    """
    for node in path:
        node["visits"] += 1
        node["value_sum"] += reward


def mcts(root, iterations, rng, target=TARGET, c=1.4, rollout_depth=3):
    """LATS в миниатюре: select -> expand -> simulate -> backpropagate.

    Возвращает ребёнка корня с наибольшим числом посещений — то есть ветку,
    которую поиск счёл лучшей. Если корень так и не раскрыли, возвращает корень.

    rng — источник случайности, обязательно параметром (random.Random(0)):
    прогон обязан воспроизводиться, иначе тест на выбор ветки будет мигать.

    Симуляция — случайный докат до конца из выбранного узла; награда за
    лист уходит наверх и меняет оценки всех предков. Именно поэтому ветка
    с хорошим листом побеждает независимо от того, в каком порядке её нашли.
    """
    for _ in range(iterations):
        path = select_path(root, c)
        leaf = path[-1]
        # раскрываем только уже посещённый лист: иначе дерево растёт вширь
        # быстрее, чем накапливается статистика
        if leaf["visits"] > 0 and len(leaf["state"]) > 1:
            leaf["children"] = expand(leaf)
            if leaf["children"]:
                leaf = rng.choice(leaf["children"])
                path.append(leaf)
        current = leaf
        for _ in range(rollout_depth):
            options = expand(current)
            if not options:
                break
            current = rng.choice(options)
        backpropagate(path, value(current, target))
    if not root["children"]:
        return root
    return max(root["children"], key=lambda ch: ch["visits"])
