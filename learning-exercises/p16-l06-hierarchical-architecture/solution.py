"""
Иерархическая архитектура и её отказ — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Эмпирический потолок из урока: три уровня уже прячут большинство ошибок
# от наблюдаемости, поэтому глубже двух не ходим.
DEPTH_CEILING = 2


class OrgError(Exception):
    """Дерево подчинения сломано: цикл, два начальника или недостижимый узел.

    Собственный класс, а не RuntimeError: NotImplementedError — потомок
    RuntimeError, и тест на родителя зеленел бы на пустой заготовке.
    """


def build_org(edges):
    """Дерево подчинения из пар (начальник, подчинённый).

    Порядок подчинённых сохраняется — от него зависит порядок обхода.
    Листья попадают в результат с пустым списком.

    build_org([("top", "a"), ("top", "b"), ("a", "w1")])
        ->  {"top": ["a", "b"], "a": ["w1"], "b": [], "w1": []}
    build_org([])  ->  {}

    Структуру никто не проверяет: build_org честно соберёт и цикл. За
    проверку отвечает validate_org.
    """
    org = {}
    for manager, report in edges:
        org.setdefault(manager, []).append(report)
        org.setdefault(report, [])  # лист тоже узел, просто бездетный
    return org


def validate_org(org, root):
    """Обход дерева от корня. Возвращает узлы в порядке обхода сверху вниз.

    Ловит три поломки, все через OrgError:
      * корня нет в дереве,
      * цикл в подчинении (a -> b -> a),
      * узел с двумя начальниками или недостижимый от корня узел.

    validate_org({"top": ["a"], "a": []}, "top")   ->  ["top", "a"]
    validate_org({"a": ["b"], "b": ["a"]}, "a")    ->  OrgError

    Цикл в подчинении — не абстракция: это тот самый reconciliation loop,
    когда два sub-manager'а пересылают друг другу несогласованную задачу.
    """
    if root not in org:
        raise OrgError(f"root not in org: {root}")

    order = []
    seen = set()

    def walk(node, path):
        if node in path:
            raise OrgError("cycle in org: " + " -> ".join(path + [node]))
        if node in seen:
            raise OrgError(f"node has two managers: {node}")
        seen.add(node)
        order.append(node)
        for child in org.get(node, []):
            walk(child, path + [node])

    walk(root, [])
    unreachable = set(org) - seen
    if unreachable:
        raise OrgError("unreachable from root: " + ", ".join(sorted(unreachable)))
    return order


def depth(org, root):
    """Глубина дерева: сколько уровней подчинения ниже корня.

    depth({"top": []}, "top")                      ->  0
    depth({"top": ["w"], "w": []}, "top")          ->  1
    depth({"t": ["s"], "s": ["w"], "w": []}, "t")  ->  2

    Считаются рёбра, а не узлы. Один менеджер и один работник — это один
    уровень, а не два.

    Дерево с циклом уронит функцию в бесконечную рекурсию: сначала
    validate_org, потом всё остальное.
    """
    children = org.get(root, [])
    if not children:
        return 0
    return 1 + max(depth(org, child) for child in children)


def leaves(org, root):
    """Листья дерева слева направо. Только они делают работу.

    leaves({"top": ["a", "b"], "a": ["w1", "w2"], "b": [], "w1": [], "w2": []},
           "top")   ->  ["w1", "w2", "b"]

    Внутренние узлы только планируют, делегируют и сводят результат. Если
    менеджер что-то «сделал сам» — это уже не иерархия.
    """
    children = org.get(root, [])
    if not children:
        return [root]
    found = []
    for child in children:
        found.extend(leaves(org, child))
    return found


def provenance(org, root, node):
    """Путь от корня до узла — цепочка провенанса «кто кому это передал».

    provenance({"t": ["s"], "s": ["w"], "w": []}, "t", "w")  ->  ["t", "s", "w"]
    provenance(org, "t", "ghost")                            ->  OrgError

    Урок требует провенанс на каждом синтезе: без цепочки до листа нельзя
    понять, откуда взялось утверждение в финальном ответе.
    """
    def walk(current, path):
        path = path + [current]
        if current == node:
            return path
        for child in org.get(current, []):
            found = walk(child, path)
            if found is not None:
                return found
        return None

    path = walk(root, [])
    if path is None:
        raise OrgError(f"no path from {root} to {node}")
    return path


def too_deep(org, root, ceiling=DEPTH_CEILING):
    """Узлы, до которых от корня дальше ceiling рёбер. Пустой список — норма.

    too_deep({"t": ["s"], "s": ["w"], "w": []}, "t")             ->  []
    too_deep({"t": ["s"], "s": ["m"], "m": ["w"], "w": []}, "t") ->  ["w"]

    Это правило «Cap tree depth at 2» из чеклиста урока в исполняемом
    виде: список — это ровно те узлы, чьи ошибки уже не видно сверху.
    """
    deep = []

    def walk(node, level):
        if level > ceiling:
            deep.append(node)
        for child in org.get(node, []):
            walk(child, level + 1)

    walk(root, 0)
    return deep


def delegate(org, root, chosen_labels, required_labels):
    """Разбор поручения верхнего менеджера: куда ушло, куда не ушло.

    Возвращает dict с тремя списками в порядке аргументов:
      "delegated" — ветки, которые действительно подчинены корню,
      "unknown"   — названные ветки, которых в подчинении нет,
      "uncovered" — нужные ветки, которые менеджер не назвал.

    delegate(org, "vp", ["eng", "finance"], ["eng", "legal"])
        ->  {"delegated": ["eng", "finance"], "unknown": [],
             "uncovered": ["legal"]}

    Именно так выглядит decomposition drift: "finance" — настоящая ветка,
    работа сделана честно, ошибку не видно нигде, кроме "uncovered". А
    юридический вопрос пользователя остался без ответа.
    """
    children = org.get(root, [])
    return {
        "delegated": [label for label in chosen_labels if label in children],
        "unknown": [label for label in chosen_labels if label not in children],
        "uncovered": [label for label in required_labels if label not in chosen_labels],
    }


def aggregate(org, root, leaf_answers, summarize=None):
    """Сборка результата снизу вверх: лист отдаёт ответ, узел сводит детей.

    summarize(node, parts) -> строка. По умолчанию "[node] p1 | p2".
    Лист без ответа отдаёт "[no answer from <лист>]".

    org = {"t": ["s"], "s": ["w1", "w2"], "w1": [], "w2": []}
    aggregate(org, "t", {"w1": "A", "w2": "B"})
        ->  "[t] [s] A | B"

    Верхний узел видит СВОДКИ детей, а не сырые ответы листьев — в этом
    вся выгода иерархии и весь её риск: смысл искажается на каждом уровне.
    """
    # сводка по умолчанию живёт внутри: снаружи она никому не нужна, а
    # отдельная функция уровня модуля стала бы ещё одной заготовкой
    combine = summarize if summarize is not None else (
        lambda node, parts: f"[{node}] " + " | ".join(parts)
    )
    children = org.get(root, [])
    if not children:
        return leaf_answers.get(root, f"[no answer from {root}]")
    # рекурсия вниз, сборка на обратном ходе — вот и «снизу вверх»
    parts = [aggregate(org, child, leaf_answers, summarize) for child in children]
    return combine(root, parts)
