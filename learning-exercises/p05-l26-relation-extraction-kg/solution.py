"""
Relation extraction и сборка knowledge graph — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def extract_triples(text, patterns):
    """Вытащить триплеты (subject, relation, object) регулярками.

    patterns — список пар (regex, relation). В regex обязаны быть именованные
    группы `s` (subject) и `o` (object).

    extract_triples(
        "Tim founded Apple.",
        [("(?P<s>[A-Za-z]+) founded (?P<o>[A-Za-z]+)", "founded")],
    )  ->  [("Tim", "founded", "Apple")]

    extract_triples("nothing here", [...])  ->  []

    Порядок результата: сначала все совпадения первого паттерна (слева
    направо по тексту), потом второго и так далее. Тесты на это опираются.

    Hearst-паттерны до сих пор живут в продакшене именно потому, что их
    можно отладить глазами.
    """
    triples = []
    for pattern, relation in patterns:
        # finditer, а не findall: нужны именованные группы, а не кортежи
        for match in re.finditer(pattern, text):
            triples.append((match.group("s"), relation, match.group("o")))
    return triples


def verify_span(text, surface, span):
    """Правда ли, что text[start:end] в точности равен surface.

    verify_span("Tim Cook works", "Tim Cook", (0, 8))   ->  True
    verify_span("Tim Cook works", "Steve Jobs", (0, 8)) ->  False
    verify_span("Tim Cook works", "Tim Cook", (0, 999)) ->  False
    verify_span("Tim Cook works", "", (0, 0))           ->  False

    Ловушка: срез в Python не падает на выходе за границы, "abc"[0:999] это
    просто "abc". Проверяй границы САМ, иначе провалишь смысл шага verify.
    Пустой surface считаем неподтверждённым.

    Это ядро шага Verify из AEVS: LLM охотно возвращает правдоподобные
    сущности, которых в тексте нет.
    """
    start, end = span
    if not surface:
        return False
    if not (0 <= start <= end <= len(text)):
        return False
    return text[start:end] == surface


def filter_verified(text, extractions):
    """Оставить только триплеты, у которых обе сущности реально есть в тексте.

    extractions — список dict-ов вида
    {"subject": ..., "subject_span": (a, b),
     "relation": ..., "object": ..., "object_span": (c, d)}

    Вернуть список кортежей (subject, relation, object) для выживших.

    filter_verified("Tim founded Apple", [
        {"subject": "Tim", "subject_span": (0, 3), "relation": "founded",
         "object": "Apple", "object_span": (12, 17)},
    ])  ->  [("Tim", "founded", "Apple")]

    Триплет проходит, только если подтвердились ОБА спана. Полу-подтверждённых
    не бывает: галлюцинированный object делает весь факт мусором.
    """
    kept = []
    for item in extractions:
        subject_ok = verify_span(text, item["subject"], item["subject_span"])
        object_ok = verify_span(text, item["object"], item["object_span"])
        if subject_ok and object_ok:
            kept.append((item["subject"], item["relation"], item["object"]))
    return kept


def hallucination_rate(text, extractions):
    """Доля триплетов, отвергнутых проверкой спанов.

    hallucination_rate("Tim founded Apple", [один валидный])   ->  0.0
    hallucination_rate("Tim founded Apple", [один выдуманный]) ->  1.0
    hallucination_rate("любой текст", [])                      ->  0.0

    Пустой вход — 0.0, а не деление на ноль.

    Это метрика, ради которой шаг verify вообще существует: её меряют до и
    после включения проверки и показывают аудиту.
    """
    if not extractions:
        return 0.0
    survived = len(filter_verified(text, extractions))
    return (len(extractions) - survived) / len(extractions)


def canonicalize(relation, relation_map):
    """Свести поверхностную формулировку отношения к id онтологии.

    Сравнение — по нижнему регистру и без хвостовых пробелов.

    canonicalize("  Was Born In ", {"was born in": "P19"})  ->  "P19"
    canonicalize("hangs out with", {"was born in": "P19"})  ->  None

    None означает "в онтологии такого нет" — такие отношения либо
    выбрасывают, либо отправляют на ручную разметку. Молча пропускать их в
    граф нельзя: open IE выдаёт "was born in", "came from", "is a native of"
    как три разных ребра, и граф перестаёт быть запрашиваемым.
    """
    return relation_map.get(relation.strip().lower())


def canonicalize_triples(triples, relation_map):
    """Прогнать список триплетов через canonicalize, выбросив неотображённые.

    canonicalize_triples(
        [("Tim", "was born in", "Alabama"), ("Tim", "likes", "jazz")],
        {"was born in": "P19"},
    )  ->  [("Tim", "P19", "Alabama")]

    Субъект и объект не трогаем: их канонизирует entity linking (урок 25),
    это отдельная задача.
    """
    result = []
    for subject, relation, obj in triples:
        canonical = canonicalize(relation, relation_map)
        if canonical is not None:
            result.append((subject, canonical, obj))
    return result


def build_graph(triples):
    """Собрать граф: dict узел -> список рёбер (relation, object).

    build_graph([("Tim", "P108", "Apple"), ("Tim", "P19", "Alabama")])
        ->  {"Tim": [("P108", "Apple"), ("P19", "Alabama")]}

    build_graph([("Tim", "P108", "Apple"), ("Tim", "P108", "Apple")])
        ->  {"Tim": [("P108", "Apple")]}

    Порядок рёбер — порядок первого появления. Точные дубликаты
    схлопываются: один и тот же факт, встреченный в десяти документах, — это
    одно ребро, а не десять.
    """
    graph = {}
    for subject, relation, obj in triples:
        edges = graph.setdefault(subject, [])
        edge = (relation, obj)
        # линейная проверка: рёбер на узел единицы-десятки, отдельный set
        # ради дедупликации сломал бы порядок первого появления
        if edge not in edges:
            edges.append(edge)
    return graph


def neighbors(graph, node, relation=None):
    """Рёбра узла, при желании отфильтрованные по типу отношения.

    neighbors(g, "Tim")               ->  [("P108", "Apple"), ("P19", "Alabama")]
    neighbors(g, "Tim", "P108")       ->  [("P108", "Apple")]
    neighbors(g, "неизвестный узел")  ->  []

    Возвращай новый список, а не внутренний: граф должен пережить любого
    вызывающего.

    Это атом любого RAG-over-KG: один хоп по графу.
    """
    edges = graph.get(node, [])
    return [(rel, obj) for rel, obj in edges if relation is None or rel == relation]
