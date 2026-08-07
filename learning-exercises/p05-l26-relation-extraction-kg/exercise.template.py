"""
Relation extraction и сборка knowledge graph

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l26-relation-extraction-kg
Разбор:  /check-code p05-l26-relation-extraction-kg
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def hallucination_rate(text, extractions):
    """Доля триплетов, отвергнутых проверкой спанов.

    hallucination_rate("Tim founded Apple", [один валидный])   ->  0.0
    hallucination_rate("Tim founded Apple", [один выдуманный]) ->  1.0
    hallucination_rate("любой текст", [])                      ->  0.0

    Пустой вход — 0.0, а не деление на ноль.

    Это метрика, ради которой шаг verify вообще существует: её меряют до и
    после включения проверки и показывают аудиту.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def canonicalize_triples(triples, relation_map):
    """Прогнать список триплетов через canonicalize, выбросив неотображённые.

    canonicalize_triples(
        [("Tim", "was born in", "Alabama"), ("Tim", "likes", "jazz")],
        {"was born in": "P19"},
    )  ->  [("Tim", "P19", "Alabama")]

    Субъект и объект не трогаем: их канонизирует entity linking (урок 25),
    это отдельная задача.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def neighbors(graph, node, relation=None):
    """Рёбра узла, при желании отфильтрованные по типу отношения.

    neighbors(g, "Tim")               ->  [("P108", "Apple"), ("P19", "Alabama")]
    neighbors(g, "Tim", "P108")       ->  [("P108", "Apple")]
    neighbors(g, "неизвестный узел")  ->  []

    Возвращай новый список, а не внутренний: граф должен пережить любого
    вызывающего.

    Это атом любого RAG-over-KG: один хоп по графу.
    """
    raise NotImplementedError
