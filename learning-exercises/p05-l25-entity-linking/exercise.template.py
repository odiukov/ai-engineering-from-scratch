"""
Entity linking и разрешение неоднозначности

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l25-entity-linking
Разбор:  /check-code p05-l25-entity-linking
"""

import re


def tokenize(text):
    """Разбить текст на нормализованные токены: нижний регистр, без пунктуации.

    tokenize("Paris, France!")    ->  ["paris", "france"]
    tokenize("NBA 1998 Finals")   ->  ["nba", "1998", "finals"]
    tokenize("")                  ->  []

    Ловушка: пунктуацию надо отбрасывать. Иначе "france!" и "france" —
    два разных слова, и пересечение контекста с описанием сущности
    провалится на ровном месте.
    """
    raise NotImplementedError


def build_alias_index(pairs):
    """Инвертированный индекс alias -> список entity id из пар (alias, entity_id).

    build_alias_index([("Paris", "Q90"), ("paris", "Q663094")])
        ->  {"paris": ["Q90", "Q663094"]}
    build_alias_index([("Apple", "Q312"), ("Apple", "Q312")])
        ->  {"apple": ["Q312"]}

    Ключ всегда в нижнем регистре: в тексте встретится и "Paris", и "PARIS".
    Порядок внутри списка сохраняется — он и есть prior: первой идёт самая
    популярная сущность, к ней система и скатится при равных счётах.
    Повторы одного id не дублируются.
    """
    raise NotImplementedError


def candidates(alias_index, mention):
    """Кандидаты для упоминания. Пустой список, если алиас неизвестен.

    candidates({"paris": ["Q90"]}, "Paris")   ->  ["Q90"]
    candidates({"paris": ["Q90"]}, " PARIS ") ->  ["Q90"]
    candidates({"paris": ["Q90"]}, "Berlin")  ->  []

    Возвращай КОПИЮ списка: вызывающий код не должен уметь испортить индекс.

    Здесь потолок всей системы. Чего нет в списке кандидатов, того не выберет
    никакой disambiguator — это и есть mention recall.
    """
    raise NotImplementedError


def jaccard(a, b):
    """Коэффициент Жаккара двух наборов токенов: |пересечение| / |объединение|.

    jaccard(["a", "b"], ["b", "c"])  ->  0.3333333333333333
    jaccard(["a"], ["a"])            ->  1.0
    jaccard([], [])                  ->  0.0

    Ловушка: пустое объединение — деление на ноль. Договорённость: 0.0.

    Это игрушечная замена косинусу на эмбеддингах: в BLINK тут был бы
    dot-product двух нормированных векторов.
    """
    raise NotImplementedError


def disambiguate(mention, context, alias_index, descriptions):
    """Выбрать кандидата, чьё описание больше всего пересекается с контекстом.

    Вернуть кортеж (entity_id, score). Для неизвестного упоминания — (None, 0.0).

    disambiguate("Jordan", "the NBA finals", index, desc)  ->  ("Q41421", 0.09...)
    disambiguate("Nobody", "any text", index, desc)        ->  (None, 0.0)

    Счёт — jaccard(токены контекста, токены описания кандидата).

    Ловушка: при равных счётах побеждает ПЕРВЫЙ кандидат, то есть самый
    популярный. Сравнивай строго больше (>), а не >=. Это и есть popularity
    bias: на бессмысленном контексте система уверенно вернёт баскетболиста.
    """
    raise NotImplementedError


def link_with_nil(mention, context, alias_index, descriptions, threshold=0.05):
    """То же, что disambiguate, но слабую привязку заменяет на NIL (None).

    link_with_nil("Jordan", "the NBA finals", index, desc, 0.05)  ->  "Q41421"
    link_with_nil("Jordan", "zzz qqq", index, desc, 0.05)         ->  None
    link_with_nil("Nobody", "any text", index, desc)              ->  None

    Возвращает один entity_id или None, без счёта.

    Зачем: часть упоминаний просто нет в KB (новые компании, малоизвестные
    люди). Система обязана уметь сказать "не знаю" вместо уверенной чуши.
    """
    raise NotImplementedError


def mention_recall(examples, alias_index):
    """Доля примеров, где правильная сущность вообще попала в список кандидатов.

    Каждый пример — кортеж (surface, context, gold_id); context здесь не нужен.

    mention_recall([("Paris", "...", "Q90")], {"paris": ["Q90"]})  ->  1.0
    mention_recall([("Paris", "...", "Q90")], {})                  ->  0.0
    mention_recall([], index)                                      ->  0.0

    Это пол всего пайплайна. Отчёт без mention recall бессмыслен: 99%
    disambiguation при 80% recall — это 80% системы, а не 99%.
    """
    raise NotImplementedError


def evaluate_linker(examples, alias_index, descriptions):
    """Три числа сразу: mention_recall, disambiguation_accuracy, pipeline_accuracy.

    Вернуть dict с этими тремя ключами.

    disambiguation_accuracy считается ТОЛЬКО по примерам, где gold попал в
    кандидаты (иначе мы штрафуем disambiguator за чужую ошибку). Если таких
    примеров нет — 0.0.
    pipeline_accuracy — доля правильных ответов по всем примерам.

    evaluate_linker([("Paris", "capital of France", "Q90")], index, desc)
        ->  {"mention_recall": 1.0, "disambiguation_accuracy": 1.0,
             "pipeline_accuracy": 1.0}

    Проверь себя: pipeline_accuracy обязана равняться произведению двух
    других. Если не сходится — где-то посчитал не по той выборке.
    """
    raise NotImplementedError
