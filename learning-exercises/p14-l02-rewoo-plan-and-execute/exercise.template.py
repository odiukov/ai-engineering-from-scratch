"""
ReWOO: план отдельно, исполнение отдельно

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l02-rewoo-plan-and-execute
Разбор:  /check-code p14-l02-rewoo-plan-and-execute
"""

import re

STEP_RE = re.compile(r"^#(E\d+)\s*=\s*(\w+)\[(.*)\]$")
REF_RE = re.compile(r"#(E\d+)")


def parse_plan(text):
    """Разобрать текст плана в список шагов [{'id', 'tool', 'arg'}, ...].

    parse_plan("Plan: найти столицу\\n#E1 = search[capital of France]")
        ->  [{'id': 'E1', 'tool': 'search', 'arg': 'capital of France'}]
    parse_plan("")  ->  []

    Строки, не начинающиеся с '#E', — это комментарии планировщика
    ("Plan: ..."), их пропускаем молча. А вот строка, которая НАЧИНАЕТСЯ
    с '#E' и при этом не разбирается, — ValueError: тихо проглотить кривой
    шаг хуже, чем упасть, потому что решатель потом получит дыру в evidence.
    """
    raise NotImplementedError


def find_references(text):
    """Список id, на которые ссылается строка, без повторов, в порядке появления.

    find_references("population of #E1")     ->  ['E1']
    find_references("#E2 minus #E1 plus #E2") ->  ['E2', 'E1']
    find_references("capital of France")     ->  []

    Порядок сохраняем ради воспроизводимости сообщений об ошибках: set()
    в Python неупорядочен, и тест на текст ошибки начнёт мигать.
    """
    raise NotImplementedError


def substitute_references(text, evidence):
    """Подставить в строку собранные evidence вместо #E1, #E2, ...

    substitute_references("population of #E1", {"E1": "Paris"})
        ->  'population of Paris'
    substitute_references("population of #E9", {"E1": "Paris"})
        ->  'population of #E9'

    Неизвестная ссылка остаётся как есть — так ошибка доедет до инструмента
    видимой строкой, а не превратится в пустую подстановку, которую потом
    не отличить от честного пустого результата.
    """
    raise NotImplementedError


def validate_plan(steps, tool_names):
    """Список претензий к плану. Пустой список — план валиден.

    validate_plan([{'id': 'E1', 'tool': 'search', 'arg': 'x'}], {'search'})
        ->  []
    validate_plan([{'id': 'E1', 'tool': 'nope', 'arg': 'x'}], {'search'})
        ->  ["E1: неизвестный инструмент 'nope'"]

    Проверяем три вещи, ровно как в скилле из урока: инструмент существует,
    id не повторяется, ссылка указывает на УЖЕ определённый шаг. Третья
    проверка ловит и циклы, и опечатки в номере разом: план — это DAG,
    ссылка вперёд в нём невозможна по определению.
    """
    raise NotImplementedError


def topological_order(steps):
    """Переставить шаги так, чтобы зависимости шли раньше зависимых.

    topological_order([{'id': 'E2', 'arg': 'x #E1'}, {'id': 'E1', 'arg': 'y'}])
        ->  [{'id': 'E1', ...}, {'id': 'E2', ...}]

    Результат не зависит от порядка на входе — в этом весь смысл: планировщик
    выдал DAG, а не последовательность, и исполнитель обязан сам разложить
    его по уровням. Шаги без зависимостей внутри одного уровня сохраняют
    исходный относительный порядок — именно они и уходят в параллель.

    Неразрешимая ссылка или цикл — ValueError.
    """
    raise NotImplementedError


def run_workers(steps, tools):
    """Выполнить шаги в порядке зависимостей и собрать evidence {id: строка}.

    run_workers([{'id': 'E1', 'tool': 'up', 'arg': 'ab'}], {'up': str.upper})
        ->  {'E1': 'AB'}

    Каждый worker получает УЖЕ подставленный аргумент — свой и только свой,
    без истории мыслей. Это и есть экономия токенов из статьи.

    Ошибка инструмента становится строкой evidence, а не исключением:
    решатель увидит её в контексте плана и деградирует аккуратно. Это
    вторая половина обещания ReWOO — локализация отказа по узлу.
    """
    raise NotImplementedError


def run_rewoo(question, planner, tools, solver):
    """Полный проход ReWOO: planner -> workers -> solver.

    Возвращает {'plan', 'evidence', 'answer', 'llm_calls'}.

    planner — callable(question) -> текст плана.
    solver  — callable(question, evidence) -> итоговый ответ.

    run_rewoo("столица?", lambda q: "#E1 = search[capital of France]",
              {"search": ...}, lambda q, e: e["E1"])
        ->  {'plan': [...], 'evidence': {'E1': 'Paris'}, 'answer': 'Paris',
             'llm_calls': 2}

    Ключевое свойство, ради которого всё затевалось: llm_calls всегда 2,
    сколько бы шагов ни было в плане. ReAct на том же задании сходил бы
    к модели N+1 раз.

    План проверяется ДО исполнения: если validate_plan вернул претензии —
    ValueError, и ни один инструмент не вызывается. Дешевле упасть на
    плане, чем оплатить половину DAG и упереться в несуществующий tool.
    """
    raise NotImplementedError


def prompt_sizes(question, steps, mode):
    """Размеры промптов (в символах) по каждому обращению к модели.

    steps — список шагов с ключами 'tool', 'arg', 'evidence'.
    mode  — 'react' или 'rewoo'.

    prompt_sizes("q", [{"tool": "s", "arg": "a", "evidence": "e"}], "rewoo")
        ->  [1, 4]
    len(prompt_sizes(q, steps, "react")) == len(steps) + 1

    ReAct тащит в каждый следующий промпт всю историю: N+1 обращений,
    размеры строго растут. ReWOO — ровно два обращения: планировщик видит
    только вопрос, решатель — вопрос плюс план плюс evidence.

    Другой mode — ValueError: опечатка в имени режима не должна тихо
    посчитать не то.
    """
    raise NotImplementedError
