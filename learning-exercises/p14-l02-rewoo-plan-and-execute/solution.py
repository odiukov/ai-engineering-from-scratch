"""
ReWOO: план отдельно, исполнение отдельно — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re

# Строка плана в нотации статьи ReWOO (arXiv:2305.18323):
#     #E1 = search[capital of France]
STEP_RE = re.compile(r"^#(E\d+)\s*=\s*(\w+)\[(.*)\]$")

# Ссылка на результат прошлого шага: #E1, #E2, ...
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
    steps = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#E"):
            continue
        m = STEP_RE.match(line)
        if m is None:
            raise ValueError(f"malformed plan line: {line!r}")
        steps.append({"id": m.group(1), "tool": m.group(2), "arg": m.group(3)})
    return steps


def find_references(text):
    """Список id, на которые ссылается строка, без повторов, в порядке появления.

    find_references("population of #E1")     ->  ['E1']
    find_references("#E2 minus #E1 plus #E2") ->  ['E2', 'E1']
    find_references("capital of France")     ->  []

    Порядок сохраняем ради воспроизводимости сообщений об ошибках: set()
    в Python неупорядочен, и тест на текст ошибки начнёт мигать.
    """
    seen = []
    for rid in REF_RE.findall(text):
        if rid not in seen:
            seen.append(rid)
    return seen


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
    return REF_RE.sub(lambda m: evidence.get(m.group(1), m.group(0)), text)


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
    errors = []
    known = set()
    for step in steps:
        sid = step["id"]
        if step["tool"] not in tool_names:
            errors.append(f"{sid}: неизвестный инструмент {step['tool']!r}")
        if sid in known:
            errors.append(f"{sid}: id повторяется")
        for ref in find_references(step["arg"]):
            if ref not in known:
                errors.append(f"{sid}: ссылка на #{ref}, который ещё не посчитан")
        known.add(sid)
    return errors


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
    ordered = []
    known = set()
    pending = list(steps)
    while pending:
        rest = []
        progress = False
        for step in pending:
            if all(r in known for r in find_references(step["arg"])):
                ordered.append(step)
                known.add(step["id"])
                progress = True
            else:
                rest.append(step)
        if not progress:
            # ни один шаг не стал разрешимым за целый проход — дальше не будет
            raise ValueError(f"цикл или висячая ссылка: {[s['id'] for s in rest]}")
        pending = rest
    return ordered


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
    evidence = {}
    for step in topological_order(steps):
        arg = substitute_references(step["arg"], evidence)
        fn = tools.get(step["tool"])
        if fn is None:
            evidence[step["id"]] = f"error: unknown tool {step['tool']!r}"
            continue
        try:
            evidence[step["id"]] = str(fn(arg))
        except Exception as e:
            evidence[step["id"]] = f"error: {type(e).__name__}: {e}"
    return evidence


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
    plan_text = planner(question)
    steps = parse_plan(plan_text)
    errors = validate_plan(steps, set(tools))
    if errors:
        raise ValueError("невалидный план: " + "; ".join(errors))
    evidence = run_workers(steps, tools)
    return {
        "plan": steps,
        "evidence": evidence,
        "answer": solver(question, evidence),
        "llm_calls": 2,
    }


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
    if mode == "react":
        sizes = []
        carried = 0
        for step in steps:
            sizes.append(len(question) + carried)
            carried += len(step["tool"]) + len(step["arg"]) + len(step["evidence"])
        sizes.append(len(question) + carried)
        return sizes
    if mode == "rewoo":
        plan_chars = sum(len(s["tool"]) + len(s["arg"]) for s in steps)
        evidence_chars = sum(len(s["evidence"]) for s in steps)
        return [len(question), len(question) + plan_chars + evidence_chars]
    raise ValueError(f"неизвестный режим {mode!r}")
