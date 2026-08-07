"""
Паттерн supervisor / orchestrator-worker

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l05-supervisor-orchestrator-pattern
Разбор:  /check-code p16-l05-supervisor-orchestrator-pattern
"""

import math

SIMPLE_TOOL_CALLS = 10
MAX_WORKERS = 10
ASPECTS = (
    "historical origins",
    "state of the art",
    "open problems",
    "known failure modes",
    "cost and latency",
    "who disagrees",
    "tooling and frameworks",
    "evaluation",
    "adoption in production",
    "what changes next",
)


class SupervisorError(Exception):
    """Ошибка планирования у lead-агента.

    Собственный класс: RuntimeError и его потомок NotImplementedError
    неотличимы, и тест на RuntimeError зеленел бы на пустой заготовке.
    """
    pass


def scale_effort(estimated_tool_calls):
    """Сколько воркеров запускать под оценку сложности запроса.

    До SIMPLE_TOOL_CALLS включительно — один агент. Дальше по воркеру на
    каждые SIMPLE_TOOL_CALLS вызовов, но не больше MAX_WORKERS.

    scale_effort(3)    ->  1
    scale_effort(10)   ->  1
    scale_effort(11)   ->  2
    scale_effort(999)  ->  10

    Оценку делает lead, а не вызывающий: в этом смысл «scale effort to
    query complexity». Запуск десяти воркеров под запрос из трёх вызовов
    инструментов — чистый убыток.
    """
    raise NotImplementedError


def plan(query, worker_count):
    """Разбиение запроса на подвопросы — по одному на воркера.

    plan("what changed?", 1)   ->  ["what changed?"]
    plan("what changed?", 2)
        ->  ["what changed? -- historical origins",
             "what changed? -- state of the art"]

    worker_count вне диапазона 1..len(ASPECTS)  ->  SupervisorError

    Ловушка и одновременно ключевое свойство: при одном воркере запрос НЕ
    режется. Supervisor с единственным исполнителем обязан выродиться в
    обычного одиночного агента, иначе воркер исследует не то, что спросили.
    """
    raise NotImplementedError


def run_workers(sub_questions, worker):
    """Запуск воркеров. worker(sub_question) -> dict с ключом "answer".

    К каждому результату добавляется его подвопрос. Упавший воркер не
    роняет весь запуск: у него answer=None и текст ошибки в "error", lead
    синтезирует ответ из оставшихся.

    run_workers(["q"], lambda q: {"answer": "a", "seconds": 0.3})
        ->  [{"sub_question": "q", "answer": "a", "seconds": 0.3}]

    Порядок результатов совпадает с порядком подвопросов, даже если в бою
    воркеры бегут параллельно: иначе синтез перепутает, кто что сказал.
    """
    raise NotImplementedError


def sequential_seconds(worker_seconds, plan_seconds, synth_seconds):
    """Время, если те же подвопросы прогнать одним агентом подряд.

    sequential_seconds([0.3, 0.3, 0.3], 0.05, 0.05)  ->  1.0

    Это базовая линия, с которой сравнивают supervisor.
    """
    raise NotImplementedError


def parallel_seconds(worker_seconds, plan_seconds, synth_seconds, spawn_seconds):
    """Время supervisor: план + порождение воркеров + самый долгий из них + синтез.

    Воркеры работают одновременно, поэтому берётся максимум, а не сумма.
    Но каждый воркер стоит spawn_seconds — это и есть цена координации.

    parallel_seconds([0.3, 0.3, 0.3], 0.05, 0.05, 0.02)  ->  0.46
    parallel_seconds([], 0.05, 0.05, 0.02)               ->  0.1

    Сравни с sequential_seconds на тех же числах (1.0): выигрыш есть,
    пока воркеры долгие. На коротких подзадачах spawn_seconds съедает всё.
    """
    raise NotImplementedError


def detect_conflicts(results):
    """Утверждения, по которым воркеры разошлись: {claim: [вердикты по алфавиту]}.

    Каждый результат может нести dict "claims" вида {утверждение: вердикт}.
    Конфликт — это одно утверждение с разными вердиктами у разных воркеров.

    detect_conflicts([{"claims": {"x": "yes"}}, {"claims": {"x": "no"}}])
        ->  {"x": ["no", "yes"]}
    detect_conflicts([{"claims": {"x": "yes"}}, {"claims": {"x": "yes"}}])
        ->  {}

    Никакого LLM здесь не нужно: расхождение видно по структуре. Молча
    выбрать одну сторону — худший из возможных исходов синтеза, потому что
    пользователь никогда не узнает, что спор был.
    """
    raise NotImplementedError


def synthesize(query, results, conflicts):
    """Финальный ответ lead-агента: сводка подответов, пропуски и разногласия.

    Формат — по строке на пункт:

        Answer to 'q':
        - q -- state of the art: ...
        - MISSING q -- open problems: worker timeout
        ! CONFLICT on react-compiler: auto-memoizes vs manual memo

    Lead не читает сырые материалы, только подответы воркеров — ровно то,
    что делает production-система Anthropic.
    """
    raise NotImplementedError


def supervisor_run(query, worker, estimated_tool_calls, plan_seconds=0.05,
                   synth_seconds=0.05, spawn_seconds=0.02):
    """Полный проход supervisor: оценка -> план -> воркеры -> синтез.

    Возвращает (answer, stats), где stats содержит worker_count,
    sequential_seconds, parallel_seconds, coordination_cost, conflicts,
    failed.

    coordination_cost = parallel_seconds - sequential_seconds. Плюс
    означает, что мультиагент проиграл: на мелких задачах именно так и
    выходит, и это главная причина не включать supervisor по умолчанию.

    При estimated_tool_calls <= SIMPLE_TOOL_CALLS запускается ровно один
    воркер с исходным запросом — вырожденный одиночный агент.
    """
    raise NotImplementedError
