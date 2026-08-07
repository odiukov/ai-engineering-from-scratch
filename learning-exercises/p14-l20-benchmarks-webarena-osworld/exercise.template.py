"""
Бенчмарки: WebArena и OSWorld

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l20-benchmarks-webarena-osworld
Разбор:  /check-code p14-l20-benchmarks-webarena-osworld
"""

CATALOG = {
    "sku-001": {"name": "headphones", "price": 199},
    "sku-002": {"name": "keyboard", "price": 129},
    "sku-003": {"name": "mouse", "price": 59},
}
STEP_SUCCESS = "success"
STEP_GROUNDING = "grounding"
STEP_PLANNING = "planning"


def new_state():
    """Пустое состояние приложения: корзина пуста, заказов нет.

    new_state()  ->  {"cart": {}, "orders": []}

    Отдельная функция, а не константа модуля: словарь-константу первый же
    тест испортит, и следующий прогон стартует с чужой корзиной.
    """
    raise NotImplementedError


def apply_action(state, action):
    """Один шаг агента. Вернуть (новое состояние, строка-наблюдение).

    action — кортеж: ("list_items",), ("add_to_cart", sku),
    ("remove_from_cart", sku) или ("checkout",).

    apply_action(new_state(), ("add_to_cart", "sku-003"))
        ->  ({"cart": {"sku-003": 1}, "orders": []}, "added sku-003")
    apply_action(new_state(), ("checkout",))
        ->  (тот же state, "error: empty cart")

    Правила:
      * неизвестный sku и пустая корзина дают "error: ..." и НЕ меняют мир;
      * checkout складывает заказ {"oid", "items", "total"} и чистит корзину;
      * oid нумеруется по числу уже сделанных заказов: "ord-001", "ord-002";
      * неизвестный вид действия — ValueError.

    Ловушка: входной state менять нельзя. Harness прогоняет одну и ту же
    начальную точку по нескольким агентам; мутация превращает сравнение в
    кашу. Возвращай КОПИЮ.
    """
    raise NotImplementedError


def run_trajectory(actions, state=None):
    """Прогнать список действий. Вернуть (финальное состояние, наблюдения).

    run_trajectory([("add_to_cart", "sku-003"), ("checkout",)])[1]
        ->  ["added sku-003", "ord-001"]

    state=None означает «начать с чистого приложения». Переданное состояние
    не мутируется — apply_action уже вернул копию, просто не сломай это.
    """
    raise NotImplementedError


def task_succeeded(state, expected_items):
    """Execution-based проверка: есть ли заказ ровно с таким набором позиций.

    expected_items — словарь {sku: количество}.

    task_succeeded({"cart": {}, "orders": [{"oid": "ord-001",
                    "items": {"sku-001": 1}, "total": 199}]}, {"sku-001": 1})
        ->  True

    Сравнение строгое: лишняя позиция в заказе — это провал, а не «почти».
    Агент, который положил в корзину клавиатуру «на всякий случай», задачу
    не выполнил.

    Зачем: WebArena смотрит на состояние приложения, а не на текст ответа
    агента. Красивое «I placed your order» без заказа в базе — ноль баллов.
    """
    raise NotImplementedError


def trajectory_efficiency(steps, gold_steps):
    """Во сколько раз агент длиннее золотой траектории.

    trajectory_efficiency(6, 3)  ->  2.0
    trajectory_efficiency(3, 3)  ->  1.0

    gold_steps <= 0 — ValueError: делить на длину, которой не бывает, нельзя.
    steps < 0 — тоже ValueError.

    Зачем: OSWorld-Human показал, что лучшие агенты тратят в 1.4-2.7 раза
    больше шагов, чем нужно. Success rate этого не видит совсем.
    """
    raise NotImplementedError


def classify_step(record):
    """Отнести шаг к success / grounding / planning.

    record — словарь с ключами "intended" (какой элемент был нужен),
    "clicked" (во что попали, None если мимо) и "plan_ok" (было ли намерение
    верным).

    classify_step({"intended": "buy", "clicked": "buy", "plan_ok": True})
        ->  STEP_SUCCESS
    classify_step({"intended": "buy", "clicked": "cart", "plan_ok": True})
        ->  STEP_GROUNDING
    classify_step({"intended": "cart", "clicked": "cart", "plan_ok": False})
        ->  STEP_PLANNING

    Порядок проверок важен: сначала план, потом попадание. Агент, который
    метко кликнул не в ту кнопку, провалил ПЛАНИРОВАНИЕ, а не grounding —
    иначе метрика соврёт про то, что чинить.
    """
    raise NotImplementedError


def failure_breakdown(records):
    """Сколько шагов каждого класса. Всегда все три ключа, даже нулевые.

    failure_breakdown([{"intended": "a", "clicked": "a", "plan_ok": True}])
        ->  {"success": 1, "grounding": 0, "planning": 0}

    Нули важны: дашборд с пропавшей категорией читается как «такого не
    бывает», хотя на деле просто не встретилось в этом прогоне.
    """
    raise NotImplementedError


def benchmark_report(results):
    """Свести прогон в отчёт: доля успеха и средняя избыточность шагов.

    results — список словарей с ключами "task_id", "success", "steps",
    "gold_steps".

    Возвращает {"tasks": ..., "solved": ..., "success_rate": ...,
                "efficiency": ...}.

    efficiency считается ТОЛЬКО по решённым задачам: длина проваленной
    траектории ничего не говорит об избыточности — агент мог сдаться на
    первом шаге и выглядеть «эффективнее» человека. Если решённых нет,
    efficiency = 0.0.

    Отчёт не зависит от порядка задач. Повторяющийся task_id — ValueError.
    """
    raise NotImplementedError
