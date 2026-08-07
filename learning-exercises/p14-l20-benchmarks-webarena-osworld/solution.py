"""
Бенчмарки: WebArena и OSWorld — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками то, что WebArena даёт через gym API, а OSWorld-Human — через
размеченные золотые траектории: детерминированный «магазин» как состояние,
execution-based проверка успеха (смотрим на СОСТОЯНИЕ, а не на текст ответа),
метрику trajectory efficiency (шаги против золота) и разделение провалов на
grounding и planning в духе OSWorld-G. Ни сети, ни браузера.
"""

# Прибитый каталог «самохостящегося приложения»: WebArena фиксирует версии
# приложений именно для того, чтобы прогон был воспроизводим.
CATALOG = {
    "sku-001": {"name": "headphones", "price": 199},
    "sku-002": {"name": "keyboard", "price": 129},
    "sku-003": {"name": "mouse", "price": 59},
}

# Классы шага для разбора провалов (OSWorld-G отделяет grounding от планирования).
STEP_SUCCESS = "success"
STEP_GROUNDING = "grounding"
STEP_PLANNING = "planning"


def new_state():
    """Пустое состояние приложения: корзина пуста, заказов нет.

    new_state()  ->  {"cart": {}, "orders": []}

    Отдельная функция, а не константа модуля: словарь-константу первый же
    тест испортит, и следующий прогон стартует с чужой корзиной.
    """
    return {"cart": {}, "orders": []}


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
    kind = action[0]
    cart = dict(state["cart"])
    orders = list(state["orders"])

    if kind == "list_items":
        return {"cart": cart, "orders": orders}, f"listed {len(CATALOG)} items"

    if kind == "add_to_cart":
        sku = action[1]
        if sku not in CATALOG:
            return {"cart": cart, "orders": orders}, "error: unknown sku"
        cart[sku] = cart.get(sku, 0) + 1
        return {"cart": cart, "orders": orders}, f"added {sku}"

    if kind == "remove_from_cart":
        sku = action[1]
        if sku not in cart:
            return {"cart": cart, "orders": orders}, "error: not in cart"
        del cart[sku]
        return {"cart": cart, "orders": orders}, f"removed {sku}"

    if kind == "checkout":
        if not cart:
            return {"cart": cart, "orders": orders}, "error: empty cart"
        total = sum(CATALOG[sku]["price"] * qty for sku, qty in cart.items())
        oid = f"ord-{len(orders) + 1:03d}"
        orders.append({"oid": oid, "items": dict(cart), "total": total})
        return {"cart": {}, "orders": orders}, oid

    raise ValueError(f"неизвестное действие: {kind!r}")


def run_trajectory(actions, state=None):
    """Прогнать список действий. Вернуть (финальное состояние, наблюдения).

    run_trajectory([("add_to_cart", "sku-003"), ("checkout",)])[1]
        ->  ["added sku-003", "ord-001"]

    state=None означает «начать с чистого приложения». Переданное состояние
    не мутируется — apply_action уже вернул копию, просто не сломай это.
    """
    current = new_state() if state is None else state
    observations = []
    for action in actions:
        current, obs = apply_action(current, action)
        observations.append(obs)
    return current, observations


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
    return any(order["items"] == expected_items for order in state["orders"])


def trajectory_efficiency(steps, gold_steps):
    """Во сколько раз агент длиннее золотой траектории.

    trajectory_efficiency(6, 3)  ->  2.0
    trajectory_efficiency(3, 3)  ->  1.0

    gold_steps <= 0 — ValueError: делить на длину, которой не бывает, нельзя.
    steps < 0 — тоже ValueError.

    Зачем: OSWorld-Human показал, что лучшие агенты тратят в 1.4-2.7 раза
    больше шагов, чем нужно. Success rate этого не видит совсем.
    """
    if gold_steps <= 0:
        raise ValueError("gold_steps должен быть положительным")
    if steps < 0:
        raise ValueError("steps не может быть отрицательным")
    return steps / gold_steps


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
    if not record["plan_ok"]:
        return STEP_PLANNING
    if record["clicked"] != record["intended"]:
        return STEP_GROUNDING
    return STEP_SUCCESS


def failure_breakdown(records):
    """Сколько шагов каждого класса. Всегда все три ключа, даже нулевые.

    failure_breakdown([{"intended": "a", "clicked": "a", "plan_ok": True}])
        ->  {"success": 1, "grounding": 0, "planning": 0}

    Нули важны: дашборд с пропавшей категорией читается как «такого не
    бывает», хотя на деле просто не встретилось в этом прогоне.
    """
    counts = {STEP_SUCCESS: 0, STEP_GROUNDING: 0, STEP_PLANNING: 0}
    for record in records:
        counts[classify_step(record)] += 1
    return counts


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
    seen = set()
    solved_ratios = []
    for item in results:
        if item["task_id"] in seen:
            raise ValueError(f"task_id встречается дважды: {item['task_id']!r}")
        seen.add(item["task_id"])
        if item["success"]:
            solved_ratios.append(
                trajectory_efficiency(item["steps"], item["gold_steps"])
            )
    tasks = len(seen)
    solved = len(solved_ratios)
    return {
        "tasks": tasks,
        "solved": solved,
        "success_rate": (solved / tasks) if tasks else 0.0,
        "efficiency": (sum(solved_ratios) / solved) if solved else 0.0,
    }
