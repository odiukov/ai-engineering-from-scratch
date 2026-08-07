"""
Мультимодальные агенты и computer-use — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json

# Схема действий агента: имя -> обязательные поля. Всё, что VLM выдаёт
# сверх этого (например element_desc), допустимо и используется при
# восстановлении после промаха.
ACTION_SCHEMA = {
    "click": ("x", "y"),
    "type": ("text",),
    "scroll": ("direction", "amount"),
    "drag": ("x0", "y0", "x1", "y1"),
    "select": ("option_index",),
    "hover": ("x", "y"),
    "navigate": ("url",),
    "wait": ("ms",),
    "done": ("success",),
}

# Поля-координаты, которые надо пересчитывать при смене разрешения.
X_FIELDS = ("x", "x0", "x1")
Y_FIELDS = ("y", "y0", "y1")


def validate_action(action):
    """Проверка действия по ACTION_SCHEMA. Возвращает список ошибок; пустой — годно.

    validate_action({"action": "click", "x": 10, "y": 20})       ->  []
    validate_action({"action": "click", "x": 10})                ->  ["missing field: y"]
    validate_action({"action": "fly"})                           ->  ["unknown action: fly"]

    Лишние поля разрешены: element_desc и прочие подсказки модели нужны для
    восстановления после промаха.

    Функция НЕ бросает исключение. VLM ошибается в схеме регулярно, и
    агенту надо уметь показать модели список претензий и попросить ещё раз,
    а не падать на первом же кривом JSON.
    """
    if not isinstance(action, dict) or "action" not in action:
        return ["missing key: action"]
    name = action["action"]
    if name not in ACTION_SCHEMA:
        return [f"unknown action: {name}"]
    # порядок ошибок берём из схемы, а не из словаря модели: одинаковый
    # кривой ответ обязан давать одинаковый список претензий
    return [f"missing field: {f}" for f in ACTION_SCHEMA[name] if f not in action]


def parse_action(text):
    """Достать JSON-действие из ответа VLM. Возвращает словарь.

    parse_action('{"action": "click", "x": 1, "y": 2}')
        ->  {"action": "click", "x": 1, "y": 2}
    parse_action('Сначала нажму поиск.\\n```json\\n{"action": "wait", "ms": 5}\\n```')
        ->  {"action": "wait", "ms": 5}

    Модель почти всегда обрамляет JSON рассуждением и ```json-заборчиком.
    Надёжный приём: взять кусок от первой "{" до последней "}" и отдать его
    json.loads.

    Ни JSON, ни разбора — ValueError. Тихо вернуть пустой словарь нельзя:
    агент выполнит "ничего" и решит, что шаг удался.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the model reply")
    # json.JSONDecodeError сам по себе наследник ValueError, поэтому
    # перехватывать и переоборачивать его не надо
    return json.loads(text[start:end + 1])


def scale_click(action, from_size, to_size):
    """Пересчёт координат действия из разрешения модели в разрешение экрана.

    from_size и to_size — пары (width, height). Возвращает НОВОЕ действие.

    scale_click({"action": "click", "x": 100, "y": 50}, (1000, 500), (2000, 1000))
        ->  {"action": "click", "x": 200, "y": 100}

    Классический баг GUI grounding: VLM видит картинку, ужатую до 1120x1120,
    и выдаёт координаты в НЕЙ, а клик уходит на реальный экран 2560x1440.
    Промах тем больше, чем дальше цель от левого верхнего угла.

    Ловушка: оси масштабируются РАЗНЫМИ коэффициентами. Пропорции при
    ресайзе почти никогда не сохраняются.

    Некоординатные поля копируются как есть.
    """
    fx = to_size[0] / from_size[0]
    fy = to_size[1] / from_size[1]
    out = dict(action)
    for f in X_FIELDS:
        if f in out:
            out[f] = int(round(out[f] * fx))
    for f in Y_FIELDS:
        if f in out:
            out[f] = int(round(out[f] * fy))
    return out


def apply_action(state, action):
    """Выполнить действие в мок-браузере. Возвращает НОВОЕ состояние.

    Состояние: {"url": str, "elements": [...], "fields": dict, "error": str|None}.
    Элемент: {"desc": str, "bbox": (x0, y0, x1, y1), "goto": str|None}.

    Поведение:
      click    — попали в bbox элемента с goto: меняется url; мимо: error;
      type     — текст кладётся в fields по ключу action.get("field", "input");
      navigate — url меняется напрямую;
      прочее   — состояние сохраняется, error сбрасывается.

    Успешный шаг всегда сбрасывает error: иначе одна старая ошибка тянется
    через весь эпизод и агент бесконечно «восстанавливается».

    Ловушка: состояние на входе править нельзя — ни сам словарь, ни
    вложенный fields. Бенчмарк прогоняет один и тот же стартовый state по
    десяти задачам подряд.
    """
    new = dict(state)
    new["fields"] = dict(state.get("fields", {}))
    new["error"] = None
    name = action.get("action")

    if name == "click":
        hit = None
        for el in state.get("elements", ()):
            x0, y0, x1, y1 = el["bbox"]
            if x0 <= action["x"] <= x1 and y0 <= action["y"] <= y1:
                hit = el
                break
        if hit is None:
            new["error"] = f"no element at ({action['x']}, {action['y']})"
        elif hit.get("goto"):
            new["url"] = hit["goto"]
    elif name == "type":
        new["fields"][action.get("field", "input")] = action["text"]
    elif name == "navigate":
        new["url"] = action["url"]
    return new


def recover(action, state):
    """Перецелить промахнувшийся клик по семантической подсказке element_desc.

    Возвращает новое действие click в ЦЕНТР найденного элемента или None,
    если перецелиться не по чему.

    recover({"action": "click", "x": 0, "y": 0, "element_desc": "Search"}, state)
        ->  {"action": "click", "x": 350, "y": 60, "element_desc": "Search"}

    Ради этого element_desc в схеме и живёт: между двумя скриншотами вёрстка
    уезжает, координаты протухают, а описание — нет.

    Без element_desc или без подходящего элемента — None, и это сигнал
    агенту перепланировать, а не кликать ещё раз в то же место.
    """
    desc = action.get("element_desc")
    if not desc:
        return None
    for el in state.get("elements", ()):
        if el["desc"] == desc:
            x0, y0, x1, y1 = el["bbox"]
            out = dict(action)
            # центр, а не левый верхний угол: у угла легко промахнуться
            # на рамку в один пиксель
            out["x"] = (x0 + x1) // 2
            out["y"] = (y0 + y1) // 2
            return out
    return None


def compress_history(history, keep_live=4):
    """Summary-chain: оставить живыми последние keep_live скриншотов, остальные снять.

    history — список шагов {"screenshot": ..., "action": {...}}.
    Возвращает НОВЫЙ список: у старых шагов screenshot заменён на None,
    действия сохранены все и в том же порядке.

    len([s for s in compress_history(h, 4) if s["screenshot"] is not None])  ->  4

    Почему так: двадцать шагов — это двадцать картинок по паре тысяч
    токенов каждая, контекст кончится раньше задачи. Текстовый лог
    действий стоит копейки и в проде (computer-use API у Claude) работает
    надёжнее, чем перечитывание старых скриншотов.
    """
    cutoff = len(history) - keep_live
    out = []
    for i, step in enumerate(history):
        new_step = dict(step)
        if i < cutoff:
            new_step["screenshot"] = None
        out.append(new_step)
    return out


def agent_loop(state, policy, max_steps=10):
    """Цикл агента: воспринять -> решить -> подействовать -> повторить.

    policy(state) -> действие. Возвращает кортеж (final_state, trace), где
    trace — список выполненных действий.

    Цикл останавливается на действии "done" или на max_steps шагах.

    Ловушка: потолок шагов обязателен. Агент, который не понял, что задача
    решена, будет кликать по одной и той же кнопке до конца бюджета — а
    каждый шаг это вызов VLM с картинкой.

    Действие "done" попадает в trace, но НЕ применяется к состоянию: это
    отчёт агента, а не событие в браузере.
    """
    trace = []
    for _ in range(max_steps):
        action = policy(state)
        trace.append(action)
        if action.get("action") == "done":
            break
        state = apply_action(state, action)
    return state, trace


def success_rate(results):
    """Доля успешных задач бенчмарка.

    results — список словарей с ключом "success".

    success_rate([{"success": True}, {"success": False}])  ->  0.5
    success_rate([])                                       ->  0.0

    Пустой прогон — 0.0, а не деление на ноль и не 1.0. Ноль задач это
    сломанный харнесс, и он не должен выглядеть как идеальный результат.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)
