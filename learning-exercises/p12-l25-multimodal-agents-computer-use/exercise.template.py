"""
Мультимодальные агенты и computer-use

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l25-multimodal-agents-computer-use
Разбор:  /check-code p12-l25-multimodal-agents-computer-use
"""

import json

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def success_rate(results):
    """Доля успешных задач бенчмарка.

    results — список словарей с ключом "success".

    success_rate([{"success": True}, {"success": False}])  ->  0.5
    success_rate([])                                       ->  0.0

    Пустой прогон — 0.0, а не деление на ноль и не 1.0. Ноль задач это
    сломанный харнесс, и он не должен выглядеть как идеальный результат.
    """
    raise NotImplementedError
