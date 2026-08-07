"""
Dialogue state tracking — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json
import re


def extract_slots(utterance, ontology):
    """Вытащить из реплики значения слотов по онтологии синонимов.

    ontology — {слот: {каноническое значение: [синонимы]}}. Возвращается
    dict только с НАЙДЕННЫМИ слотами: ненайденных в нём нет вовсе, и это
    не то же самое, что слот со значением None (тот означает «очищен»).

    ONT = {"price": {"cheap": ["cheap", "budget"]},
           "area": {"north": ["north"]}}
    extract_slots("a cheap place in the north", ONT)
        ->  {"price": "cheap", "area": "north"}
    extract_slots("hello", ONT)  ->  {}

    Ловушка: искать надо по границам слов. Простое вхождение подстроки
    найдёт "north" внутри "northern lights" и подставит район, которого
    пользователь не называл.

    Первое совпавшее каноническое значение выигрывает, порядок берётся из
    онтологии.
    """
    low = utterance.lower()
    found = {}
    for slot, values in ontology.items():
        for canonical, synonyms in values.items():
            # re.escape: синонимы бывают многословные и с дефисом
            # ("mid-range"), их спецсимволы не должны стать регэкспом
            if any(re.search(rf"\b{re.escape(syn)}\b", low) for syn in synonyms):
                found[slot] = canonical
                break
    return found


def is_negated(utterance, slot, cues):
    """Просит ли пользователь забыть про этот слот.

    Признак — сигнальная фраза отказа И упоминание самого слота.

    is_negated("never mind the cuisine", "cuisine", ["never mind"])  ->  True
    is_negated("never mind the cuisine", "area", ["never mind"])     ->  False

    Проверять только сигнальную фразу нельзя: "never mind the cuisine"
    очистило бы заодно район и цену, которые пользователь не трогал.
    """
    low = utterance.lower()
    return any(cue in low for cue in cues) and slot.lower() in low


def update_state(state, utterance, ontology, negation_cues):
    """Обновить состояние по одной реплике. Вернуть НОВЫЙ dict.

    Три инварианта:
      * слот, которого пользователь не касался, остаётся как был;
      * явный отказ ("never mind the cuisine") ставит слот в None;
      * новое значение перезаписывает старое, а не дописывается к нему.

    update_state({"price": "cheap"}, "actually moderate", ONT, CUES)
        ->  {"price": "moderate"}

    Ловушка: `state` править нельзя — DST хранит историю состояний по
    ходам, и мутация задним числом испортит все предыдущие снимки.

    Извлечение сильнее отказа: в "never mind the cuisine, any food is
    fine" слот получает значение "any", а не None.
    """
    new_state = dict(state)
    found = extract_slots(utterance, ontology)
    new_state.update(found)
    for slot in ontology:
        if slot not in found and is_negated(utterance, slot, negation_cues):
            new_state[slot] = None
    return new_state


def track_dialogue(turns, ontology, negation_cues, initial_state=None):
    """Прогнать диалог по ходам. Вернуть список состояний — по одному на ход.

    track_dialogue(["cheap place", "actually moderate"], ONT, CUES)
        ->  [{"price": "cheap"}, {"price": "moderate"}]

    Ловушка: в списке должны лежать НЕЗАВИСИМЫЕ снимки. Если класть один и
    тот же dict, после последнего хода вся история окажется одинаковой, а
    именно по ней считается joint goal accuracy по ходам.
    """
    state = dict(initial_state) if initial_state else {}
    history = []
    for turn in turns:
        # update_state и так возвращает новый dict — благодаря этому
        # каждый снимок независим, копировать ещё раз не нужно
        state = update_state(state, turn, ontology, negation_cues)
        history.append(state)
    return history


def validate_state(raw_state, schema):
    """Отфильтровать состояние по схеме: как Pydantic, только руками.

    schema — {слот: список допустимых значений}, либо None вместо списка
    для свободного слота (имя ресторана, время).

    SCHEMA = {"price": ["cheap", "moderate"], "name": None}
    validate_state({"price": "CHEAP ", "name": "The Kettle"}, SCHEMA)
        ->  {"price": "cheap", "name": "the kettle"}
    validate_state({"price": "gratis", "colour": "red"}, SCHEMA)  ->  {}

    Что делаем: неизвестные слоты выбрасываем, строки приводим к нижнему
    регистру и обрезаем пробелы, значения вне закрытого списка
    выбрасываем. None оставляем — это «слот очищен», а не мусор.

    Зачем: LLM охотно придумает слот, которого нет в схеме, и напишет
    "Italian" там, где бэкенд ждёт "italian". Это ровно то, что в проде
    закрывают structured outputs.
    """
    clean = {}
    for slot, value in raw_state.items():
        if slot not in schema:
            continue
        if value is None:
            clean[slot] = None
            continue
        if isinstance(value, str):
            value = value.strip().lower()
        allowed = schema[slot]
        if allowed is None or value in allowed:
            clean[slot] = value
    return clean


def llm_dst(history, llm, schema):
    """Пересобрать состояние целиком по всей истории через LLM-заглушку.

    llm — заглушка вместо модели: вызывается как llm(prompt) и возвращает
    строку с JSON-состоянием. Промпт содержит ВСЕ ходы, по одному на
    строку в виде "user: <реплика>". Ответ парсится и прогоняется через
    validate_state.

    llm_dst(["cheap food"], lambda p: '{"price": "cheap"}', SCHEMA)
        ->  {"price": "cheap"}
    llm_dst(["cheap food"], lambda p: 'sorry!', SCHEMA)  ->  None

    None означает «модель не отдала JSON» — это не пустое состояние.
    Пустое означало бы, что пользователь ничего не просил, и бэкенд
    спокойно поехал бы дальше со стёртой бронью.

    Почему всю историю, а не последний ход: регенерация состояния целиком
    сама разруливает correction ("actually make it Thursday") и
    coreference, которые инкрементальному апдейту не даются. Цена —
    O(n^2) токенов по диалогу.
    """
    prompt = "\n".join(f"user: {turn}" for turn in history)
    raw = llm(prompt)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except ValueError:  # JSONDecodeError — подкласс ValueError
        return None
    if not isinstance(parsed, dict):
        return None
    return validate_state(parsed, schema)


def joint_goal_accuracy(predicted, gold):
    """Доля ходов, где состояние совпало с эталоном ЦЕЛИКОМ.

    predicted и gold — списки состояний одинаковой длины.

    joint_goal_accuracy([{"a": 1}], [{"a": 1}])            ->  1.0
    joint_goal_accuracy([{"a": 1, "b": 2}], [{"a": 1}])    ->  0.0
    joint_goal_accuracy([], [])                            ->  0.0

    Метрика «всё или ничего»: один неверный слот из четырёх обнуляет весь
    ход. Поэтому 83% на MultiWOZ 2.4 — это сильный результат, а не
    посредственный.

    Списки разной длины — ValueError.
    """
    if len(predicted) != len(gold):
        raise ValueError("списки состояний разной длины")
    if not predicted:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    return correct / len(predicted)


def slot_accuracy(predicted, gold):
    """Доля верных слотов по всем ходам (micro, не по ходам, а по слотам).

    Считаем по ОБЪЕДИНЕНИЮ ключей предсказания и эталона: слот, который
    система забыла предсказать, обязан считаться ошибкой, а не исчезать
    из знаменателя.

    slot_accuracy([{"a": 1, "b": 2}], [{"a": 1, "b": 9}])  ->  0.5
    slot_accuracy([{}], [{}])                              ->  0.0

    Всегда >= joint goal accuracy. Разрыв между ними и есть диагноз: JGA
    низкая при высокой slot accuracy значит, что ошибки размазаны по
    разным ходам, а не сидят в одном сломанном слоте.

    Списки разной длины — ValueError.
    """
    if len(predicted) != len(gold):
        raise ValueError("списки состояний разной длины")
    total = 0
    correct = 0
    for p, g in zip(predicted, gold):
        for slot in set(p) | set(g):
            total += 1
            # .get, а не [], — отсутствующий слот сравнивается с None и
            # честно считается ошибкой
            if p.get(slot) == g.get(slot):
                correct += 1
    return correct / total if total else 0.0
