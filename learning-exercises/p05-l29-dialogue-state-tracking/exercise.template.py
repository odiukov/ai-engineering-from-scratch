"""
Dialogue state tracking

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l29-dialogue-state-tracking
Разбор:  /check-code p05-l29-dialogue-state-tracking
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
    raise NotImplementedError


def is_negated(utterance, slot, cues):
    """Просит ли пользователь забыть про этот слот.

    Признак — сигнальная фраза отказа И упоминание самого слота.

    is_negated("never mind the cuisine", "cuisine", ["never mind"])  ->  True
    is_negated("never mind the cuisine", "area", ["never mind"])     ->  False

    Проверять только сигнальную фразу нельзя: "never mind the cuisine"
    очистило бы заодно район и цену, которые пользователь не трогал.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def track_dialogue(turns, ontology, negation_cues, initial_state=None):
    """Прогнать диалог по ходам. Вернуть список состояний — по одному на ход.

    track_dialogue(["cheap place", "actually moderate"], ONT, CUES)
        ->  [{"price": "cheap"}, {"price": "moderate"}]

    Ловушка: в списке должны лежать НЕЗАВИСИМЫЕ снимки. Если класть один и
    тот же dict, после последнего хода вся история окажется одинаковой, а
    именно по ней считается joint goal accuracy по ходам.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
