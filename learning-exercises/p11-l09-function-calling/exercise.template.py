"""
Function calling: схемы инструментов, валидация, диспетчер, агентный цикл

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l09-function-calling
Разбор:  /check-code p11-l09-function-calling
"""

import json

JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units, defaults to celsius",
        },
    },
    "required": ["city"],
}
CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"},
        "precision": {"type": "integer", "description": "Decimal places in result"},
    },
    "required": ["expression"],
}


def register_tool(registry, name, description, parameters, function):
    """Положить инструмент в реестр: схему для модели и функцию для себя.

    reg = register_tool({}, "get_weather", "Get current weather", WEATHER_SCHEMA, fn)
    reg["get_weather"]["definition"]["function"]["name"]         ->  'get_weather'
    reg["get_weather"]["definition"]["function"]["parameters"]   ->  WEATHER_SCHEMA
    reg["get_weather"]["function"]                               ->  fn

    Формат definition — тот, что уходит в OpenAI:
        {"type": "function", "function": {"name", "description", "parameters"}}
    У Anthropic та же информация лежит плоско и поле называется input_schema:
        {"name", "description", "input_schema"}.

    description — это НЕ комментарий, а промпт для выбора инструмента. Модель
    читает именно его, решая, что вызвать. "gets weather" выбирается заметно
    хуже, чем "Get current weather for a city. Returns temperature in Celsius".

    Реестр возвращается тот же самый (изменённый на месте) — так удобно
    регистрировать несколько инструментов подряд.
    """
    raise NotImplementedError


def validate_arguments(schema, arguments):
    """Проверить аргументы против JSON Schema. Вернуть список текстов ошибок.

    validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo"})   ->  []
    validate_arguments(WEATHER_SCHEMA, {})
        ->  ['Missing required argument: city']
    validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "units": "kelvin"})
        ->  ["Argument 'units': 'kelvin' not in ['celsius', 'fahrenheit']"]

    Пустой список означает "всё в порядке". Проверяются четыре вещи:
    обязательные поля на месте, лишних полей нет, типы совпадают, значение
    входит в enum.

    Ловушка: в Python bool — подкласс int, поэтому isinstance(True, int) даёт
    True, и {"precision": True} проедет как целое число. Булев случай надо
    отсекать отдельно.

    Зачем вообще: модель генерирует аргументы текстом. Она может прислать
    город "; DROP TABLE users; --" или строку там, где ждали число. Между
    моделью и твоей функцией обязан стоять валидатор.
    """
    raise NotImplementedError


def parse_tool_calls(raw):
    """Разобрать ответ модели в список вызовов [{"name", "arguments"}, ...].

    parse_tool_calls('{"name": "get_weather", "arguments": {"city": "Tokyo"}}')
        ->  [{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}]
    parse_tool_calls('[{"name": "a", "arguments": {}}, {"name": "b"}]')
        ->  [{'name': 'a', 'arguments': {}}, {'name': 'b', 'arguments': {}}]

    Один объект и список объектов принимаются одинаково: параллельные вызовы
    приходят списком, одиночный — объектом. Пропущенное поле arguments
    считается пустым словарём. id и call_id нормализуются в call_id:
    параллельные результаты связываются со своими вызовами именно по нему.

    Ловушка из настоящего API: OpenAI отдаёт tool_calls[].function.arguments
    СТРОКОЙ с JSON внутри, а не объектом. Такую строку надо распарсить ещё
    раз, иначе валидатор увидит str там, где ждал dict.

    На мусоре на входе поднимай ValueError с внятным текстом — модель
    иногда возвращает оборванный JSON, и молча проглатывать это нельзя.
    """
    raise NotImplementedError


def execute_tool_call(registry, call):
    """Выполнить один вызов. Вернуть {"tool", "ok", "result"}.

    execute_tool_call(reg, {"name": "get_weather", "arguments": {"city": "Tokyo"}})
        ->  {'tool': 'get_weather', 'ok': True, 'result': {...}}
    execute_tool_call(reg, {"name": "rm_rf", "arguments": {}})
        ->  {'tool': 'rm_rf', 'ok': False,
             'result': {'error': True, 'code': 'UNKNOWN_TOOL', 'message': ...}}

    Порядок проверок: инструмент есть в реестре -> аргументы валидны ->
    только потом вызов. Коды ошибок: UNKNOWN_TOOL, INVALID_ARGUMENTS,
    TOOL_ERROR.

    Правило allowlist: вызвать можно только то, что лежит в реестре. Общего
    "выполни функцию по имени" быть не должно — иначе модель получает доступ
    ко всему, что импортировано.

    Исключение из инструмента наружу выпускать нельзя. Модель не умеет
    ловить traceback — она умеет читать {"error": true, "message": ...} и
    исправлять аргументы. Поэтому падение превращается в структурный ответ.
    """
    raise NotImplementedError


def run_tool_calls(registry, calls, max_calls=10):
    """Выполнить пачку вызовов подряд, не превышая бюджет max_calls.

    run_tool_calls(reg, [call_a, call_b])       ->  два результата
    run_tool_calls(reg, [call] * 12)            ->  12 результатов, последние
                                                    два с кодом CALL_LIMIT

    Параллельные вызовы приходят пачкой: "погода в Токио и Нью-Йорке" — это
    два вызова get_weather в одном ходе модели. Здесь они выполняются по
    очереди, в проде — конкурентно, но результат тот же.

    Лишние вызовы НЕ выполняются: на них возвращается ошибка CALL_LIMIT.
    Это защита от модели, зациклившейся на инструменте, — иначе один баг
    съедает месячный бюджет за час.
    """
    raise NotImplementedError


def agent_loop(registry, user_message, decide, max_iterations=5):
    """Агентный цикл: модель решает -> код выполняет -> результат обратно.

    Вернуть {"conversation", "results", "iterations"}.

    decide(user_message, conversation) — заглушка модели: возвращает список
    вызовов или пустой список, если инструменты не нужны.

    agent_loop(reg, "Hello", lambda m, c: [])
        ->  iterations = 0, conversation из одного сообщения пользователя

    На каждой итерации в conversation дописывается ход ассистента
    {"role": "assistant", "content": None, "tool_calls": [...]} и по одному
    сообщению {"role": "tool", ...} на каждый результат. Так его и собирают
    настоящие API: результат инструмента — это отдельное сообщение, а не
    текст внутри ответа модели.

    max_iterations — жёсткий предохранитель. Модель, которая на каждом шаге
    просит один и тот же вызов, обязана остановиться сама; если не
    останавливается — останавливает цикл.
    """
    raise NotImplementedError


def tool_selection_accuracy(decide, cases):
    """Насколько часто модель выбирает правильный инструмент.

    cases — список пар (запрос, ожидаемое имя инструмента или None).
    None означает "здесь инструмент не нужен, отвечай текстом".

    tool_selection_accuracy(decide, [("Weather in Tokyo?", "get_weather"),
                                     ("Tell me a joke", None)])
        ->  {'total': 2, 'correct': 2, 'accuracy': 1.0, 'errors': []}

    Сравнивается ПЕРВЫЙ вызов: остальные относятся к параллельным запросам и
    на выбор инструмента не влияют.

    Зачем метрика: без неё "агент иногда путает поиск и калькулятор" остаётся
    ощущением. С ней видно, какие именно запросы путаются, — errors хранит
    ровно эти случаи.
    """
    raise NotImplementedError
