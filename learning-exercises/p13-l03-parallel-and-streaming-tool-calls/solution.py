"""
Параллельные и потоковые вызовы инструментов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Веерный запуск и разбор потока — то, что SDK обычно делает за тебя:
`asyncio.gather` для первого, `stream.get_final_message()` для второго.
Здесь мы пишем обе половины руками. Соответствие настоящему API:

    sequential_duration    <-  цикл по вызовам без параллелизма
    parallel_duration      <-  asyncio.gather / ThreadPoolExecutor.map
    speedup                <-  то, ради чего вводили parallel_tool_calls
    try_parse_arguments    <-  безопасная попытка разобрать частичный JSON
    accumulate_stream      <-  накопитель чанков input_json_delta по id
    stream_completion_order<-  порядок, в котором можно стартовать вызовы
    correlate_results      <-  сборка ответа по tool_call_id
    parallel_batches       <-  планировщик с учётом зависимостей

Настоящих потоков здесь нет: время моделируется числами, а поток событий —
списком словарей. Так тест воспроизводим, а смысл ровно тот же.

Событие потока — словарь одной из трёх форм:
    {"type": "call_start", "id": ..., "name": ...}
    {"type": "args_delta", "id": ..., "chunk": "..."}
    {"type": "call_stop",  "id": ...}
"""

import json

# Типы событий потока, которые накопитель умеет обрабатывать.
STREAM_EVENTS = ("call_start", "args_delta", "call_stop")


def sequential_duration(latencies):
    """Сколько миллисекунд займут вызовы, если запускать их по очереди.

    sequential_duration([400, 600, 800])  ->  1800
    sequential_duration([])               ->  0

    Последовательный прогон платит СУММУ задержек. Именно это и происходит,
    когда модель эмитит вызовы по одному за ход.
    """
    return sum(latencies)


def parallel_duration(latencies):
    """Сколько миллисекунд займут те же вызовы веером.

    parallel_duration([400, 600, 800])  ->  800
    parallel_duration([])               ->  0

    Параллельный прогон платит МАКСИМУМ, а не сумму: медленный вызов не
    ждёт своей очереди, он идёт вместе с остальными.

    Ловушка: max([]) бросает ValueError. Пустой веер длится ноль.
    """
    return max(latencies, default=0)


def speedup(latencies):
    """Во сколько раз веер быстрее очереди: сумма / максимум.

    speedup([400, 600, 800])  ->  2.25
    speedup([500, 500])       ->  2.0
    speedup([800])            ->  1.0
    speedup([])               ->  1.0

    Отсюда видно, когда параллелизм перестаёт окупаться: если один вызов
    сильно медленнее остальных, максимум почти равен сумме и ускорение
    стремится к единице.

    Ловушка: деление на ноль. Пустой список и все нулевые задержки дают 1.0.
    """
    slowest = parallel_duration(latencies)
    if slowest == 0:
        return 1.0
    return sequential_duration(latencies) / slowest


def try_parse_arguments(buffer):
    """Попытаться разобрать накопленный буфер аргументов. None, если рано.

    try_parse_arguments('{"city": "Tokyo"}')  ->  {"city": "Tokyo"}
    try_parse_arguments('{"city": "Tok')      ->  None
    try_parse_arguments('')                   ->  {}
    try_parse_arguments('[1, 2]')             ->  None

    Пустой буфер означает вызов без аргументов — провайдеры шлют его как
    пустой объект, и {} тут правильнее, чем None.

    Ловушка «разобрать слишком рано»: json.loads на неполном куске бросает
    JSONDecodeError. Считать фигурные скобки вместо разбора — не спасение:
    в строке '{"city": "a}"' скобки сбалансированы, а JSON невалиден. Пробуй
    разобрать и лови исключение — это дешевле и честнее.

    Не-объект (массив, число, строка) — тоже None: arguments по спецификации
    всегда объект.
    """
    if buffer == "":
        return {}
    try:
        value = json.loads(buffer)
    except ValueError:  # JSONDecodeError — её подкласс
        return None
    return value if isinstance(value, dict) else None


def accumulate_stream(events):
    """Разобрать поток чанков в готовые вызовы. Только завершённые.

    Результат: {id: {"name": ..., "arguments": {...}}}.

    accumulate_stream([
        {"type": "call_start", "id": "A", "name": "get_weather"},
        {"type": "args_delta", "id": "A", "chunk": '{"city"'},
        {"type": "args_delta", "id": "A", "chunk": ':"Tokyo"}'},
        {"type": "call_stop",  "id": "A"},
    ])  ->  {"A": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}

    Три параллельных вызова идут по одному проводу вперемешку, поэтому
    накопитель нужен ОТДЕЛЬНЫЙ НА КАЖДЫЙ id. Один общий буфер склеит чанки
    разных вызовов в мусор.

    Вызов, для которого не пришёл call_stop, в результат не попадает: поток
    оборвался или вызов отменили, разбирать нечего.

    Ошибки: чанк для незнакомого id и неизвестный тип события — ValueError.
    Буфер, который после call_stop не разбирается, — тоже ValueError: это
    поломанный поток, и молча его проглотить хуже, чем упасть.
    """
    names = {}
    buffers = {}
    done = {}
    for event in events:
        kind = event["type"]
        if kind not in STREAM_EVENTS:
            raise ValueError(f"unknown stream event: {kind}")
        call_id = event["id"]
        if kind == "call_start":
            names[call_id] = event["name"]
            buffers[call_id] = ""
            continue
        if call_id not in buffers:
            raise ValueError(f"{kind} for unknown call id: {call_id}")
        if kind == "args_delta":
            buffers[call_id] += event["chunk"]
        else:
            arguments = try_parse_arguments(buffers[call_id])
            if arguments is None:
                raise ValueError(f"incomplete arguments for call id: {call_id}")
            done[call_id] = {"name": names[call_id], "arguments": arguments}
    return done


def stream_completion_order(events):
    """В каком порядке вызовы становятся готовы к запуску.

    stream_completion_order(events)  ->  ["B", "A", "C"]

    Это НЕ порядок call_start. Аргументы короткого вызова дописываются
    раньше, чем длинного, и хост может стартовать его исполнитель, не дожидаясь
    конца всего потока. Ровно на этом экономится время на веерных запросах.

    Считаем только те id, которые дошли до call_stop, — их и вернул
    accumulate_stream.
    """
    ready = accumulate_stream(events)
    return [e["id"] for e in events if e["type"] == "call_stop" and e["id"] in ready]


def correlate_results(calls, results):
    """Сложить результаты обратно к вызовам по id. Порядок — как у вызовов.

    calls    — [{"id", "name", "arguments"}, ...] в порядке эмиссии моделью
    results  — [{"tool_call_id", "content"}, ...] в порядке ЗАВЕРШЕНИЯ

    correlate_results(
        [{"id": "A", "name": "w", "arguments": {}},
         {"id": "B", "name": "w", "arguments": {}}],
        [{"tool_call_id": "B", "content": "b"},
         {"tool_call_id": "A", "content": "a"}])
        ->  [{"role": "tool", "tool_call_id": "A", "name": "w", "content": "a"},
             {"role": "tool", "tool_call_id": "B", "name": "w", "content": "b"}]

    Веер завершается в непредсказуемом порядке, и id — единственное, что
    связывает результат с вызовом. Сопоставление по позиции в списке или по
    имени инструмента ломается ровно там, где два параллельных вызова идут
    в один и тот же инструмент.

    Ошибки: результат без своего вызова, вызов без результата и два
    результата с одним id — всё это ValueError.
    """
    by_id = {}
    for result in results:
        call_id = result["tool_call_id"]
        if call_id in by_id:
            raise ValueError(f"duplicate result for call id: {call_id}")
        by_id[call_id] = result["content"]

    known = {call["id"] for call in calls}
    for call_id in by_id:
        if call_id not in known:
            raise ValueError(f"result for unknown call id: {call_id}")

    messages = []
    for call in calls:
        if call["id"] not in by_id:
            raise ValueError(f"missing result for call id: {call['id']}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": by_id[call["id"]],
            }
        )
    return messages


def parallel_batches(calls, depends_on):
    """Разложить вызовы по волнам: внутри волны их можно пускать веером.

    depends_on — {id вызова: [id вызовов, которые обязаны отработать раньше]}.
    Ключа может не быть вовсе — значит, зависимостей нет.

    parallel_batches(
        [{"id": "create"}, {"id": "write"}, {"id": "weather"}],
        {"write": ["create"]})
        ->  [["create", "weather"], ["write"]]

    create_file и write_file параллелить нельзя: второй читает результат
    первого. weather ни от кого не зависит и едет в первой же волне.

    Внутри волны порядок — как в исходном списке вызовов, чтобы результат
    был воспроизводим.

    Ошибки: ссылка на несуществующий id и цикл зависимостей — ValueError.
    Цикл узнаётся по тому, что очередная волна оказалась пустой, а
    неразобранные вызовы ещё остались.
    """
    order = [call["id"] for call in calls]
    known = set(order)
    for call_id, deps in depends_on.items():
        if call_id not in known:
            raise ValueError(f"dependency declared for unknown call id: {call_id}")
        for dep in deps:
            if dep not in known:
                raise ValueError(f"unknown dependency: {dep}")

    waves = []
    resolved = set()
    remaining = list(order)
    while remaining:
        wave = [
            call_id
            for call_id in remaining
            if all(dep in resolved for dep in depends_on.get(call_id, ()))
        ]
        if not wave:
            raise ValueError(f"dependency cycle among: {sorted(remaining)}")
        waves.append(wave)
        resolved.update(wave)
        remaining = [call_id for call_id in remaining if call_id not in resolved]
    return waves
