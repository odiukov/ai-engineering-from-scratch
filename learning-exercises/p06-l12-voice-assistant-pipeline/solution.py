"""
Голосовой ассистент целиком — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import collections


def capture_turn(chunks, vad, chunk_ms=20, pre_roll_ms=300, silence_ms=500):
    """Вырезать из потока одну реплику пользователя: от начала речи до паузы.

    chunks — поток кадров, vad — функция «кадр → говорят ли». Возвращается
    список кадров реплики; если речи не было вообще, возвращается пустой
    список.

    Две вещи, ради которых функция и существует:

    1. PRE-ROLL. VAD срабатывает на долю секунды позже начала слова, поэтому
       мы всё время держим последние pre_roll_ms тишины и, сработав,
       добавляем их В НАЧАЛО реплики. Без этого у пользователя стабильно
       съедается первое «эй».
    2. END-POINTING. Реплика заканчивается не на первой же тишине, а когда
       тишина длится silence_ms подряд. Пауза между словами счётчик сбрасывает.

    capture_turn([тихо, тихо, громко, тихо, тихо], vad, 20, 40, 40)
        ->  [тихо, тихо, громко, тихо, тихо]   -- две тишины впереди это pre-roll

    Поток кончился раньше, чем набралась пауза — отдаём что есть.
    Неположительный chunk_ms или отрицательные pre_roll_ms/silence_ms —
    ValueError.
    """
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if pre_roll_ms < 0 or silence_ms < 0:
        raise ValueError("pre_roll_ms and silence_ms must not be negative")
    # deque(maxlen=...) сам выбрасывает лишнее — кольцо ровно на pre-roll
    pre = collections.deque(maxlen=pre_roll_ms // chunk_ms)
    buf, triggered, silent = [], False, 0
    for chunk in chunks:
        if vad(chunk):
            if not triggered:
                buf = list(pre)  # <- вот это и спасает первое слово
                triggered = True
            buf.append(chunk)
            silent = 0  # пауза внутри фразы не считается концом реплики
        elif triggered:
            buf.append(chunk)
            silent += chunk_ms
            if silent >= silence_ms:
                return buf
        else:
            pre.append(chunk)
    return buf if triggered else []


def wake_word_gate(transcript, wake_word):
    """Отделить команду от wake word. Нет wake word — None, ассистент спит.

    wake_word_gate("Hey assistant set a timer", "hey assistant")  ->  "set a timer"
    wake_word_gate("hey assistant", "hey assistant")              ->  ""
    wake_word_gate("please stop", "hey assistant")                ->  None

    Сравнение регистронезависимое и ПОСЛОВНОЕ, wake word должен стоять
    В НАЧАЛЕ фразы. Слово «assistants» не должно будить ассистента, и фраза
    «I told my hey assistant story» — тоже: иначе всегда включённый микрофон
    начнёт реагировать на пересказ.

    Пустой wake_word — ValueError: пустой префикс совпадёт с чем угодно, и
    гейта не будет вовсе.

    Зачем: всегда включённый микрофон — это юридический риск. Wake word
    (Porcupine, openWakeWord) отделяет «слушаю» от «записываю».
    """
    words = transcript.lower().split()
    key = wake_word.lower().split()
    if not key:
        raise ValueError("wake_word must not be empty")
    if words[: len(key)] != key:
        return None
    return " ".join(words[len(key) :])


def filter_silence_hallucination(text, had_speech, blocklist):
    """Погасить выдумку модели: на тишине и на дежурных фразах отдаём "".

    filter_silence_hallucination("Thanks for watching", False, {...})  ->  ""
    filter_silence_hallucination("set a timer", True, {...})           ->  "set a timer"

    Два правила:
      1. had_speech ложно — текста быть не может в принципе, что бы ASR ни
         вернул. Это и есть VAD-gate.
      2. Текст ЦЕЛИКОМ совпал с фразой из blocklist (без учёта регистра,
         пробелов по краям и завершающей точки) — гасим.

    Ловушка калибровки: сравнение именно полное, а не «подстрока входит».
    Реальная реплика «thanks for watching the demo, what is next» обязана
    пройти. Перефильтруешь — ассистент будет твердить «я не могу помочь».

    Зачем: Whisper на тишине уверенно выдаёт «Thanks for watching» —
    это его самая известная галлюцинация.
    """
    if not had_speech:
        return ""
    normalized = text.strip().rstrip(".").strip().lower()
    if normalized in {phrase.strip().rstrip(".").strip().lower() for phrase in blocklist}:
        return ""
    return text


def dispatch_tool(call, tools):
    """Вызвать инструмент по имени. Ошибка возвращается ДАННЫМИ, не исключением.

    dispatch_tool({"name": "add", "args": {"a": 1, "b": 2}}, {"add": add})
        ->  {"ok": True, "result": 3}
    dispatch_tool({"name": "nope", "args": {}}, {})
        ->  {"ok": False, "error": "unknown tool: nope"}

    call — словарь с полями "name" и "args" (словарь именованных аргументов).

    Почему не бросать исключение: результат уходит обратно в LLM как
    очередное сообщение. Модель умеет прочитать «tool failed: timeout» и
    попробовать иначе, а упавший стек она не увидит вообще — упадёт весь
    голосовой круг, и пользователь услышит тишину.

    Исключение внутри самого инструмента ловится и превращается в
    {"ok": False, "error": str(exc)}.
    """
    name = call["name"]
    if name not in tools:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        return {"ok": True, "result": tools[name](**call["args"])}
    except Exception as exc:  # noqa: BLE001 — падение инструмента не должно ронять круг
        return {"ok": False, "error": str(exc)}


def run_tool_with_retry(call, tools, max_attempts=2):
    """Вызвать инструмент, при неудаче повторить, потом честно сдаться.

    run_tool_with_retry(call, tools)  ->  {"ok": True, "result": ..., "attempts": 1}

    К ответу dispatch_tool добавляются "attempts" (сколько раз реально
    позвали) и, если так и не вышло, "degraded": True — сигнал ассистенту
    сказать «сервис погоды сейчас недоступен», а не молчать.

    Успех прекращает попытки немедленно: повторять удавшийся вызов нельзя,
    у инструментов бывают побочные эффекты (таймер завёлся бы дважды).

    max_attempts меньше 1 — ValueError.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        result = dispatch_tool(call, tools)
        if result["ok"]:
            return {**result, "attempts": attempt}
    return {**result, "attempts": max_attempts, "degraded": True}


def first_audio_latency(stages, tokens_before_tts, ms_per_token):
    """Сколько пройдёт от конца речи пользователя до ПЕРВОГО байта ответа.

    first_audio_latency({"stt": 150, "tts": 100}, 20, 10)  ->  450

    stages — фиксированные стадии в миллисекундах (end-pointing, STT, первый
    чанк TTS). К их сумме добавляется ожидание LLM: пока не набежит
    tokens_before_tts токенов по ms_per_token каждый, озвучивать нечего.

    Компромисс, который тут виден числом: копить больше токенов — ровнее
    просодия у TTS, но дольше молчание. Цель урока — уложиться в 800 мс.

    Пустые stages, отрицательные задержки или отрицательное число токенов —
    ValueError.
    """
    if not stages:
        raise ValueError("stages must not be empty")
    if any(v < 0 for v in stages.values()):
        raise ValueError("stage latency must not be negative")
    if tokens_before_tts < 0 or ms_per_token < 0:
        raise ValueError("token budget must not be negative")
    return sum(stages.values()) + tokens_before_tts * ms_per_token


def prune_turn_log(log, now_ts, retention_days=30):
    """Выбросить из журнала реплики старше срока хранения. Новый список.

    prune_turn_log([{"ts": 0, ...}], now_ts=86400 * 40)  ->  []

    Запись хранится, пока now_ts - ts НЕ БОЛЬШЕ retention_days суток;
    ровно на границе она ещё жива. Время в секундах.

    Входной список не меняется: журнал обычно общий, и подчистить его
    на месте посреди разговора — верный способ уронить другой поток.

    Отрицательный retention_days — ValueError.

    Зачем: аудио целой реплики — это персональные данные почти везде.
    Тридцать дней и шифрование на диске — минимальная приличная политика.
    """
    if retention_days < 0:
        raise ValueError("retention_days must not be negative")
    limit = retention_days * 86400
    return [entry for entry in log if now_ts - entry["ts"] <= limit]


def assistant_turn(chunks, vad, stt, llm, tts, chunk_ms=20, pre_roll_ms=300,
                   silence_ms=500):
    """Полный круг: захват → STT → LLM → TTS. Возвращает словарь итогов.

    Ключи ответа: "transcript", "reply", "audio", "spoke".

    Если capture_turn не нашёл речи, круг обрывается СРАЗУ: возвращается
    {"transcript": "", "reply": "", "audio": [], "spoke": False}, и ни stt,
    ни llm, ни tts не вызываются вообще.

    Это не оптимизация, а корректность. LLM, которому скормили галлюцинацию
    Whisper с тишины, честно на неё ответит, и ассистент заговорит сам с
    собой в пустой комнате. Плюс каждый несделанный вызов — сэкономленные
    деньги и миллисекунды.

    stt, llm, tts — заглушки вместо настоящих моделей, приходят параметрами:
    stt(кадры) -> строка, llm(строка) -> строка, tts(строка) -> список сэмплов.
    """
    audio = capture_turn(chunks, vad, chunk_ms, pre_roll_ms, silence_ms)
    if not audio:
        return {"transcript": "", "reply": "", "audio": [], "spoke": False}
    transcript = stt(audio)
    reply = llm(transcript)
    return {
        "transcript": transcript,
        "reply": reply,
        "audio": tts(reply),
        "spoke": True,
    }
