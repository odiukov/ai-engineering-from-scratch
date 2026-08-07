"""
Голосовые агенты: Pipecat и LiveKit

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l22-voice-agents-pipecat-livekit
Разбор:  /check-code p14-l22-voice-agents-pipecat-livekit
"""

UPSTREAM_KINDS = ("cancel", "metrics", "error", "interruption")
LATENCY_TIERS = ((600, "premium"), (1200, "common"), (1500, "degraded"))
CONFIDENCE_THRESHOLD = 0.6
REPEAT_PROMPT = "could you repeat that?"
CONTINUATION_WORDS = ("and", "but", "so", "um", "uh", "because")
SILENCE_THRESHOLD_MS = 700
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
DEFAULT_REPLY = "sorry, I did not catch that"


def frame_direction(kind):
    """Куда течёт фрейм: "upstream" или "downstream".

    frame_direction("cancel")      ->  "upstream"
    frame_direction("transcript")  ->  "downstream"

    Всё, чего нет в UPSTREAM_KINDS, идёт вниз. Управляющих типов мало, и
    перечислять надо именно их: список полезных данных растёт каждую неделю.
    """
    raise NotImplementedError


def latency_budget(stages):
    """Сложить бюджет конвейера. stages — список пар (имя стадии, мс).

    latency_budget([("vad", 40), ("stt", 200), ("llm", 250), ("tts", 150)])
        ->  {"total_ms": 640, "worst_stage": "llm", "tier": "common"}

    tier берётся из LATENCY_TIERS по первому порогу, который total не
    превысил; всё, что выше последнего порога, — "broken".

    worst_stage — стадия с максимумом; при равенстве побеждает ПЕРВАЯ, чтобы
    отчёт не прыгал между прогонами.

    Пустой список и отрицательная задержка — ValueError.

    Зачем: премиальный стек живёт в 450-600 мс. Сумму надо посчитать до
    релиза, а не услышать от пользователя.
    """
    raise NotImplementedError


def gate_transcript(text, confidence, threshold=CONFIDENCE_THRESHOLD):
    """Пустить транскрипт в LLM или попросить повторить.

    Возвращает {"accepted": bool, "text": ...}.

    gate_transcript("refund please", 0.9)
        ->  {"accepted": True, "text": "refund please"}
    gate_transcript("refund please", 0.2)
        ->  {"accepted": False, "text": REPEAT_PROMPT}
    gate_transcript("   ", 0.99)
        ->  {"accepted": False, "text": REPEAT_PROMPT}

    Порог включающий: confidence ровно на пороге — принимаем.
    Пустой (или из одних пробелов) транскрипт не принимаем никогда, какой бы
    уверенности ни был: STT честно распознала тишину.

    Принятый текст отдаём обрезанным по краям — хвостовой пробел уедет в
    ключ кэша промпта и разведёт одинаковые реплики.
    """
    raise NotImplementedError


def is_end_of_turn(transcript, silence_ms, threshold_ms=SILENCE_THRESHOLD_MS):
    """Договорил ли человек. Семантика важнее таймера.

    is_end_of_turn("what is my balance?", 100)  ->  True
    is_end_of_turn("I want to", 100)            ->  False
    is_end_of_turn("I want to", 900)            ->  True
    is_end_of_turn("I want to and", 5000)       ->  False

    Правила по порядку:
      1. заканчивается на ".", "?" или "!" — ход окончен сразу;
      2. последнее слово из CONTINUATION_WORDS — ход НЕ окончен, сколько бы
         человек ни молчал (он подбирает слово, а не закончил);
      3. иначе решает тишина: silence_ms >= threshold_ms.

    Пустой транскрипт — ход не окончен.

    Зачем: LiveKit ставит сюда трансформер semantic turn detection. Голый
    таймер режет человека на полуслове — самая частая жалоба на голосовых
    ботов.
    """
    raise NotImplementedError


def turn_transition(state, event):
    """Шаг конечного автомата хода. Вернуть (новое состояние, действия).

    События: "speech_start", "speech_end", "llm_reply", "tts_end".

    turn_transition(STATE_IDLE, "speech_start")   ->  (STATE_LISTENING, [])
    turn_transition(STATE_SPEAKING, "speech_start")
        ->  (STATE_LISTENING, ["cancel_tts", "cancel_llm"])

    Barge-in — это именно переход SPEAKING -> LISTENING: он поднимает
    UPSTREAM-фреймы отмены. Из THINKING перебивание тоже возможно, но там
    гасить нечего кроме LLM.

    Событие, которое в текущем состоянии не имеет смысла, оставляет состояние
    как есть и не даёт действий — конвейер не должен падать от лишнего
    "tts_end". А вот НЕИЗВЕСТНОЕ событие — ValueError: это опечатка в коде.
    """
    raise NotImplementedError


def play_tts(text, cancel_at_word=None):
    """Разложить реплику на произнесённое и недоговорённое.

    cancel_at_word — сколько слов успели произнести до отмены; None означает
    «дослушали до конца».

    play_tts("hi there friend")     ->  (["hi", "there", "friend"], [])
    play_tts("hi there friend", 1)  ->  (["hi"], ["there", "friend"])
    play_tts("hi there friend", 9)  ->  (["hi", "there", "friend"], [])

    Инвариант, который держит всю отладку barge-in: произнесённое плюс
    недоговорённое всегда равно исходному тексту. Отрицательный
    cancel_at_word — ValueError.
    """
    raise NotImplementedError


def run_turn_script(script, replies):
    """Прогнать сценарий разговора через автомат. Вернуть отчёт.

    script — список пар (событие, полезная нагрузка):
      ("speech_start", None)      человек заговорил (в SPEAKING это barge-in);
      ("speech_end", "текст")     STT отдала финальный транскрипт;
      ("llm_reply", None)         LLM ответила (текст берём из replies);
      ("tts_progress", 2)         TTS успела произнести столько слов;
      ("tts_end", None)           реплика договорена до конца.

    replies — словарь {реплика человека: ответ бота}; чего нет, отвечаем
    DEFAULT_REPLY.

    Возвращает {"state", "heard", "spoken", "interrupted", "actions"}:
      heard       — всё, что человек сказал, В ПОРЯДКЕ поступления;
      spoken      — реплики, договорённые до конца;
      interrupted — пары (реплика, произнесённая часть) для перебитых;
      actions     — накопленные действия автомата (cancel_tts / cancel_llm).

    Главное свойство: barge-in обрывает воспроизведение, но реплика человека,
    сказанная поверх бота, всё равно попадает в heard. Реализации, которые
    сбрасывают контекст по cancel, теряют её — и бот переспрашивает.

    "tts_progress" — не событие автомата, а телеметрия: состояние оно не
    меняет, только запоминает, сколько слов уже прозвучало.
    """
    raise NotImplementedError
