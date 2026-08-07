"""
Обработка звука в реальном времени — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import collections


def frame_length(sample_rate, frame_ms):
    """Сколько сэмплов в кадре длиной frame_ms при частоте sample_rate.

    frame_length(16000, 20)  ->  320   (канонический кадр WebRTC)
    frame_length(48000, 20)  ->  960
    frame_length(44100, 10)  ->  441

    Формула: sample_rate * frame_ms / 1000. Результат обязан быть ЦЕЛЫМ:
    44100 Гц и 5 мс дают 220.5 сэмпла, такой кадр нарезать нечем — ValueError.
    Именно поэтому телефония живёт на 8/16/48 кГц и кадрах 10/20/30 мс.

    Неположительные sample_rate или frame_ms — тоже ValueError.
    """
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    exact = sample_rate * frame_ms
    if exact % 1000 != 0:
        raise ValueError("frame does not contain a whole number of samples")
    return exact // 1000


def buffer_latency_ms(n_samples, sample_rate):
    """Задержка, которую вносит буфер такого размера, в миллисекундах.

    buffer_latency_ms(32000, 16000)  ->  2000.0   (двухсекундное кольцо)
    buffer_latency_ms(320, 16000)    ->  20.0

    Формула: n_samples * 1000 / sample_rate.

    Главная мысль урока: буфер — это НЕ страховка, это пол задержки. Пока
    данные лежат в нём, реагировать не на что. «Возьму 500 мс на всякий
    случай» означает «мой ассистент отвечает минимум на полсекунды позже».

    Отрицательный размер или неположительная частота — ValueError.
    """
    if n_samples < 0:
        raise ValueError("n_samples must not be negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return n_samples * 1000.0 / sample_rate


class RingBuffer:
    """Кольцевой буфер фиксированного размера: producer пишет, consumer читает.

    rb = RingBuffer(4); rb.write([1, 2]); rb.read(2)  ->  [1, 2]

    Порядок строго FIFO. При переполнении САМЫЕ СТАРЫЕ сэмплы вытесняются —
    в реальном времени свежий звук важнее просроченного, и остановить
    микрофон всё равно нельзя.

    Размер кольца задаёт максимальную задержку: 32000 сэмплов при 16 кГц —
    это 2 секунды.
    """

    def __init__(self, capacity):
        """Создать кольцо на capacity сэмплов. capacity < 1 — ValueError."""
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        # deque(maxlen=...) сам вытесняет старое с левого конца за O(1),
        # без перевыделения памяти — ровно то, что нужно в hot path
        self._buf = collections.deque(maxlen=capacity)

    def write(self, frame):
        """Дописать кадр в конец. Лишнее вытесняет старое начало."""
        self._buf.extend(frame)

    def read(self, n):
        """Забрать до n самых старых сэмплов. Меньше — вернуть сколько есть.

        Нехватка данных — это НЕ ошибка: consumer просто пришёл раньше
        producer'а, такое в реальном времени происходит постоянно.
        """
        return [self._buf.popleft() for _ in range(min(n, len(self._buf)))]

    def level(self):
        """Сколько сэмплов лежит прямо сейчас. Никогда не больше capacity."""
        return len(self._buf)


def energy_vad(frame, threshold):
    """Простейший VAD: средний квадрат кадра выше threshold ** 2.

    energy_vad([0.0, 0.0], 0.01)     ->  False   (тишина)
    energy_vad([0.5, -0.5], 0.01)    ->  True    (речь)

    Считаем sum(x * x) / len(frame) и сравниваем с threshold ** 2 — это то
    же самое, что RMS > threshold, но без лишнего корня на каждом кадре.

    Ловушка: обычное среднее вместо среднего квадратов даёт ноль на любом
    знакопеременном сигнале, и громкий кадр [-1, 1, -1, 1] будет принят за
    тишину. Квадрат убирает знак.

    Пустой кадр или отрицательный threshold — ValueError.

    Зачем: VAD гасит всю тяжёлую часть пайплайна, пока никто не говорит.
    Silero VAD 4.0 делает то же самое умнее и укладывается в 1 мс на кадр.
    """
    if not frame:
        raise ValueError("frame must not be empty")
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    return sum(x * x for x in frame) / len(frame) > threshold ** 2


def jitter_buffer(packets, depth):
    """Собрать пакеты в правильном порядке, придержав их в буфере глубины depth.

    jitter_buffer([(0, "a"), (2, "c"), (1, "b")], depth=2)  ->  ["a", "b", "c"]
    jitter_buffer([(0, "a"), (2, "c"), (3, "d"), (1, "b")], depth=1)
        ->  ["a", "c", "d"]   -- "b" опоздал, его место уже проиграно

    packets — список пар (seq, payload) в порядке ПРИХОДА, а не в порядке
    номеров. Буфер копит пакеты; как только их становится больше depth,
    самый старый ожидаемый номер уходит в вывод. Номера, до которых очередь
    уже дошла, назад не принимаются — пакет просто выбрасывается.

    Компромисс, ради которого всё это: чем больше depth, тем меньше потерь и
    тем больше задержка. На практике 60-80 мс.

    depth меньше 1 — ValueError.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    pending = {}
    out = []
    next_seq = 0
    for seq, payload in packets:
        if seq < next_seq:
            continue  # опоздал: его место в потоке уже проиграно
        pending[seq] = payload
        while len(pending) > depth:
            if next_seq in pending:
                out.append(pending.pop(next_seq))
            next_seq += 1  # номера нет — это дырка, ждать её больше нельзя
    while pending:  # поток кончился, сливаем остаток по порядку
        if next_seq in pending:
            out.append(pending.pop(next_seq))
        next_seq += 1
    return out


def pipeline_latency(stages):
    """Сквозная задержка каскада: сумма задержек всех стадий.

    pipeline_latency({"vad": 10, "asr": 150, "llm": 100})  ->  260

    Стадии выстроены цепочкой, и один кадр проходит их одну за другой,
    поэтому именно СУММА определяет, через сколько пользователь услышит
    ответ. Бюджет из урока — около 400 мс на весь круг.

    Пустой набор стадий или отрицательная задержка — ValueError.
    """
    if not stages:
        raise ValueError("pipeline must have at least one stage")
    values = list(stages.values())
    if any(v < 0 for v in values):
        raise ValueError("stage latency must not be negative")
    return sum(values)


def keeps_up_with_realtime(stages, frame_ms):
    """Успевает ли конвейер за потоком: САМАЯ МЕДЛЕННАЯ стадия против кадра.

    keeps_up_with_realtime({"vad": 10, "asr": 150}, frame_ms=20)   ->  False
    keeps_up_with_realtime({"vad": 10, "asr": 18}, frame_ms=20)    ->  True

    Ключевое отличие от pipeline_latency: стадии работают ОДНОВРЕМЕННО, но
    над разными кадрами. Пока ASR жуёт кадр №5, VAD уже смотрит на №6.
    Поэтому пропускную способность ограничивает максимум, а не сумма: набор
    из двадцати стадий по 15 мс имеет сумму 300 мс и при этом прекрасно
    держит поток с кадром 20 мс.

    Стадия ровно в размер кадра ещё успевает — сравнение нестрогое.

    Пустой набор или неположительный frame_ms — ValueError.
    """
    if not stages:
        raise ValueError("pipeline must have at least one stage")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    return max(stages.values()) <= frame_ms


def barge_in(state, user_speaking, reaction_ms, limit_ms=100):
    """Перебивание: пользователь заговорил поверх TTS — гасим ответ.

    state — словарь с полями "tts_playing" (bool) и "pending_chunks" (список
    ещё не сказанных кусков). Возвращается НОВЫЙ словарь с полями
    "tts_playing", "pending_chunks", "cancelled", "late". Входной state не
    меняется: его держит другой поток.

    barge_in({"tts_playing": True, "pending_chunks": ["a", "b"]}, True, 40)
        ->  {"tts_playing": False, "pending_chunks": [], "cancelled": True,
             "late": False}

    Отмена — это ДВА действия сразу: остановить проигрывание и выбросить
    очередь LLM. Если оставить очередь, бот замолчит на секунду и договорит
    старую фразу поверх пользователя.

    "late" — реакция дольше limit_ms. Формально мы всё отменили, но человек
    уже успел решить, что ассистент глухой; порог из урока — 100 мс.

    Пользователь молчит или TTS и так не играет — отмены нет.
    Отрицательный reaction_ms — ValueError.
    """
    if reaction_ms < 0:
        raise ValueError("reaction_ms must not be negative")
    new = {
        "tts_playing": state["tts_playing"],
        "pending_chunks": list(state["pending_chunks"]),
        "cancelled": False,
        "late": False,
    }
    if user_speaking and state["tts_playing"]:
        new["tts_playing"] = False
        new["pending_chunks"] = []  # без этой строки бот договорит поверх
        new["cancelled"] = True
        new["late"] = reaction_ms > limit_ms
    return new
