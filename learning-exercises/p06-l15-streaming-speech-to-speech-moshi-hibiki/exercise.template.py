"""
Стриминговый speech-to-speech: Moshi и Hibiki

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p06-l15-streaming-speech-to-speech-moshi-hibiki
Разбор:  /check-code p06-l15-streaming-speech-to-speech-moshi-hibiki
"""


def frame_ms_from_rate(frame_rate_hz):
    """Длительность одного кадра кодека в миллисекундах.

    frame_ms_from_rate(12.5)  ->  80.0   (Mimi: 12.5 Hz — это кадр 80 мс)
    frame_ms_from_rate(25.0)  ->  40.0

    Кадр Mimi — единица времени всей архитектуры Moshi: за один кадр
    трансформер успевает прочитать пользователя и выдать себя.

    frame_rate_hz <= 0 — ValueError.
    """
    raise NotImplementedError


def tokens_per_second(frame_rate_hz, codebooks, streams=1):
    """Сколько акустических токенов в секунду тащит трансформер.

    tokens_per_second(12.5, 8)             ->  100.0   (один поток Mimi)
    tokens_per_second(12.5, 8, streams=2)  ->  200.0   (пользователь + Moshi)

    Full-duplex стоит ровно вдвое: Moshi слушает свой собственный выход
    так же, как чужой вход, иначе он не сможет себя перебить.

    Любой неположительный аргумент — ValueError.
    """
    raise NotImplementedError


def theoretical_latency_ms(frame_ms=80.0, acoustic_delay_frames=1):
    """Нижняя граница задержки full-duplex модели: кадр плюс акустическая задержка.

    theoretical_latency_ms()            ->  160.0   (80 мс кадр + 80 мс задержка)
    theoretical_latency_ms(80.0, 0)     ->  80.0
    theoretical_latency_ms(40.0, 2)     ->  120.0

    Один кадр уходит на то, чтобы услышать, ещё acoustic_delay_frames —
    на то, чтобы акустические коды успели за текстом.

    frame_ms <= 0 или acoustic_delay_frames < 0 — ValueError.
    """
    raise NotImplementedError


def pipeline_latency_ms(stage_ms):
    """Задержка классического конвейера: стадии складываются, а не перекрываются.

    pipeline_latency_ms([30.0, 120.0, 200.0, 90.0])  ->  440.0
    pipeline_latency_ms([])                          ->  0.0

    Это и есть потолок из уроков 11-12: VAD -> STT -> LLM -> TTS. Каждая
    стадия ждёт предыдущую целиком, поэтому сумма — не среднее и не
    максимум.

    Отрицательная стадия — ValueError.
    """
    raise NotImplementedError


def build_frame(text_token, acoustic_tokens, n_codebooks=8):
    """Один кадр Moshi: сначала токен внутреннего монолога, потом кодбуки.

    build_frame("да", [1, 2, 3], n_codebooks=3)  ->  ["да", 1, 2, 3]
    build_frame("", [0, 0], n_codebooks=2)       ->  ["", 0, 0]

    Порядок принципиален: текст предсказывается ПЕРЕД аудио. Внутренний
    монолог задаёт, что будет сказано, а кодбуки — как это прозвучит.
    Перевернёшь — потеряешь весь смысл inner monologue.

    Длина acoustic_tokens не равна n_codebooks — ValueError.
    """
    raise NotImplementedError


def depth_decode(context, heads):
    """Depth transformer: кодбуки одного кадра предсказываются по очереди.

    heads — список функций head(context, already_decoded) -> токен. Каждая
    следующая голова видит всё, что предсказали предыдущие.

    depth_decode("ctx", [lambda c, p: len(p)] * 3)   ->  [0, 1, 2]
    depth_decode("ctx", [lambda c, p: sum(p) + 1] * 3)  ->  [1, 2, 4]

    Почему не параллельно: кодбуки residual-квантизации зависят друг от
    друга — второй уточняет ошибку первого. Предскажешь все восемь разом —
    получишь рассогласованный кадр.

    Пустой список heads — ValueError: кадр без кодбуков не бывает.
    """
    raise NotImplementedError


class DuplexSession:
    """Full-duplex сессия: за один кадр и слушаем, и говорим.

        s = DuplexSession(frame_ms=80.0, n_codebooks=2)
        s.step([7, 7], "привет", [1, 2])  ->  ["привет", 1, 2]
        s.step([8, 8], "", [3, 4])        ->  ["", 3, 4]
        s.elapsed_ms()                    ->  160.0
        s.transcript()                    ->  "привет"

    Отличие от конвейера: тут нет стадии, которая ждёт другую. Кадр
    пользователя и кадр модели живут на одном и том же индексе времени,
    поэтому Moshi слышит перебивание в тот же кадр, в который говорит сам.

    pad_token — «в этом кадре модель молчит»: во внутреннем монологе таких
    кадров большинство, и в транскрипт они не попадают.
    """

    def __init__(self, frame_ms=80.0, n_codebooks=8, pad_token=""):
        """Пустая сессия: оба потока и внутренний монолог начинаются с нуля."""
        raise NotImplementedError

    def step(self, user_acoustic, own_text, own_acoustic):
        """Обработать один кадр: принять кадр пользователя, вернуть свой кадр."""
        raise NotImplementedError

    def elapsed_ms(self):
        """Сколько времени прошло по сессии: кадры на длительность кадра."""
        raise NotImplementedError

    def transcript(self):
        """Внутренний монолог как текст: непустые токены через пробел."""
        raise NotImplementedError
