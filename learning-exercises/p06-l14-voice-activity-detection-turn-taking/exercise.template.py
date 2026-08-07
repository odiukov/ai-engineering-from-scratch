"""
Детекция речи и передача хода

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p06-l14-voice-activity-detection-turn-taking
Разбор:  /check-code p06-l14-voice-activity-detection-turn-taking
"""

import math


def rms(chunk):
    """Среднеквадратичная амплитуда куска аудио.

    rms([1.0, -1.0, 1.0, -1.0])  ->  1.0
    rms([0.0, 0.0])              ->  0.0

    Именно RMS, а не среднее: у речи среднее около нуля, потому что
    колебание симметрично, и по нему тишину от крика не отличить.

    Пустой chunk — ValueError.
    """
    raise NotImplementedError


def dbfs(chunk):
    """Громкость куска в dBFS: 20 * log10(rms), 0 dBFS — максимум шкалы.

    dbfs([1.0, -1.0])   ->  0.0
    dbfs([0.1, -0.1])   ->  -20.0
    dbfs([0.0, 0.0])    ->  -200.0

    Ловушка: log10(0) — это ValueError у math и минус бесконечность по
    математике. Цифровая тишина встречается постоянно, поэтому rms
    зажимают снизу на 1e-10, что и даёт пол в -200 dBFS.
    """
    raise NotImplementedError


def energy_vad(chunk, threshold_dbfs=-40.0):
    """Первый ярус каскада: речь ли это по одной только громкости.

    energy_vad([0.3, -0.3])       ->  True
    energy_vad([0.0005, -0.0005]) ->  False

    Самый дешёвый детектор и самый глупый: он одинаково срабатывает на
    речь, на кашель, на стук клавиатуры и на проезжающую машину. В 2026
    году он годится только как отсев заведомой тишины перед Silero.
    """
    raise NotImplementedError


def hysteresis_flags(probs, on_threshold=0.5, off_threshold=0.35):
    """Порог с гистерезисом: включаемся по on_threshold, выключаемся по off_threshold.

    hysteresis_flags([0.6, 0.4, 0.2])  ->  [True, True, False]
    hysteresis_flags([0.4, 0.6])       ->  [False, True]

    Зачем два порога вместо одного: вероятность от Silero на границе речи
    колеблется вокруг 0.5. С одним порогом флаг дребезжит на каждом
    фрейме, и турн-детектор сходит с ума. С гистерезисом переход
    происходит один раз.

    off_threshold > on_threshold — ValueError: такая пара порогов
    перевернула бы логику наизнанку.
    """
    raise NotImplementedError


def pre_roll(chunks, pre_roll_ms, chunk_ms=20):
    """Хвост буфера длиной pre_roll_ms: аудио ДО того, как сработал VAD.

    pre_roll(["a", "b", "c", "d"], 40)  ->  ["c", "d"]
    pre_roll(["a", "b"], 200)           ->  ["a", "b"]
    pre_roll(["a", "b"], 0)             ->  []

    Без пре-ролла у пользователя срезается первое слово: VAD принимает
    решение уже после того, как звук начался.

    Ловушка: количество кусков считается округлением ВВЕРХ, и при нулевом
    пре-ролле срез chunks[-0:] вернёт весь список целиком, а не пустой.

    Отрицательный pre_roll_ms или неположительный chunk_ms — ValueError.
    """
    raise NotImplementedError


def flush_latency_ms(lookahead_ms, speedup):
    """Задержка после flush-сигнала: буфер look-ahead, дожатый на скорости speedup.

    flush_latency_ms(500.0, 4.0)   ->  125.0
    flush_latency_ms(2500.0, 4.0)  ->  625.0

    Трюк Kyutai: у стримингового STT есть look-ahead (500 мс у STT-1B).
    Обычно ты ждёшь эти 500 мс реального времени. Но STT считает в 4 раза
    быстрее реального времени, и если по концу речи послать flush, буфер
    досчитается за 125 мс.

    speedup <= 0 или lookahead_ms < 0 — ValueError.
    """
    raise NotImplementedError


class TurnDetector:
    """Конечный автомат передачи хода: START в начале реплики, END в конце.

    Кормится побитово, по одному флагу VAD на кусок:

        td = TurnDetector(silence_hangover_ms=500, min_speech_ms=250)
        td.update(True)   ->  None      (20 мс речи — мало для START)
        ... ещё 12 раз True ...
        td.update(True)   ->  "START"   (набралось 260 мс >= 250)
        ... 24 раза False ...
        td.update(False)  ->  "END"     (набралось 500 мс тишины)

    Три смысла параметров:
      * min_speech_ms — сколько речи подряд нужно, чтобы поверить, что это
        реплика, а не кашель. 250 мс.
      * silence_hangover_ms — сколько тишины ждать перед тем, как объявить
        конец хода. 500-800 мс: короче — перебиваешь человека, длиннее —
        агент кажется тормозом.
      * chunk_ms — длительность одного куска, обычно 20 мс.

    Ловушка: счётчик речи надо обнулять на тишине в состоянии idle. Иначе
    тринадцать кашлей, разбросанных по минуте, сложатся в 260 мс и выдадут
    ложный START.
    """

    def __init__(self, silence_hangover_ms=500, min_speech_ms=250, chunk_ms=20):
        """Завести автомат в состоянии idle с нулевыми счётчиками."""
        raise NotImplementedError

    def update(self, is_speech):
        """Скормить один флаг VAD. Вернуть "START", "END" или None."""
        raise NotImplementedError
