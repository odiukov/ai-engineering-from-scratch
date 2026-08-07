"""
Анти-спуфинг и водяные знаки в аудио

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p06-l16-anti-spoofing-audio-watermarking
Разбор:  /check-code p06-l16-anti-spoofing-audio-watermarking
"""

import math
import random


def spectral_rolloff(spec, percentile=0.85):
    """Номер бина, ниже которого лежит percentile всей энергии спектра.

    spectral_rolloff([10, 0, 0, 0])          ->  0   (вся энергия внизу)
    spectral_rolloff([1, 1, 1, 1], 0.5)      ->  1
    spectral_rolloff([0, 0, 0])              ->  0   (тишина: считать нечего)

    Синтетическая речь часто «размазана» по верху спектра, у живой энергия
    валится с частотой. Rolloff — самый дешёвый способ это заметить.

    Пустой spec, отрицательная энергия или percentile вне (0, 1] — ValueError.
    """
    raise NotImplementedError


def is_suspicious(spec, ratio=0.92, percentile=0.85):
    """Игрушечный детектор дипфейка: rolloff подозрительно близко к верху.

    is_suspicious([0.01] * 90 + [10] * 10)          ->  True
    is_suspicious([1 / (k + 1) for k in range(100)]) ->  False

    Так делать в проде нельзя: настоящий countermeasure — это AASIST или
    RawNet2, обученные на ASVspoof. Здесь важна интуиция: у генеративной
    модели верх спектра слишком «живой».
    """
    raise NotImplementedError


def chip_sequence(length, seed=0):
    """Псевдослучайная последовательность из +1 и -1 — ключ водяного знака.

    chip_sequence(4, seed=1) == chip_sequence(4, seed=1)   ->  True
    set(chip_sequence(100))                                ->  {1.0, -1.0}

    Это тот самый «секрет», который знают генератор и детектор AudioSeal и
    не знает атакующий. Без ключа корреляция даёт шум около нуля.

    Обязательно принимает seed: детектор должен уметь повторить в точности
    ту же последовательность, иначе водяной знак не найдётся.

    length <= 0 — ValueError.
    """
    raise NotImplementedError


def embed_watermark(signal, payload, strength=0.05, seed=0):
    """Встроить биты payload в сигнал: сигнал делится на равные сегменты, по биту на сегмент.

    Внутри сегмента ко всем отсчётам прибавляется strength * chip * знак
    бита: +1 для единицы, -1 для нуля.

    embed_watermark([0.0] * 4, [1], strength=0.5, seed=1)  ->  четыре +-0.5
    len(embed_watermark(sig, payload)) == len(sig)         ->  True

    Почему сложение, а не замена: водяной знак обязан быть неслышимым.
    strength порядка 0.05 — это -26 dB относительно полной шкалы, ухо этого
    не ловит, а корреляция по тысячам отсчётов ловит уверенно.

    Хвост, не поместившийся в целое число сегментов, остаётся чистым.

    Пустой payload, биты не из {0, 1} или сигнал короче payload — ValueError.
    """
    raise NotImplementedError


def add_noise(signal, snr_db, rng):
    """Атака на водяной знак: подмешать белый шум с заданным SNR в дБ.

    add_noise([1.0] * 4, 10.0, random.Random(0))  ->  четыре числа около 1.0

    SNR 10 дБ означает, что мощность шума в 10 раз меньше мощности сигнала.
    AudioSeal обязан переживать +10 дБ, и это тот минимум, на котором стоит
    проверять свою реализацию.

    rng — обязательный аргумент, а не глобальный random: иначе тест на
    робастность нельзя повторить.

    Пустой сигнал — ValueError.
    """
    raise NotImplementedError


def detect_watermark(signal, n_bits, strength=0.05, seed=0):
    """Достать водяной знак: вернуть (score, bits).

    score — уверенность в [0, 1], bits — восстановленный payload.

    detect_watermark(embed_watermark(sig, [1, 0, 1]), 3)  ->  (~1.0, [1, 0, 1])
    detect_watermark(sig, 3)                              ->  (около 0, мусор)

    Как работает: корреляция сегмента с тем же chip-ключом. У помеченного
    сегмента она равна примерно strength со знаком бита, у чистого — шум
    около нуля. Отсюда и score: средний модуль корреляции, поделённый на
    strength и зажатый единицей.

    Ловушка: знак корреляции несёт бит, а модуль — факт наличия знака. Если
    брать только знак, детектор «найдёт» водяной знак в любой записи.

    n_bits <= 0 или сигнал короче n_bits — ValueError.
    """
    raise NotImplementedError


def bit_recovery_accuracy(sent, recovered):
    """Bit Recovery Accuracy: доля битов, переживших атаку.

    bit_recovery_accuracy([1, 0, 1], [1, 0, 1])  ->  1.0
    bit_recovery_accuracy([1, 0, 1], [0, 1, 0])  ->  0.0
    bit_recovery_accuracy([1, 0], [1, 1])        ->  0.5

    0.5 — это не «половина сохранилась», а «знака нет вовсе»: столько же
    даст подбрасывание монетки. Именно ниже 0.6 падают все водяные знаки
    под pitch-shift по AudioMarkBench.

    Разная длина или пустые списки — ValueError.
    """
    raise NotImplementedError


def eer(real_scores, fake_scores):
    """Equal Error Rate: точка, где FAR и FRR сходятся.

    eer([0.9, 0.95], [0.1, 0.2])  ->  0.0   (детектор разделяет идеально)
    eer([0.1, 0.2], [0.9, 0.95])  ->  1.0   (детектор перевёрнут)

    FAR — доля фейков со score >= порога, FRR — доля живых со score < порога.
    Перебираем все встречающиеся значения как пороги и берём ту точку, где
    разрыв между ошибками минимален.

    Одна метрика вместо пары «точность/полнота» — потому что порог у
    биометрии всё равно двигают под нагрузку.

    Пустая любая из выборок — ValueError.
    """
    raise NotImplementedError
