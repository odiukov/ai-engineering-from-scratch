"""
Анти-спуфинг и водяные знаки в аудио — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
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
    if not spec:
        raise ValueError("пустой спектр")
    if not 0 < percentile <= 1:
        raise ValueError("percentile должен лежать в (0, 1]")
    if any(v < 0 for v in spec):
        raise ValueError("энергия бина не может быть отрицательной")
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    cum = 0.0
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    # сюда попадаем только из-за накопленной ошибки float на последнем бине
    return len(spec) - 1


def is_suspicious(spec, ratio=0.92, percentile=0.85):
    """Игрушечный детектор дипфейка: rolloff подозрительно близко к верху.

    is_suspicious([0.01] * 90 + [10] * 10)          ->  True
    is_suspicious([1 / (k + 1) for k in range(100)]) ->  False

    Так делать в проде нельзя: настоящий countermeasure — это AASIST или
    RawNet2, обученные на ASVspoof. Здесь важна интуиция: у генеративной
    модели верх спектра слишком «живой».
    """
    return spectral_rolloff(spec, percentile) / len(spec) > ratio


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
    if length <= 0:
        raise ValueError("длина последовательности должна быть положительной")
    rng = random.Random(seed)
    return [rng.choice((-1.0, 1.0)) for _ in range(length)]


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
    n_bits = len(payload)
    if n_bits == 0:
        raise ValueError("пустой payload")
    if any(b not in (0, 1) for b in payload):
        raise ValueError("payload состоит из битов 0 и 1")
    seg = len(signal) // n_bits
    if seg == 0:
        raise ValueError("сигнал короче payload")
    chips = chip_sequence(seg * n_bits, seed)
    out = list(signal)  # копия: исходную запись портить нельзя
    for i, bit in enumerate(payload):
        sign = 1.0 if bit == 1 else -1.0
        for j in range(seg):
            k = i * seg + j
            out[k] = signal[k] + strength * sign * chips[k]
    return out


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
    if not signal:
        raise ValueError("пустой сигнал")
    power = sum(x * x for x in signal) / len(signal)
    sigma = math.sqrt(power / (10 ** (snr_db / 10.0)))
    return [x + rng.gauss(0.0, sigma) for x in signal]


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
    if n_bits <= 0:
        raise ValueError("n_bits должен быть положительным")
    seg = len(signal) // n_bits
    if seg == 0:
        raise ValueError("сигнал короче n_bits")
    chips = chip_sequence(seg * n_bits, seed)
    bits = []
    total_abs = 0.0
    for i in range(n_bits):
        start = i * seg
        corr = sum(signal[start + j] * chips[start + j] for j in range(seg)) / seg
        bits.append(1 if corr > 0 else 0)
        total_abs += abs(corr)
    score = min(1.0, (total_abs / n_bits) / strength)
    return score, bits


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
    if len(sent) != len(recovered):
        raise ValueError("длины payload не совпадают")
    if not sent:
        raise ValueError("пустой payload")
    return sum(1 for a, b in zip(sent, recovered) if a == b) / len(sent)


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
    if not real_scores or not fake_scores:
        raise ValueError("нужны обе выборки")
    best_gap = None
    best_rate = 0.0
    for t in sorted(set(list(real_scores) + list(fake_scores))):
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap, best_rate = gap, (far + frr) / 2
    return best_rate
