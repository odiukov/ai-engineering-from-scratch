"""
Синтез речи (TTS) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def normalize_text(text, abbreviations):
    """Текстовый фронтенд: строка → список произносимых токенов.

    normalize_text("Dr. Smith, 6 pm.", {"dr": "doctor", "pm": "p m"})
        ->  ["doctor", "smith", "6", "p", "m"]
    normalize_text("Hello!", {})  ->  ["hello"]

    Порядок шагов: разбить по пробелам, снять пунктуацию с краёв токена
    (.,!?;:"'()), привести к нижнему регистру, заменить по словарю
    abbreviations. Замена может дать несколько слов через пробел — тогда
    они попадают в результат отдельными токенами. Пустые токены выкинуть.

    Нормализация делается ДО фонемизатора: он не знает, что "Dr." — это
    "doctor", и прочитает его как "драйв".
    """
    out = []
    for raw in text.split():
        # пунктуация снимается только по краям: апостроф внутри "don't"
        # нести смысл продолжает, а точка в конце — нет
        token = raw.strip(".,!?;:\"'()").lower()
        if not token:
            continue
        # словарь может развернуть один токен в несколько слов
        out.extend(abbreviations.get(token, token).split())
    return out


def grapheme_to_phoneme(tokens, lexicon, fallback):
    """Слова → плоский список фонем. Незнакомые слова отдаём в fallback.

    grapheme_to_phoneme(["cat"], {"cat": ["K", "AE", "T"]}, lambda w: [])
        ->  ["K", "AE", "T"]
    grapheme_to_phoneme(["ghu"], {}, lambda w: list(w.upper()))
        ->  ["G", "H", "U"]

    fallback — заглушка вместо нейросетевой G2P-модели: настоящая
    предсказывает произношение незнакомых имён по буквам. Здесь её роль
    играет функция, которую передают параметром, чтобы тесты оставались
    воспроизводимыми.

    Если слово есть в lexicon, fallback вызывать НЕ надо: словарное
    произношение всегда точнее предсказанного.
    """
    phonemes = []
    for token in tokens:
        # lexicon.get(token) or fallback(token) — плохая идея: пустой
        # список из словаря она молча заменит на предсказание
        if token in lexicon:
            phonemes.extend(lexicon[token])
        else:
            phonemes.extend(fallback(token))
    return phonemes


def predict_durations(phonemes, ms_per_phoneme, default_ms=58.0, frame_ms=10.0):
    """Duration predictor: сколько mel-кадров занимает каждая фонема.

    predict_durations(["K", "AE", "T"], {"AE": 116.0})  ->  [5, 11, 5]
    predict_durations(["K"], {"K": 3.0})                ->  [1]

    Длительность фонемы в миллисекундах берём из ms_per_phoneme, для
    незнакомой — default_ms. Кадров = длительность / frame_ms, округляем
    ВНИЗ.

    Ловушка: округление вниз даёт 0 кадров на коротких фонемах — и фонема
    просто исчезает из речи. Минимум — один кадр.

    Это выход FastSpeech 2: неавторегрессионная модель предсказывает
    длительности заранее, поэтому весь mel считается за один проход.
    """
    durations = []
    for p in phonemes:
        ms = ms_per_phoneme.get(p, default_ms)
        # int() режет к нулю, для положительных это и есть floor
        frames = int(ms / frame_ms)
        durations.append(max(1, frames))
    return durations


def length_regulate(phoneme_vectors, durations):
    """Length regulator: растянуть по вектору на фонему в вектор на кадр.

    length_regulate([[1.0], [2.0]], [2, 3])
        ->  [[1.0], [1.0], [2.0], [2.0], [2.0]]
    length_regulate([[7.0]], [1])  ->  [[7.0]]

    Каждый вектор фонемы повторяется столько раз, сколько кадров ей выдал
    duration predictor. Длина результата обязана быть равна sum(durations)
    — это главное свойство функции.

    Две ловушки:
      * длины phoneme_vectors и durations могут не совпасть — это ValueError,
        а не молчаливый zip, который обрежет лишнее;
      * класть в результат один и тот же список нельзя: правка одного кадра
        поменяет все его копии. Копируй.
    """
    if len(phoneme_vectors) != len(durations):
        raise ValueError("phoneme_vectors and durations must have equal length")
    frames = []
    for vec, count in zip(phoneme_vectors, durations):
        for _ in range(count):
            # list(vec) — новая копия на каждый кадр, иначе все кадры
            # окажутся одним и тем же объектом
            frames.append(list(vec))
    return frames


def vocode(frames, hop, sample_rate):
    """Вокодер-игрушка: кадры (амплитуда, частота) → сэмплы волны.

    vocode([(1.0, 0.0)], 4, 8000)               ->  [0.0, 0.0, 0.0, 0.0]
    len(vocode([(0.5, 440.0), (0.5, 880.0)], 128, 16000))  ->  256

    Каждый кадр разворачивается в hop сэмплов синуса заданной амплитуды и
    частоты. hop — это шаг между mel-кадрами в сэмплах (у HiFi-GAN он 256).

    Ловушка: фазу НЕЛЬЗЯ обнулять на границе кадра. Если начинать каждый
    кадр с sin(0), на каждом стыке будет разрыв — щелчок. Фаза
    накапливается через всю волну.

    Настоящий вокодер делает ровно это отображение mel → волна, только
    вместо синуса у него свёрточная сеть.
    """
    wav = []
    phase = 0.0
    for amp, freq in frames:
        # приращение фазы за один сэмпл; частота в кадре постоянна,
        # поэтому шаг считаем один раз на кадр, а не на сэмпл
        step = 2.0 * math.pi * freq / sample_rate
        for _ in range(hop):
            wav.append(amp * math.sin(phase))
            phase += step
    return wav


def clip_waveform(wav, lo=-1.0, hi=1.0):
    """Ограничить сэмплы диапазоном. Вернуть (волна, сколько срезано).

    clip_waveform([0.5, 1.4, -2.0])  ->  ([0.5, 1.0, -1.0], 2)
    clip_waveform([0.1, -0.1])       ->  ([0.1, -0.1], 0)

    Счётчик считает сэмплы, которые реально пришлось подрезать. Ровно
    граничное значение (1.0 при hi=1.0) подрезать не надо.

    Зачем: рассинхрон mel-масштаба между обучением и инференсом
    выбрасывает вокодер за ±1.0. На слух это треск, а счётчик сразу
    показывает, что дело именно в этом, а не в модели.
    """
    out = []
    clipped = 0
    for x in wav:
        if x > hi:
            out.append(hi)
            clipped += 1
        elif x < lo:
            out.append(lo)
            clipped += 1
        else:
            out.append(x)
    return out, clipped


def resample_linear(wav, sr_in, sr_out):
    """Пересемплировать волну линейной интерполяцией.

    resample_linear([0.0, 1.0, 2.0, 3.0], 4, 2)  ->  [0.0, 2.0]
    resample_linear([0.0, 1.0], 1, 2)            ->  [0.0, 0.5, 1.0, 1.0]

    Выходных сэмплов: len(wav) * sr_out / sr_in, округляем вниз. Позиция
    i-го выходного сэмпла в исходной сетке — i * sr_in / sr_out, значение
    берём линейной интерполяцией между соседями. Если правый сосед за
    концом массива — тянем последний сэмпл.

    sr_in или sr_out <= 0 — это ValueError, а не деление на ноль.

    Kokoro отдаёт 24 kHz, а ASR ниже по конвейеру ждёт 16 kHz. Без
    пересемплирования получишь речь не той скорости и не той высоты.
    """
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    ratio = sr_in / sr_out
    n_out = int(len(wav) * sr_out / sr_in)
    out = []
    for i in range(n_out):
        pos = i * ratio
        left = int(pos)
        frac = pos - left
        a = wav[left] if left < len(wav) else wav[-1]
        b = wav[left + 1] if left + 1 < len(wav) else a
        out.append(a + (b - a) * frac)
    return out


def character_error_rate(reference, hypothesis):
    """CER: расстояние Левенштейна, делённое на длину reference.

    character_error_rate("cat", "cat")  ->  0.0
    character_error_rate("cat", "cut")  ->  примерно 0.333
    character_error_rate("cat", "")     ->  1.0
    character_error_rate("", "")        ->  0.0

    Вставка, удаление и замена стоят по 1. Пустой reference: если и
    hypothesis пуст — 0.0, иначе 1.0 (делить на ноль нельзя).

    CER может быть БОЛЬШЕ 1.0: если синтез наболтал вдвое больше текста,
    чем просили, вставок окажется больше, чем символов в эталоне.

    Так меряют разборчивость TTS: прогнали синтез через Whisper и сравнили
    расшифровку с исходным текстом. CER выше 5% — синтез невнятный.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    # две строки таблицы вместо полной матрицы: память O(len(hypothesis)),
    # а не O(len(reference) * len(hypothesis))
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        cur = [i]
        for j, h in enumerate(hypothesis, start=1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(reference)
