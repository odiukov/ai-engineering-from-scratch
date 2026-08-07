"""
Нейронные аудиокодеки — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def nearest_code(codebook, value):
    """Индекс ближайшего кода в codebook к числу value.

    nearest_code([-1.0, 0.0, 1.0], 0.4)   ->  1
    nearest_code([-1.0, 0.0, 1.0], 0.6)   ->  2
    nearest_code([0.0, 1.0], 0.5)         ->  0   (ничья — берём меньший индекс)

    Возвращать нужно ИНДЕКС, а не сам код: кодек передаёт по сети именно
    номера, 10 бит на кодовое слово при codebook из 1024 элементов.

    На пустом codebook — ValueError, иначе ошибка вылезет позже и в другом месте.
    """
    if not codebook:
        raise ValueError("codebook пустой")
    # min по ключу возвращает ПЕРВЫЙ минимум, значит ничья решается
    # в пользу меньшего индекса — это и нужно для воспроизводимости
    return min(range(len(codebook)), key=lambda i: abs(codebook[i] - value))


def uniform_codebook(size, scale=1.0):
    """Равномерный codebook из size кодов, симметричный, от -scale до +scale.

    uniform_codebook(3, 1.0)  ->  [-1.0, 0.0, 1.0]
    uniform_codebook(5, 2.0)  ->  [-2.0, -1.0, 0.0, 1.0, 2.0]
    uniform_codebook(1, 7.0)  ->  [0.0]

    Настоящие кодеки codebook не задают формулой, а обучают. Но у
    равномерного есть свойство, ради которого он здесь: при нечётном size
    в нём есть ровно 0.0, и тогда квантование НИКОГДА не увеличивает
    остаток. На этом держится вся каскадная схема RVQ.

    size < 1 — ValueError.
    """
    if size < 1:
        raise ValueError("size должен быть >= 1")
    if size == 1:
        # один код — единственный осмысленный выбор это ноль,
        # деление на (size - 1) ниже дало бы ZeroDivisionError
        return [0.0]
    step = 2.0 * scale / (size - 1)
    return [-scale + step * i for i in range(size)]


def quantize_layer(values, codebook):
    """Один слой квантования: вернуть (indices, residuals).

    quantize_layer([0.4, -0.9], [-1.0, 0.0, 1.0])  ->  ([1, 0], [0.4, 0.1])

    indices — что кодек передаёт, residuals — то, что он НЕ передал.
    Именно residuals уходят в следующий codebook каскада: отсюда и
    название Residual Vector Quantization.
    """
    indices = []
    residuals = []
    for v in values:
        i = nearest_code(codebook, v)
        indices.append(i)
        residuals.append(v - codebook[i])
    return indices, residuals


def rvq_encode(values, codebooks):
    """Каскад RVQ: первый codebook кодирует сигнал, каждый следующий — остаток.

    rvq_encode([0.4], [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])  ->  [[1], [1]]

    Возвращает список списков индексов — по одному списку на codebook.
    Длина результата равна len(codebooks), длина каждого списка — len(values).

    Ловушка: в следующий слой уходит ОСТАТОК, а не исходный сигнал. Если
    подать исходный, все слои посчитают одно и то же и качество не вырастет.
    """
    residuals = list(values)
    all_indices = []
    for cb in codebooks:
        indices, residuals = quantize_layer(residuals, cb)
        all_indices.append(indices)
    return all_indices


def rvq_decode(all_indices, codebooks, length):
    """Декодер RVQ: сумма выбранных кодов по всем слоям.

    rvq_decode([[1], [2]], [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], 1)  ->  [0.5]
    rvq_decode([], [], 3)                                            ->  [0.0, 0.0, 0.0]

    Декодер не «выбирает лучший» слой — он складывает все. Поэтому можно
    оборвать каскад на любом слое: получится тот же сигнал, только грубее.
    Это и есть переменный битрейт у EnCodec одной и той же моделью.
    """
    out = [0.0] * length
    for indices, cb in zip(all_indices, codebooks):
        for i, idx in enumerate(indices):
            out[i] += cb[idx]
    return out


def reconstruction_mse(original, reconstructed):
    """Средняя квадратичная ошибка восстановления.

    reconstruction_mse([1.0, 2.0], [1.0, 2.0])  ->  0.0
    reconstruction_mse([0.0, 0.0], [1.0, -1.0]) ->  1.0

    Разная длина — ValueError: молчаливое обрезание по zip спрячет баг,
    когда декодер вернул не столько отсчётов, сколько было на входе.
    """
    if len(original) != len(reconstructed):
        raise ValueError("длины не совпадают")
    if not original:
        raise ValueError("пустой сигнал")
    return sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / len(original)


def codec_cost(seconds, frame_rate_hz, n_codebooks, codebook_size):
    """Цена клипа в кодеке: сколько фреймов, токенов и бит в секунду.

    Вернуть словарь с ключами "frames", "tokens", "bitrate_bps".

    codec_cost(10, 12.5, 8, 1024)
        ->  {"frames": 125.0, "tokens": 1000.0, "bitrate_bps": 1000.0}
    codec_cost(1, 75.0, 8, 1024)
        ->  {"frames": 75.0, "tokens": 600.0, "bitrate_bps": 6000.0}

    frames = seconds * frame_rate_hz
    tokens = frames * n_codebooks          — длина последовательности для LM
    bitrate_bps = frame_rate_hz * n_codebooks * log2(codebook_size)

    Ради этого числа и придумали Mimi: 10 секунд речи — 1000 токенов,
    трансформер такой контекст не замечает. У EnCodec-24k на 75 Hz те же
    10 секунд дали бы 6000 токенов.

    codebook_size < 2 — ValueError: один код не несёт ни бита.
    """
    if seconds < 0 or frame_rate_hz <= 0 or n_codebooks < 1:
        raise ValueError("некорректные параметры кодека")
    if codebook_size < 2:
        raise ValueError("codebook_size должен быть >= 2")
    frames = seconds * frame_rate_hz
    bits_per_code = math.log2(codebook_size)
    return {
        "frames": frames,
        "tokens": frames * n_codebooks,
        "bitrate_bps": frame_rate_hz * n_codebooks * bits_per_code,
    }


def split_semantic_acoustic(frames):
    """Разрезать последовательность фреймов на semantic и acoustic части.

    frames — список фреймов, каждый фрейм — список кодов всех codebook'ов.

    split_semantic_acoustic([[5, 1, 2], [7, 3, 4]])  ->  ([5, 7], [[1, 2], [3, 4]])

    В Mimi codebook 0 дистиллирован из WavLM и несёт содержание — что
    сказано. Codebook'и 1..7 несут тембр, просодию, шум. Text-to-semantic
    модель предсказывает первый, acoustic-декодер — остальные; отсюда
    zero-shot клонирование голоса.

    Пустой фрейм — ValueError: semantic-кода взять неоткуда.
    """
    semantic = []
    acoustic = []
    for frame in frames:
        if not frame:
            raise ValueError("фрейм без кодов")
        semantic.append(frame[0])
        acoustic.append(list(frame[1:]))
    return semantic, acoustic
