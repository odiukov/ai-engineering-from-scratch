"""
Chameleon и ранняя фузия: картинка как токены — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Игрушечный общий словарь из урока: текст, картинка и разделители в одном
# пространстве целых id. У настоящего Chameleon это 32000 + 8192 + 2.
TEXT_VOCAB = 32  # текстовые id 0..31
IMAGE_VOCAB = 16  # индексы кодовой книги, в общем словаре это 32..47
BOI = TEXT_VOCAB + IMAGE_VOCAB  # 48, открывающий тег <image>
EOI = BOI + 1  # 49, закрывающий тег </image>


def nearest_code(vector, codebook):
    """Индекс ближайшей записи кодовой книги по евклидову расстоянию.

    Это и есть квантование в VQ-VAE: непрерывный признак заменяется целым
    номером записи.

    nearest_code([0.9, 0.1], [[0.0, 0.0], [1.0, 0.0]])  ->  1
    nearest_code([0.4, 0.0], [[0.0, 0.0], [1.0, 0.0]])  ->  0

    Корень из суммы квадратов считать не нужно: sqrt монотонен, и argmin по
    квадрату расстояния тот же самый. Экономия — половина времени функции.

    При равном расстоянии берём МЕНЬШИЙ индекс: иначе токенизация одного и
    того же изображения будет плавать от запуска к запуску.

    Пустая книга или несовпадение размерностей — ValueError.
    """
    if not codebook:
        raise ValueError("пустая кодовая книга")
    best_index, best_distance = None, None
    for index, code in enumerate(codebook):
        if len(code) != len(vector):
            raise ValueError(
                f"запись {index} имеет размерность {len(code)}, а вектор {len(vector)}"
            )
        distance = sum((a - b) ** 2 for a, b in zip(vector, code))
        # строгое < сохраняет первый из равных, то есть меньший индекс
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def quantize(vectors, codebook):
    """Превратить список признаков в список индексов кодовой книги.

    quantize([[0.9, 0.1], [0.1, 0.0]], [[0.0, 0.0], [1.0, 0.0]])  ->  [1, 0]

    После этого шага картинка перестаёт быть картинкой и становится
    последовательностью целых — ровно тем, что трансформер умеет предсказывать.
    """
    return [nearest_code(v, codebook) for v in vectors]


def dequantize(indices, codebook):
    """Вернуть по индексам сами векторы кодовой книги.

    dequantize([1, 0], [[0.0, 0.0], [1.0, 0.0]])  ->  [[1.0, 0.0], [0.0, 0.0]]

    Отдавай КОПИИ, а не сами записи книги: иначе тот, кто поправит результат
    на месте, незаметно испортит кодовую книгу для всех следующих картинок.

    Индекс за границами книги — IndexError (это и так сделает список).
    """
    return [list(codebook[i]) for i in indices]


def reconstruction_mse(vectors, codebook):
    """Средний квадрат ошибки квантования на одну координату.

    reconstruction_mse([[1.0, 0.0]], [[1.0, 0.0]])  ->  0.0
    reconstruction_mse([[0.5, 0.0]], [[0.0, 0.0], [1.0, 0.0]])  ->  0.125

    Это численный аналог «потолка реконструкции» из урока: сколько бы ни
    была умна модель поверх токенов, качество картинки ограничено этой
    ошибкой. Больше записей в книге — ошибка не растёт никогда.

    Пустой список векторов — ValueError.
    """
    if not vectors:
        raise ValueError("нет векторов для оценки")
    codes = dequantize(quantize(vectors, codebook), codebook)
    total = 0.0
    count = 0
    for original, code in zip(vectors, codes):
        for a, b in zip(original, code):
            total += (a - b) ** 2
            count += 1
    return total / count


def compression_ratio(width, height, n_tokens, codebook_size):
    """Во сколько раз VQ-токены легче сырого 24-битного RGB.

    Сырых бит: width * height * 24. Бит в токенах: n_tokens * log2(K).

    compression_ratio(512, 512, 1024, 8192)   ->  472.6   (Chameleon)
    compression_ratio(512, 512, 4096, 32768)  ->  102.4   (Emu3)

    Числа показывают компромисс урока целиком: Emu3 сжимает вчетверо
    слабее, зато его реконструкция ближе к диффузии. Сжатие всегда с
    потерями — обратно пиксели восстанавливаются лишь приблизительно.

    Неположительные размеры, нет токенов или книга меньше двух записей —
    ValueError (при K = 1 log2 обнулится и деление сорвётся).
    """
    if width <= 0 or height <= 0 or n_tokens <= 0:
        raise ValueError("размеры и число токенов должны быть положительными")
    if codebook_size < 2:
        raise ValueError("кодовая книга должна содержать хотя бы 2 записи")
    raw_bits = width * height * 24
    token_bits = n_tokens * math.log2(codebook_size)
    return raw_bits / token_bits


def encode_document(parts):
    """Собрать смешанный документ в одну последовательность общего словаря.

    parts — список кусков ("text", [id...]) и ("image", [код...]).

    encode_document([("text", [1, 2]), ("image", [0, 5])])
        ->  [1, 2, 48, 32, 37, 49]

    Текст ложится как есть, коды картинки сдвигаются на TEXT_VOCAB и
    оборачиваются в BOI/EOI. Сдвиг обязателен: без него код 5 неотличим от
    текстового токена 5, и общий словарь превращается в кашу.

    Текстовый id вне [0, TEXT_VOCAB), код вне [0, IMAGE_VOCAB) или
    неизвестный тип куска — ValueError.
    """
    ids = []
    for part in parts:
        kind, values = part[0], part[1]
        if kind == "text":
            for v in values:
                if not 0 <= v < TEXT_VOCAB:
                    raise ValueError(f"текстовый id {v} вне словаря")
            ids.extend(values)
        elif kind == "image":
            for v in values:
                if not 0 <= v < IMAGE_VOCAB:
                    raise ValueError(f"код картинки {v} вне кодовой книги")
            ids.append(BOI)
            ids.extend(TEXT_VOCAB + v for v in values)
            ids.append(EOI)
        else:
            raise ValueError(f"неизвестный тип куска {kind!r}")
    return ids


def decode_document(ids):
    """Разобрать поток id общего словаря обратно на куски.

    decode_document([1, 2, 48, 32, 37, 49])
        ->  [("text", [1, 2]), ("image", [0, 5])]

    Подряд идущие текстовые токены собираются в ОДИН кусок — иначе обратная
    сборка перестанет совпадать с исходной разбивкой.

    Именно этим занимается софт вокруг Chameleon на генерации: увидел BOI —
    значит следующие токены надо отдать декодеру VQ-VAE, а не текстовому
    детокенизатору.

    EOI без BOI, вложенный BOI, незакрытая картинка, текстовый id внутри
    картинки — всё это ValueError: битую последовательность лучше не
    «чинить» молча, иначе в пиксели уедет мусор.
    """
    parts = []
    buffer = []
    in_image = False
    for token in ids:
        if token == BOI:
            if in_image:
                raise ValueError("вложенный BOI")
            if buffer:
                parts.append(("text", buffer))
                buffer = []
            in_image = True
        elif token == EOI:
            if not in_image:
                raise ValueError("EOI без BOI")
            parts.append(("image", buffer))
            buffer = []
            in_image = False
        elif in_image:
            if not TEXT_VOCAB <= token < TEXT_VOCAB + IMAGE_VOCAB:
                raise ValueError(f"id {token} не является кодом картинки")
            buffer.append(token - TEXT_VOCAB)
        else:
            if not 0 <= token < TEXT_VOCAB:
                raise ValueError(f"id {token} не является текстовым токеном")
            buffer.append(token)
    if in_image:
        raise ValueError("незакрытая картинка")
    if buffer:
        parts.append(("text", buffer))
    return parts


def qk_norm(vec, eps=1e-5):
    """LayerNorm без обучаемых параметров: (x - среднее) / корень(дисперсия + eps).

    qk_norm([1.0, 3.0])  ->  примерно [-1.0, 1.0]
    qk_norm([5.0, 5.0])  ->  [0.0, 0.0]   (постоянный вектор, не NaN)

    Дисперсия считается по n, а не по n-1: это нормализация активаций, а не
    оценка параметра выборки.

    eps здесь не косметика. У постоянного вектора дисперсия ровно ноль, и
    без eps получится 0/0. В обучении такое случается на мёртвых головах.

    Зачем это Chameleon: после нормировки длина q и k фиксирована, поэтому
    их скалярное произведение не может превысить размерность по модулю.
    Без QK-Norm логиты внимания растут с глубиной, softmax насыщается,
    градиенты исчезают, и обучение 34B расходится — ровно то, что описано
    в разделе про стабильность.

    Пустой вектор — ValueError.
    """
    if not vec:
        raise ValueError("пустой вектор")
    mean = sum(vec) / len(vec)
    variance = sum((x - mean) ** 2 for x in vec) / len(vec)
    scale = math.sqrt(variance + eps)
    return [(x - mean) / scale for x in vec]
