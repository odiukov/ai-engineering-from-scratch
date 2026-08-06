"""
Генерация звука: нейронный кодек и токенный авторегрессор — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def codec_token_count(seconds, frame_rate=75, num_codebooks=8):
    """Сколько дискретных токенов даёт нейронный кодек на клип длиной seconds.

    codec_token_count(5)                        ->  3000   (5 * 75 * 8)
    codec_token_count(1, 50, 4)                 ->  200
    codec_token_count(30, 75, 8)                ->  18000

    Считай честно: frame_rate кадров в секунду, на каждый кадр num_codebooks
    индексов RVQ.

    Зачем это в AI: у 5 секунд при 24 кГц — 120 000 отсчётов, ни один
    трансформер такую последовательность не переварит. Кодек сжимает их до
    3000 токенов, и вот с этим уже работает обычная языковая модель.
    Обрати внимание: результат от частоты дискретизации НЕ зависит вовсе —
    именно поэтому кодек и решает задачу.
    """
    # округление вниз: неполный кадр кодек не выдаёт
    return int(seconds * frame_rate) * num_codebooks


def rvq_encode(value, codebooks):
    """Residual vector quantization: каскад индексов, по одному на кодбук.

    Каждый следующий кодбук квантует ОСТАТОК предыдущего, а не сам value.

    books = [[0.0, 0.5, 1.0], [-0.2, -0.1, 0.0, 0.1, 0.2]]
    rvq_encode(0.62, books)  ->  [1, 3]     0.5 + 0.1 = 0.6
    rvq_encode(0.0,  books)  ->  [0, 2]     0.0 + 0.0 = 0.0

    Ловушка накопления ошибки: если первый слой промахнулся крупно (значение
    вне его диапазона), остальные слои этого уже не исправят — качество
    кодека ставит потолок качеству генерации.
    """
    residual = value
    indices = []
    for book in codebooks:
        # ближайший код к ОСТАТКУ; min по индексу, чтобы при равенстве
        # расстояний выбор был предсказуемым, а не зависел от порядка обхода
        best = min(range(len(book)), key=lambda i: abs(book[i] - residual))
        indices.append(best)
        residual -= book[best]
    return indices


def rvq_decode(indices, codebooks):
    """Декодер RVQ: сумма выбранных кодов по всем слоям.

    books = [[0.0, 0.5, 1.0], [-0.2, -0.1, 0.0, 0.1, 0.2]]
    rvq_decode([1, 3], books)  ->  0.6
    rvq_decode([2, 0], books)  ->  0.8

    Обратная операция к rvq_encode: чем больше слоёв, тем ближе сумма к
    исходному значению.
    """
    return sum(book[i] for i, book in zip(indices, codebooks))


def train_bigram(sequences, vocab_size):
    """Матрица переходов «предыдущий токен -> следующий» со сглаживанием.

    train_bigram([[0, 1, 2]], 3)  ->  [[1, 2, 1], [1, 1, 2], [1, 1, 1]]

    В каждой клетке стартовая единица (сглаживание Лапласа) плюс число
    наблюдённых переходов. Ноль недопустим: неувиденный переход должен быть
    маловероятным, а не невозможным — иначе сэмплирование однажды упрётся
    в тупик.

    Ловушка направления: counts[a][b] — переход ИЗ a В b. Матрица
    несимметрична, и это принципиально: звук во времени необратим.
    """
    counts = [[1.0] * vocab_size for _ in range(vocab_size)]
    for seq in sequences:
        # zip(seq, seq[1:]) вместо индексов: короче и без ошибок на границе
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1.0
    return counts


def next_token_probs(counts, prev_token, temperature=1.0):
    """Распределение следующего токена по строке counts, с температурой.

    counts = train_bigram([[0, 1], [0, 1]], 2)
    next_token_probs(counts, 0)                  ->  [0.25, 0.75]
    next_token_probs(counts, 0, temperature=0.1) ->  почти [0.0, 1.0]
    next_token_probs(counts, 0, temperature=50)  ->  почти [0.5, 0.5]

    Температура ниже единицы обостряет распределение, выше — размывает.
    temperature <= 0 бессмысленна: бросай ValueError.

    Ловушка переполнения: наивное p ** (1 / temperature) при маленькой
    температуре и большом словаре превращает ВСЕ вероятности в нули, и
    нормировка падает с ZeroDivisionError. Лечится делением на максимум
    ДО возведения в степень — от нормировки это ничего не меняет.
    """
    if temperature <= 0:
        raise ValueError("temperature должна быть > 0")
    row = counts[prev_token]
    total = sum(row)
    probs = [x / total for x in row]
    if temperature == 1.0:
        return probs
    peak = max(probs)
    power = 1.0 / temperature
    scaled = [(p / peak) ** power for p in probs]
    s = sum(scaled)
    return [x / s for x in scaled]


def generate_tokens(counts, start, length, rng, temperature=1.0):
    """Авторегрессивно досэмплировать последовательность токенов кодека.

    counts = train_bigram([[0, 1, 2, 3]], 4)
    generate_tokens(counts, 0, 4, random.Random(0), temperature=0.01)
    ->  [0, 1, 2, 3]

    Первый элемент — всегда start; всего в ответе ровно length токенов.
    rng — экземпляр random.Random, чтобы результат был воспроизводим.

    Так работает и VALL-E, и MusicGen: prompt из токенов кодека, дальше
    обычное next-token sampling до нужной длины.
    """
    out = [start]
    while len(out) < length:
        probs = next_token_probs(counts, out[-1], temperature)
        # обратное преобразование: идём по кумулятивной сумме одним проходом
        r = rng.random()
        acc = 0.0
        pick = len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                pick = i
                break
        out.append(pick)
    return out


def delay_streams(streams, pad=-1):
    """Delayed parallel из MusicGen: сдвинуть k-й поток кодбука на k шагов.

    delay_streams([[1, 2], [3, 4]], pad=0)  ->  [[1, 2, 0], [0, 3, 4]]

    Все потоки выравниваются по общей длине T + K - 1, где T — исходная
    длина, K — число кодбуков. Дырки затыкаются pad.

    Зачем: плоская склейка K потоков дала бы K * T токенов, а сдвиг —
    всего T + K - 1 колонок. При K=8 и T=1500 это 12000 против 1507.
    """
    k_count = len(streams)
    length = len(streams[0])
    total = length + k_count - 1
    out = []
    for k, stream in enumerate(streams):
        # слева k заглушек, справа столько, чтобы добить до общей длины
        out.append([pad] * k + list(stream) + [pad] * (total - length - k))
    return out


def undelay_streams(delayed):
    """Снять задержку: вернуть исходные потоки из delayed parallel раскладки.

    undelay_streams([[1, 2, 0], [0, 3, 4]])  ->  [[1, 2], [3, 4]]

    Обратная к delay_streams: у k-го потока отрезаем k элементов слева и
    остаток справа. Проверка правильности одна — круговой прогон
    undelay_streams(delay_streams(x)) == x.
    """
    k_count = len(delayed)
    length = len(delayed[0]) - k_count + 1
    return [list(delayed[k][k:k + length]) for k in range(k_count)]
