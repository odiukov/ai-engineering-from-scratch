"""
Оптимизация инференса: KV-кэш, батчинг, спекулятивный декодинг — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

BYTES_PER_GB = 1024 ** 3


def softmax(scores):
    """Оценки внимания -> веса внимания: неотрицательные, сумма 1.

    softmax([0.0, 0.0])     ->  [0.5, 0.5]
    softmax([1000.0, 0.0])  ->  [1.0, 0.0]   (без OverflowError)

    Ловушка: exp(1000) переполняется. Вычти максимум перед exp —
    softmax(x) == softmax(x - c) для любого c, ответ не меняется.
    """
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def matvec(vector, matrix):
    """Умножение вектора-строки на матрицу: result[j] = sum_i v[i] * M[i][j].

    matvec([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])  ->  [1.0, 2.0]
    matvec([1.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])  ->  [4.0, 6.0]

    matrix — список СТРОК: len(matrix) == len(vector), а длина строки задаёт
    размерность выхода.

    Именно эта операция считает проекции K и V из скрытого состояния токена.
    Всё, что делает KV-кэш, — избавляет от её повторного вызова на уже
    посчитанных токенах.

    Ловушка размерностей: перепутаешь индексы местами — молча получишь
    произведение на транспонированную матрицу, без всякого исключения,
    если матрица квадратная.
    """
    d_out = len(matrix[0])
    result = [0.0] * d_out
    for v, row in zip(vector, matrix):
        # идём по строкам, а не по столбцам: так матрица читается подряд,
        # и каждая строка попадает в кэш процессора один раз
        for j in range(d_out):
            result[j] += v * row[j]
    return result


def attention(query, keys, values):
    """Внимание одного запроса ко всем ключам: softmax(q·k/sqrt(d)) · v.

    attention([1.0, 0.0], [[1.0, 0.0]], [[5.0, 7.0]])  ->  [5.0, 7.0]
    attention([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 4.0]])
        ->  [1.0, 2.0]   (нулевой запрос смотрит на всех поровну)

    Делитель sqrt(d) — не украшение. Без него скалярные произведения растут
    как d, softmax насыщается, и внимание превращается в argmax по одному
    токену.

    Возвращаем взвешенную сумму values с весами внимания.
    """
    d = len(query)
    scale = math.sqrt(d)
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    weights = softmax(scores)

    d_v = len(values[0])
    out = [0.0] * d_v
    for w, value in zip(weights, values):
        for j in range(d_v):
            out[j] += w * value[j]
    return out


def generate_no_cache(inputs, w_k, w_v):
    """Наивный декодинг: на каждом шаге K и V пересчитываются с нуля.

    inputs — список скрытых состояний токенов; сам токен служит запросом.
    Вернуть {"outputs": [...], "projections": <сколько раз звали matvec>}.

    Причинность: на шаге t запрос видит токены 0..t включительно и ни одного
    следующего.

    generate_no_cache([[1.0, 0.0]], I, I)["projections"]  ->  2
    generate_no_cache([[1.0, 0.0], [0.0, 1.0]], I, I)["projections"]  ->  6

    Считаем проекции честно: на шаге t их 2*(t+1), всего n*(n+1). Это
    квадратичный рост — к тысячному токену первый спроецирован 1000 раз.
    Ровно эту работу и выбрасывает KV-кэш.
    """
    outputs = []
    projections = 0
    for t in range(len(inputs)):
        keys, values = [], []
        # весь префикс заново на каждом шаге — в этом и есть расточительство
        for i in range(t + 1):
            keys.append(matvec(inputs[i], w_k))
            values.append(matvec(inputs[i], w_v))
            projections += 2
        outputs.append(attention(inputs[t], keys, values))
    return {"outputs": outputs, "projections": projections}


def generate_with_cache(inputs, w_k, w_v):
    """То же самое через KV-кэш: каждый токен проецируется РОВНО один раз.

    Вернуть {"outputs": [...], "projections": <сколько раз звали matvec>}.

    generate_with_cache([[1.0, 0.0], [0.0, 1.0]], I, I)["projections"]  ->  4

    Контракт, который проверяют тесты: outputs обязаны совпасть с
    generate_no_cache до последнего разряда. KV-кэш — это чистая экономия,
    он не приближённый метод и не меняет ответ. Разошлось — значит кэш
    протух: где-то дописали не тот тензор или забыли сдвинуть длину.

    Проекций 2n вместо n(n+1): линейно вместо квадратично.
    """
    keys, values = [], []
    outputs = []
    projections = 0
    for x in inputs:
        # дописываем только новый токен, прошлые лежат в кэше
        keys.append(matvec(x, w_k))
        values.append(matvec(x, w_v))
        projections += 2
        outputs.append(attention(x, keys, values))
    return {"outputs": outputs, "projections": projections}


def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2):
    """Сколько байт занимает KV-кэш последовательности.

    Формула: 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes.

    kv_cache_bytes(80, 8, 128, 1)       ->  327680      (Llama 3 70B, 1 токен)
    kv_cache_bytes(80, 8, 128, 131072)  ->  42949672960 (128K контекста, 40 GiB)

    Двойка в начале — это K и V, два тензора на слой.

    num_kv_heads, а не число голов запроса: в GQA (Llama 3 70B) 64 головы
    запроса делят 8 голов ключа-значения. Возьмёшь 64 — насчитаешь кэш
    в восемь раз больше, чем нужно.

    Кэш растёт ЛИНЕЙНО по длине и не зависит от размера модели. Поэтому на
    длинном контексте он и обгоняет по памяти сами веса.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes


def batching_steps(output_lens, batch_size):
    """Сколько шагов декодинга займёт пачка запросов при двух стратегиях.

    Вернуть {"static": ..., "continuous": ..., "speedup": static / continuous}.

    batching_steps([10, 10], 2)            ->  static 10, continuous 10, speedup 1.0
    batching_steps([50, 10, 10, 10], 2)    ->  static 60, continuous 50, speedup 1.2

    Статический батчинг: режем очередь на куски по batch_size, каждый кусок
    стоит max(длин) — короткие запросы простаивают, пока не закончится
    самый длинный.

    Непрерывный батчинг: batch_size слотов, освободившийся слот немедленно
    забирает следующий запрос. Считается это как раскладка задач по слотам:
    каждый следующий запрос уходит в слот, который освободится раньше всех,
    ответ — время самого занятого слота.

    Непрерывный НИКОГДА не хуже статического, а на разнобое длин выигрывает
    в разы. Одинаковые длины — выигрыша нет, и это честный результат,
    а не поломка.
    """
    if not output_lens:
        return {"static": 0, "continuous": 0, "speedup": 1.0}

    static = 0
    for start in range(0, len(output_lens), batch_size):
        static += max(output_lens[start:start + batch_size])

    slots = [0] * batch_size
    for length in output_lens:
        # жадно в самый ранний освобождающийся слот — это и есть
        # «вставить новый запрос, как только доделался старый»
        i = min(range(batch_size), key=lambda k: slots[k])
        slots[i] += length
    continuous = max(slots)

    return {
        "static": static,
        "continuous": continuous,
        "speedup": static / continuous if continuous else 1.0,
    }


def speculative_speedup(num_speculative, acceptance_rate, draft_cost=1.0, target_cost=10.0):
    """Выигрыш спекулятивного декодинга при заданной доле принятия.

    Вернуть {"expected_accepted", "tokens_per_round", "cost_per_round",
    "speedup"}.

    speculative_speedup(5, 0.0)  ->  expected_accepted 0.0, speedup 10/15
    speculative_speedup(5, 1.0)  ->  expected_accepted 5.0, speedup 60/15 == 4.0

    Модель считалки:
      * черновая модель предлагает K токенов, каждый принимается с
        вероятностью p НЕЗАВИСИМО от предыдущих, и первый же отказ
        обрывает цепочку;
      * ожидание принятых = p + p^2 + ... + p^K;
      * за раунд выдаётся expected_accepted + 1 токен: даже при отказе на
        первом же токене целевая модель выдаёт свой, раунд не пустой;
      * стоимость раунда = K * draft_cost + target_cost (одна проверка всех
        K кандидатов разом, это как prefill — параллельно);
      * последовательный декодинг тех же токенов стоил бы
        tokens_per_round * target_cost.

    Спекулятивный декодинг МАТЕМАТИЧЕСКИ ТОЧЕН: распределение выхода
    совпадает с распределением целевой модели. Ускорение берётся не из
    приближения, а из того, что проверка K кандидатов стоит столько же,
    сколько генерация одного токена.

    Ловушка: при низкой доле принятия ускорение меньше единицы — черновик
    оплачен, а толку нет. Отсюда правило «черновая модель должна быть
    по-настоящему дешёвой».
    """
    p = acceptance_rate
    # геометрическая сумма p + p^2 + ... + p^K, аккуратно в лоб:
    # закрытая формула делится на (1 - p) и падает при p == 1
    expected_accepted = 0.0
    term = 1.0
    for _ in range(num_speculative):
        term *= p
        expected_accepted += term

    tokens_per_round = expected_accepted + 1.0
    cost_per_round = num_speculative * draft_cost + target_cost
    sequential_cost = tokens_per_round * target_cost
    return {
        "expected_accepted": expected_accepted,
        "tokens_per_round": tokens_per_round,
        "cost_per_round": cost_per_round,
        "speedup": sequential_cost / cost_per_round if cost_per_round else 1.0,
    }
