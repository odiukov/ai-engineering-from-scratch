"""
Sequence-to-sequence модели — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(logits):
    """Логиты -> распределение вероятностей: всё положительное, сумма равна 1.

    softmax([0.0, 0.0])   ->  [0.5, 0.5]
    softmax([1.0, 0.0])   ->  примерно [0.731, 0.269]

    Ловушка: math.exp(1000) кидает OverflowError. Вычти максимум из всех
    логитов перед exp — ответ математически тот же, а переполнения нет.

    Пустой список логитов — это ошибка (ValueError): распределения над
    пустым словарём не существует.

    В seq2seq это последний слой декодера: он превращает выход nn.Linear
    в вероятности следующего токена.
    """
    if not logits:
        raise ValueError("softmax of empty logits")
    # сдвиг на максимум: exp(x - m) не переполняется, а деление на сумму
    # сокращает общий множитель exp(-m), так что ответ не меняется
    shift = max(logits)
    exps = [math.exp(v - shift) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rnn_step(x, h_prev, W_x, W_h, b):
    """Один шаг простой RNN: h = tanh(W_x @ x + W_h @ h_prev + b).

    x — вектор входа (эмбеддинг токена), h_prev — предыдущее скрытое
    состояние, W_x — матрица hidden x input, W_h — матрица hidden x hidden,
    b — вектор смещений длины hidden.

    rnn_step([1.0], [0.0], [[2.0]], [[0.0]], [0.0])  ->  [tanh(2.0)]

    Ловушка: hidden берётся из len(b), и все три матрицы обязаны про него
    договориться. Если len(W_x) != len(W_h) != len(b) — это ValueError,
    а не молчаливый обрез по короткой матрице.

    tanh зажимает состояние в (-1, 1): без него активации разъезжаются
    после десятка шагов.
    """
    hidden = len(b)
    if len(W_x) != hidden or len(W_h) != hidden:
        raise ValueError("W_x, W_h and b disagree on hidden size")
    out = []
    for i in range(hidden):
        # два скалярных произведения на строку: вклад входа и вклад памяти
        acc = b[i]
        acc += sum(w * v for w, v in zip(W_x[i], x))
        acc += sum(w * v for w, v in zip(W_h[i], h_prev))
        out.append(math.tanh(acc))
    return out


def encode(token_ids, embeddings, W_x, W_h, b):
    """Энкодер: прогнать источник через RNN. Вернуть (states, context).

    embeddings[token_id] — вектор эмбеддинга, states — по одному скрытому
    состоянию на входной токен, context — последнее состояние.

    encode([], emb, W_x, W_h, b)  ->  ([], [0.0, ...])  нулевой контекст
    len(encode(src, ...)[0]) == len(src)

    Начальное состояние — нули длины len(b).

    Здесь виден bottleneck из урока: сколько бы токенов ни было на входе,
    context всегда одной и той же длины. Всё, что декодер узнает об
    источнике, обязано влезть в этот один вектор.
    """
    h = [0.0] * len(b)
    states = []
    for token_id in token_ids:
        h = rnn_step(embeddings[token_id], h, W_x, W_h, b)
        states.append(h)
    # context — это states[-1]; отдельная переменная нужна только чтобы
    # пустой вход честно вернул нулевое состояние, а не упал по индексу
    return states, h


def decode_step(token_id, hidden, embeddings, W_x, W_h, b, W_out, b_out):
    """Один шаг декодера: вернуть (logits, new_hidden).

    Декодер вызывают по одному токену за раз: на входе предыдущий токен и
    текущее состояние, на выходе логиты над целевым словарём и обновлённое
    состояние. W_out — матрица vocab x hidden, b_out — вектор длины vocab.

    len(decode_step(...)[0]) == len(b_out)   один логит на слово словаря

    Ловушка: состояние надо ВЕРНУТЬ и подать в следующий вызов. Если
    забыть — декодер на каждом шаге начинает с чистого листа и печатает
    один и тот же токен.
    """
    new_hidden = rnn_step(embeddings[token_id], hidden, W_x, W_h, b)
    # проекция скрытого состояния на словарь: строка W_out на слово
    logits = [
        b_out[v] + sum(w * h for w, h in zip(W_out[v], new_hidden))
        for v in range(len(b_out))
    ]
    return logits, new_hidden


def teacher_forcing_input(true_prev, predicted_prev, ratio, rng):
    """Что подать декодеру на следующем шаге обучения: правду или свой прогноз.

    ratio — вероятность взять ground-truth токен. rng — экземпляр
    random.Random, чтобы обучение было воспроизводимым.

    teacher_forcing_input(7, 3, 1.0, rng)  ->  7   полный teacher forcing
    teacher_forcing_input(7, 3, 0.0, rng)  ->  3   модель ест свои ошибки

    Ловушка: глобальный random.random() ломает воспроизводимость прогона.
    Генератор передают аргументом.

    Полный teacher forcing стабилизирует обучение, но создаёт exposure
    bias: на инференсе правды нет, а модель никогда не тренировалась
    выбираться из собственных ошибок. Отсюда annealing ratio к ~0.5.
    """
    # random() возвращает [0, 1), поэтому ratio=0.0 никогда не даёт правду,
    # а ratio=1.0 даёт её всегда — краевые случаи выходят сами собой
    return true_prev if rng.random() < ratio else predicted_prev


def sequence_cross_entropy(step_logits, target_ids, pad_id=0):
    """Средняя cross-entropy по шагам декодера, паддинг не считается.

    step_logits — список логитов, по одному вектору на шаг. target_ids —
    правильные id той же длины. Шаги, где target равен pad_id, выкидываются
    целиком.

    sequence_cross_entropy([[0.0, 0.0]], [1])  ->  log(2) = 0.693
    Если все шаги — паддинг, ответ 0.0.

    Ловушка: -log(0) это бесконечность. Вероятность прижимают снизу
    крошечным epsilon, иначе один уверенно неправильный шаг превращает
    весь loss в inf.

    Это ровно nn.CrossEntropyLoss(ignore_index=0) из урока, собранный руками.
    """
    if len(step_logits) != len(target_ids):
        raise ValueError("step_logits and target_ids must have the same length")
    total = 0.0
    counted = 0
    for logits, target in zip(step_logits, target_ids):
        if target == pad_id:
            continue
        p = softmax(logits)[target]
        # прижимаем снизу: loss должен быть большим числом, а не inf
        total += -math.log(max(p, 1e-12))
        counted += 1
    return total / counted if counted else 0.0


def greedy_decode(step, bos_id, eos_id, initial_hidden, max_len=20):
    """Жадное декодирование: на каждом шаге брать argmax логитов.

    step(token_id, hidden) -> (logits, new_hidden) — уже обученный декодер,
    его передают функцией. initial_hidden — контекст от энкодера.

    Ответ — список id БЕЗ bos и БЕЗ eos: eos останавливает генерацию, но в
    ответ не попадает.

    При ничьей логитов берём меньший индекс — иначе прогон невоспроизводим.

    Ловушка: без max_len модель, не выучившая eos, крутится вечно.
    Жадность близорука: выбрав токен, его уже не отменить, и один
    самоуверенный шаг уводит всю фразу.
    """
    token = bos_id
    hidden = initial_hidden
    out = []
    for _ in range(max_len):
        logits, hidden = step(token, hidden)
        # argmax руками, чтобы ничья доставалась меньшему индексу:
        # max(range(...), key=...) ведёт себя так же, но здесь это видно
        best = 0
        for i, value in enumerate(logits):
            if value > logits[best]:
                best = i
        if best == eos_id:
            break
        out.append(best)
        token = best
    return out


def beam_search(step, bos_id, eos_id, initial_hidden, beam_width=3, max_len=20):
    """Beam search: держать живыми beam_width лучших частичных гипотез.

    Счёт гипотезы — сумма log-вероятностей её токенов (softmax от логитов).
    Ответ — токены лучшей гипотезы, без bos и без eos.

    beam_search(step, bos, eos, h, beam_width=1) обязан совпасть с
    greedy_decode: ширина 1 — это и есть жадность.

    Ловушка: сумма логарифмов всегда отрицательна и падает с каждым
    токеном, поэтому голый счёт любит короткие ответы. В боевых декодерах
    его делят на длину (length penalty); здесь намеренно оставлено как есть,
    чтобы эффект было видно.

    Beam выигрывает там, где локально лучший первый токен ведёт в тупик:
    жадность уже не отыграет назад, а beam держит запасной вариант.
    """
    # гипотеза: (счёт, выданные токены, последний токен, состояние)
    live = [(0.0, [], bos_id, initial_hidden)]
    done = []
    for _ in range(max_len):
        if not live:
            break
        candidates = []
        for score, tokens, last, hidden in live:
            logits, new_hidden = step(last, hidden)
            probs = softmax(logits)
            for token_id, p in enumerate(probs):
                logp = math.log(max(p, 1e-300))
                candidates.append(
                    (score + logp, tokens, token_id, new_hidden)
                )
        # сортировка стабильна, поэтому при равных счетах впереди останется
        # кандидат с меньшим индексом токена — та же ничья, что в greedy
        candidates.sort(key=lambda c: -c[0])
        live = []
        for score, tokens, token_id, new_hidden in candidates[:beam_width]:
            if token_id == eos_id:
                # eos занимает слот луча — именно поэтому ширина 1
                # воспроизводит greedy шаг в шаг
                done.append((score, tokens))
            else:
                live.append((score, tokens + [token_id], token_id, new_hidden))
    # добравшиеся до max_len гипотезы тоже участвуют в финальном отборе
    for score, tokens, _last, _hidden in live:
        done.append((score, tokens))
    if not done:
        return []
    return max(done, key=lambda c: c[0])[1]
