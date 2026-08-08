"""
Оценка длинного контекста: NIAH, RULER, LongBench — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def build_haystack(filler, needle, depth_ratio, total_tokens):
    """Собрать haystack: filler нужной длины с needle на заданной глубине.

    Токен здесь — слово, разделитель — пробел. Filler повторяется, пока не
    наберётся нужная длина, лишнее отрезается. depth_ratio 0.0 — needle в
    самом начале, 1.0 — в самом конце.

    build_haystack("a b c d", "N", 0.0, 3)   ->  "N a b"
    build_haystack("a b c d", "N", 1.0, 3)   ->  "a b N"

    Итоговая длина в токенах равна total_tokens: needle занимает место
    filler-а, а не добавляется сверху — иначе замер «сколько влезло в
    контекст» врал бы на длину иголки.

    ValueError на depth_ratio вне [0, 1], на неположительный total_tokens,
    на filler без единого токена и если needle длиннее всего бюджета. Мы
    отвергаем такой запрос, а не обрезаем факт, который модель должна найти.
    """
    if not 0.0 <= depth_ratio <= 1.0:
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    filler_tokens = filler.split()
    if not filler_tokens:
        raise ValueError("filler produced no tokens")

    needle_tokens = needle.split()
    if len(needle_tokens) > total_tokens:
        raise ValueError("needle does not fit within total_tokens")
    body_len = total_tokens - len(needle_tokens)
    # удваиваем, а не дописываем по слову: 20 итераций вместо миллиона
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    # min: при depth_ratio ровно 1.0 int(body_len * 1.0) уже равен body_len,
    # но с плавающей точкой на длинных телах бывает +1 — срезаем
    insert_at = min(int(body_len * depth_ratio), body_len)
    return " ".join(filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:])


def insert_needles(text, needles, depths):
    """Вставить несколько needles в текст на заданных глубинах.

    insert_needles("a b c d", ["X", "Y"], [0.0, 0.5])  ->  "X a b Y c d"

    Ловушка: вставлять надо с самой ГЛУБОКОЙ иголки. Если начать с мелкой,
    текст удлинится, и все следующие позиции, посчитанные от исходной
    длины, уедут.

    ValueError, если списки разной длины или глубина вне [0, 1].

    Зачем: MRCR и multi-needle держатся ровно на этом — успех на одной
    иголке ничего не говорит про успех на трёх.
    """
    if len(needles) != len(depths):
        raise ValueError("needles и depths разной длины")
    for d in depths:
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"depth must be in [0, 1], got {d}")

    words = text.split()
    # сначала самая глубокая: позиции мелких считаются от неизменённого
    # начала текста и остаются корректными
    for depth, needle in sorted(zip(depths, needles), reverse=True):
        pos = min(int(len(words) * depth), len(words))
        words = words[:pos] + needle.split() + words[pos:]
    return " ".join(words)


def score_needle(context, question, expected, model):
    """1, если модель вернула ожидаемую строку, иначе 0.

    model — детерминированная заглушка вместо LLM: вызывается как
    model(context, question) и возвращает строку ответа. Сравнение
    без учёта регистра и по вхождению: "The magic word is PINEAPPLE"
    засчитывает expected="pineapple".

    score_needle(ctx, q, "pineapple", lambda c, q: "It is Pineapple.")  ->  1
    score_needle(ctx, q, "pineapple", lambda c, q: "no answer")         ->  0

    Модель приходит параметром, чтобы прогон был воспроизводимым и без
    сети: в проде на её месте реальный клиент.
    """
    answer = model(context, question)
    return 1 if expected.lower() in answer.lower() else 0


def score_multi_needle(context, question, expected_list, model):
    """Доля найденных иголок: сколько из expected_list попало в ответ.

    score_multi_needle(ctx, q, ["a", "b"], lambda c, q: "a")  ->  0.5
    score_multi_needle(ctx, q, [], model)                     ->  0.0

    Частичный балл принципиален: модель, назвавшая два слова из трёх, —
    это не «провал» и не «успех», а 0.67, и по этому числу видно, где
    начинается насыщение внимания.
    """
    if not expected_list:
        return 0.0
    answer = model(context, question).lower()
    hits = sum(1 for e in expected_list if e.lower() in answer)
    return hits / len(expected_list)


def niah_grid(depths, lengths, trial):
    """Прогнать сетку depth x length. Вернуть dict {(length, depth): score}.

    trial(depth, length) -> score — одна ячейка: собрать haystack нужной
    длины с иголкой на нужной глубине, спросить модель, вернуть балл.
    Порядок аргументов у trial именно такой.

    niah_grid([0.5], [100, 200], lambda d, n: 1)
        ->  {(100, 0.5): 1, (200, 0.5): 1}

    Ключ — пара (length, depth), а не одно число, потому что смотреть надо
    обе оси: одна длина при разных глубинах даёт разные результаты, это и
    есть эффект «lost in the middle».
    """
    return {(length, depth): trial(depth, length) for length in lengths for depth in depths}


def pass_rates(grid, axis):
    """Свернуть сетку по одной оси: средний балл по length либо по depth.

    grid — то, что вернул niah_grid. axis: "length" или "depth".

    pass_rates({(100, 0.5): 1, (100, 0.9): 0}, "length")  ->  {100: 0.5}
    pass_rates({(100, 0.5): 1, (200, 0.5): 0}, "depth")   ->  {0.5: 0.5}

    Любой другой axis — ValueError: молча выбрать ось за пользователя
    значит выдать таблицу деградации по глубине за таблицу по длине.
    """
    if axis == "length":
        index = 0
    elif axis == "depth":
        index = 1
    else:
        raise ValueError(f"axis must be 'length' or 'depth', got {axis!r}")

    buckets = {}
    for key, score in grid.items():
        buckets.setdefault(key[index], []).append(score)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def effective_length(rate_by_length, threshold=0.9):
    """Наибольшая длина, до которой качество НИ РАЗУ не падало ниже порога.

    Идём от коротких к длинным и останавливаемся на первой длине, где
    качество просело. Всё, что дальше, уже не считается, даже если там
    случайно снова хорошо.

    effective_length({1000: 1.0, 4000: 1.0, 16000: 0.4})  ->  4000
    effective_length({1000: 0.5})                          ->  0

    0 означает «модель не держит порог даже на самой короткой длине».

    Это и есть честная цифра для спецификации: заявленное окно 1M и
    effective retrieval length 128k — разные числа, и во втором больше
    смысла.
    """
    best = 0
    for length in sorted(rate_by_length):
        if rate_by_length[length] < threshold:
            break
        best = length
    return best


def trace_variables(text):
    """Multi-hop tracing в стиле RULER: разрешить цепочку присваиваний.

    Присваивания раскиданы по filler-у, форма — `NAME = число` либо
    `NAME = OTHER +-* число`. Вернуть dict {имя: значение} в порядке
    появления.

    trace_variables("X1 = 42. blah blah X2 = X1 + 10.")  ->  {"X1": 42, "X2": 52}
    trace_variables("noise without assignments")         ->  {}

    Повторное присваивание перезаписывает значение — модель обязана
    следить за последним, а не за первым.

    ValueError, если правая часть ссылается на ещё не определённую
    переменную: это не «ноль», это сломанная цепочка.

    Зачем: single-needle retrieval насыщается, а вот такая цепочка из трёх
    хопов роняет frontier-модели на 128k до 50-70%.
    """
    values = {}
    # (?:\s*([+\-*])\s*(-?\d+))? — второй операнд необязателен: `X1 = 42`
    # и `X2 = X1 + 10` разбираются одним проходом
    pattern = re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*|-?\d+)(?:\s*([+\-*])\s*(-?\d+))?")
    for name, operand, op, rhs in pattern.findall(text):
        if re.fullmatch(r"-?\d+", operand):
            value = int(operand)
        elif operand in values:
            value = values[operand]
        else:
            raise ValueError(f"unknown variable {operand!r}")
        if op:
            rhs = int(rhs)
            value = value + rhs if op == "+" else value - rhs if op == "-" else value * rhs
        values[name] = value
    return values
