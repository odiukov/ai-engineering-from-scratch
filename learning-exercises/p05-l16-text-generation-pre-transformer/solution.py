"""
Генерация текста до трансформеров: n-граммные языковые модели — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
from collections import Counter, defaultdict


def train_ngram(sentences, n=3):
    """Счётчики n-грамм и контекстов. Возвращает (ngrams, contexts).

    Каждое предложение — список токенов. Слева дописывается n-1 штук "<s>",
    справа один "</s>": без этого модель не знает, чем предложение
    начинается и чем заканчивается.

    Ключ ngrams — кортеж из n токенов, ключ contexts — кортеж из n-1.

    train_ngram([["a", "b"]], n=2)
        ->  ({("<s>", "a"): 1, ("a", "b"): 1, ("b", "</s>"): 1},
             {("<s>",): 1, ("a",): 1, ("b",): 1})
    train_ngram([["a"], ["a"]], n=2)[0][("<s>", "a")]  ->  2

    Ловушка: contexts — не «сколько раз этот кортеж встретился в тексте», а
    «сколько раз он был контекстом», то есть за ним что-то следовало.
    Финальный "</s>" контекстом не бывает никогда.
    """
    ngrams = Counter()
    contexts = Counter()
    for sentence in sentences:
        padded = ["<s>"] * (n - 1) + list(sentence) + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i : i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    """Оценка максимального правдоподобия: count(context, word) / count(context).

    context — последовательность токенов (список или кортеж, всё равно).

    ng, ctx = train_ngram([["a", "b"], ["a", "c"]], n=2)
    raw_probability(ng, ctx, ["a"], "b")     ->  0.5
    raw_probability(ng, ctx, ["a"], "zzz")   ->  0.0
    raw_probability(ng, ctx, ["zzz"], "b")   ->  0.0

    Ровно та формула, что стоит за каждым n-граммным движком до 2015 года.
    И ровно та причина, по которой ни один из них не работал без
    сглаживания: невиданная n-грамма получает 0, весь log-likelihood
    предложения обращается в минус бесконечность.
    """
    ctx = tuple(context)
    total = contexts.get(ctx, 0)
    # неизвестный контекст: делить не на что, честный ответ — ноль
    if total == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / total


def laplace_probability(ngrams, contexts, vocab_size, context, word):
    """Add-one сглаживание: (count + 1) / (count(context) + vocab_size).

    ng, ctx = train_ngram([["a", "b"], ["a", "c"]], n=2)
    laplace_probability(ng, ctx, 4, ["a"], "b")    ->  (1 + 1) / (2 + 4) = ~0.333
    laplace_probability(ng, ctx, 4, ["a"], "zzz")  ->  (0 + 1) / (2 + 4) = ~0.167

    Единица в числителе выдаётся каждому слову словаря, поэтому знаменатель
    прибавляет vocab_size, а не 1 — иначе сумма по словарю перестанет быть
    единицей.

    Простейшее из сглаживаний и худшее из работающих: на редких событиях оно
    отдаёт невиданному слишком много массы, отбирая её у виденного один раз.
    """
    ctx = tuple(context)
    return (ngrams.get(ctx + (word,), 0) + 1) / (contexts.get(ctx, 0) + vocab_size)


def continuation_probability(sentences):
    """P_cont(w): доля УНИКАЛЬНЫХ биграмм, которые заканчиваются на w.

    Не «как часто встречается w», а «в скольких разных контекстах». В этом
    вся идея Кнесера-Нея. Возвращает dict слово -> вероятность, сумма 1.

    continuation_probability([["a", "b"], ["c", "b"]])
        ->  у "b" два разных предшественника, у остальных по одному

    Классический пример: "Francisco" встречается часто, но почти всегда
    после "San". Обычная униграммная частота решит, что слово популярное, и
    охотно поставит его в новый контекст. Continuation probability видит
    один-единственный предшественник и придавливает слово.

    Предложение дополняется "<s>" слева и "</s>" справа, как в train_ngram.
    Само "<s>" ни за кем не следует и в ответ не попадает.
    """
    unigram_contexts = defaultdict(set)
    for sentence in sentences:
        padded = ["<s>"] + list(sentence) + ["</s>"]
        for i in range(1, len(padded)):
            unigram_contexts[padded[i]].add(padded[i - 1])

    total = sum(len(prevs) for prevs in unigram_contexts.values())
    if total == 0:
        return {}
    return {w: len(prevs) / total for w, prevs in unigram_contexts.items()}


def kneser_ney_bigram(sentences, discount=0.75):
    """Интерполированный Кнесер-Ней для биграмм. Возвращает prob(context, w).

    Три слагаемых механики:
      1. основной член — max(count(prev, w) - D, 0) / count(prev);
      2. lambda(prev) = D * (сколько разных слов следовало за prev) / count(prev)
         — ровно та масса, которую отняла скидка D;
      3. эта масса раздаётся по continuation_probability.

    model = kneser_ney_bigram([["a", "b"], ["a", "c"]])
    model(("a",), "b")     ->  положительное число меньше 0.5 (скидка съела часть)
    model(("a",), "zzz")   ->  1e-9, слова нет в корпусе, но и не ноль
    model(("zzz",), "b")   ->  P_cont("b"): контекст неизвестен, откатились вниз

    Для известного prev сумма prob(prev, w) по всем словам корпуса равна 1:
    сколько скидка забрала, столько lambda и вернула.

    Ловушка: скидка вычитается ОДИН раз с каждой биграммы, не с каждого
    вхождения. max(count - D, 0), а не count - D * count.
    """
    bigrams = Counter()
    context_totals = Counter()
    unique_follow = defaultdict(set)

    for sentence in sentences:
        padded = ["<s>"] + list(sentence) + ["</s>"]
        for i in range(1, len(padded)):
            prev, w = padded[i - 1], padded[i]
            bigrams[(prev, w)] += 1
            context_totals[prev] += 1
            unique_follow[prev].add(w)

    p_cont = continuation_probability(sentences)

    def prob(context, w):
        prev = context[-1]
        denom = context_totals.get(prev, 0)
        # контекст не виден ни разу: старший член посчитать не из чего,
        # откатываемся целиком на младшую модель
        if denom == 0:
            return p_cont.get(w, 1e-9)
        discounted = max(bigrams.get((prev, w), 0) - discount, 0) / denom
        lam = discount * len(unique_follow[prev]) / denom
        return discounted + lam * p_cont.get(w, 1e-9)

    return prob


def bits_per_token(prob_fn, sentences, n=2):
    """Средняя кросс-энтропия в битах на токен: сколько модель не угадала.

    Идёт по тем же n-граммам, что и обучение: предложение дополняется n-1
    токенами "<s>" и одним "</s>", затем считается среднее
    -log2 p(w | context). context всегда кортеж длины n-1.

    Равномерная модель на 4 слова  ->  2.0 бита на токен
    Модель, знающая ответ точно    ->  0.0 бита

    Это счёт в игре Шеннона в угадайку: 4.75 бита на букву при слепом
    переборе 27 символов, 0.6-1.3 у человека со 100 буквами контекста.

    n обязан совпадать с порядком обученной модели. Иначе триграммные
    счётчики получают однословный контекст, не находят ни одной n-граммы и
    оценка выглядит катастрофической без всякой причины.

    Ловушка: p может оказаться нулём (несглаженная модель), а log2(0) — это
    ValueError. Подставляй пол вроде max(p, 1e-12).
    """
    if n < 1:
        raise ValueError("n must be positive")
    total_bits = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] * (n - 1) + list(sentence) + ["</s>"]
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - n + 1 : i])
            p = prob_fn(context, padded[i])
            total_bits -= math.log2(max(p, 1e-12))
            total_tokens += 1
    if total_tokens == 0:
        return 0.0
    return total_bits / total_tokens


def perplexity(prob_fn, sentences, n=2):
    """exp от средней отрицательной логарифмической правдоподобности. Меньше — лучше.

    Равномерная модель на 4 слова  ->  4.0
    Идеальная модель               ->  1.0

    perplexity — это «сколько равновероятных вариантов у модели ещё
    осталось». 140 у хорошего 4-граммного KN на Brown, 15-30 у
    трансформера: те самые десять раз, ради которых поле сменило технологию.

    Проверь себя: perplexity обязана совпасть с 2 ** bits_per_token. Одна и
    та же величина, натуральные логарифмы против двоичных.
    """
    return 2.0 ** bits_per_token(prob_fn, sentences, n)


def generate(prob_fn, vocab, prefix, rng, max_len=30, n=2):
    """Сэмплирование продолжения пропорционально вероятностям. Возвращает токены.

    На каждом шаге берётся кортеж последних n-1 токенов как контекст,
    считаются веса prob_fn(context, w) по всему vocab и вытягивается слово
    пропорционально весу. Если prefix короче контекста, слева добавляются
    "<s>". Остановка — по "</s>" или после max_len шагов.

    Ответ включает prefix целиком.

    only_b = lambda context, w: 1.0 if w == "b" else 0.0
    generate(only_b, ["a", "b", "</s>"], ["<s>"], random.Random(0), max_len=3)
        ->  ["<s>", "b", "b", "b"]
    only_eos = lambda context, w: 1.0 if w == "</s>" else 0.0
    generate(only_eos, ["a", "</s>"], ["<s>"], random.Random(0), max_len=99)
        ->  ["<s>", "</s>"]

    rng — экземпляр random.Random. Глобальный random здесь запрещён: два
    запуска с одним seed обязаны дать один и тот же текст.

    Ловушка: сумма весов по vocab не обязана равняться единице — vocab может
    быть урезан. Умножай rng.random() на реальную сумму, а не на 1.0.
    """
    if n < 1:
        raise ValueError("n must be positive")
    tokens = list(prefix)
    context_width = n - 1
    for _ in range(max_len):
        padded = ["<s>"] * context_width + tokens
        context = tuple(padded[-context_width:]) if context_width else ()
        weights = [prob_fn(context, w) for w in vocab]
        total = sum(weights)
        # один rng.random() на шаг: так порядок вызовов не зависит от того,
        # где остановился перебор, и seed действительно воспроизводим
        threshold = rng.random() * total
        acc = 0.0
        chosen = vocab[-1]
        for w, weight in zip(vocab, weights):
            acc += weight
            if threshold <= acc:
                chosen = w
                break
        tokens.append(chosen)
        if chosen == "</s>":
            break
    return tokens
