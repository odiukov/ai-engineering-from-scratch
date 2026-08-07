"""
Машинный перевод — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
from collections import Counter


def ngrams(tokens, n):
    """Все n-граммы последовательности токенов, как список кортежей.

    ngrams(["the", "cat", "sat"], 2)  ->  [("the", "cat"), ("cat", "sat")]
    ngrams(["the"], 2)                ->  []   окно шире фразы

    n < 1 — это ValueError: n-грамм нулевой длины не бывает.

    Кортежи, а не списки: n-граммы дальше идут ключами Counter, а список
    нехешируем.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    # окно скользит, пока помещается: ровно len - n + 1 позиций
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def clipped_ngram_counts(hypothesis, references, n):
    """Обрезанные счётчики n-грамм: (сколько совпало, сколько всего в гипотезе).

    Совпадение каждой n-граммы обрезается сверху её максимальным числом
    вхождений среди референсов. Без обрезки перевод "the the the the"
    получил бы точность 1.0 против референса "the cat sat".

    clipped_ngram_counts(["the"]*4, [["the", "cat"]], 1)  ->  (1, 4)

    Обе половины возвращаются отдельно, потому что corpus BLEU складывает
    именно счётчики по всему корпусу, а не средние по предложениям.
    """
    hyp_counts = Counter(ngrams(hypothesis, n))
    # для каждой n-граммы берём МАКСИМУМ по референсам: любой из них
    # считается допустимым переводом
    best = Counter()
    for reference in references:
        ref_counts = Counter(ngrams(reference, n))
        for gram, count in ref_counts.items():
            if count > best[gram]:
                best[gram] = count
    matched = sum(min(count, best[gram]) for gram, count in hyp_counts.items())
    return matched, sum(hyp_counts.values())


def brevity_penalty(hyp_len, ref_len):
    """Штраф за короткий перевод: BP = 1 если hyp >= ref, иначе exp(1 - ref/hyp).

    brevity_penalty(10, 10)  ->  1.0
    brevity_penalty(5, 10)   ->  exp(-1) = 0.3679
    brevity_penalty(0, 10)   ->  0.0

    Зачем он: точность n-грамм растёт, когда перевод укорачивают до самых
    надёжных слов. Выдать одно слово "the" и получить точность 1.0 —
    ровно та дыра, которую BP закрывает.

    Длинный перевод BP не штрафует: за него уже платит точность.
    """
    if hyp_len == 0:
        return 0.0
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def sentence_bleu(hypothesis, references, max_n=4):
    """BLEU одного предложения по шкале [0, 100].

    hypothesis — список токенов, references — список списков токенов.

    sentence_bleu(t, [t])  ->  100.0   идентичный перевод
    Перевод без единой общей 4-граммы  ->  0.0

    Считается так: геометрическое среднее обрезанных точностей для
    n = 1..max_n, умноженное на brevity_penalty. Референсная длина —
    ближайшая к длине гипотезы (при ничьей берётся более короткая).

    Ловушка: одна нулевая точность обнуляет геометрическое среднее, значит
    фраза короче max_n токенов даёт ровно 0.0. Это не баг, это BLEU: у него
    нет сглаживания, поэтому на коротких строках он бесполезен.
    """
    if not hypothesis:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        matched, total = clipped_ngram_counts(hypothesis, references, n)
        if matched == 0 or total == 0:
            # ноль в геометрическом среднем — это ноль, дальше считать нечего
            return 0.0
        log_sum += math.log(matched / total)
    # ближайшая длина референса, при ничьей — более короткая
    closest = min(references, key=lambda r: (abs(len(r) - len(hypothesis)), len(r)))
    bp = brevity_penalty(len(hypothesis), len(closest))
    return 100.0 * bp * math.exp(log_sum / max_n)


def corpus_bleu(hypotheses, references_list, max_n=4):
    """BLEU всего корпуса: счётчики складываются, и только потом делятся.

    hypotheses — список списков токенов, references_list — по списку
    референсов на каждую гипотезу.

    Ловушка, ради которой всё и написано: corpus BLEU НЕ равен среднему
    sentence BLEU. Одно предложение с нулевой 4-граммной точностью даёт
    sentence BLEU 0 и утаскивает среднее вниз, а в корпусном счёте его
    совпавшие униграммы всё ещё считаются.

    Именно поэтому sacrebleu.corpus_bleu — то, что печатают в статьях, и
    сравнивать со средним по предложениям нельзя.
    """
    if len(hypotheses) != len(references_list):
        raise ValueError("hypotheses and references_list must be the same length")
    matched_by_n = [0] * (max_n + 1)
    total_by_n = [0] * (max_n + 1)
    hyp_len = 0
    ref_len = 0
    for hypothesis, references in zip(hypotheses, references_list):
        for n in range(1, max_n + 1):
            matched, total = clipped_ngram_counts(hypothesis, references, n)
            matched_by_n[n] += matched
            total_by_n[n] += total
        hyp_len += len(hypothesis)
        closest = min(references, key=lambda r: (abs(len(r) - len(hypothesis)), len(r)))
        ref_len += len(closest)
    log_sum = 0.0
    for n in range(1, max_n + 1):
        if matched_by_n[n] == 0 or total_by_n[n] == 0:
            return 0.0
        log_sum += math.log(matched_by_n[n] / total_by_n[n])
    return 100.0 * brevity_penalty(hyp_len, ref_len) * math.exp(log_sum / max_n)


def chrf(hypothesis, reference, max_n=6, beta=2.0):
    """chrF: F-мера по символьным n-граммам, шкала [0, 100].

    На вход строки, а не токены. Пробелы выбрасываются — chrF смотрит на
    символы, поэтому токенизация ему не нужна.

    chrf("les chats", "les chats")  ->  100.0
    chrf("xyz", "abc")              ->  0.0

    beta задаёт вес полноты: beta=2 (стандарт) ценит recall вчетверо выше
    precision, потому что пропущенный кусок перевода хуже лишнего.

    Порядки, которых нет ни в гипотезе, ни в референсе, из усреднения
    выкидываются — иначе короткая идентичная пара не набрала бы 100.

    Зачем это поверх BLEU: для морфологически богатых языков BLEU
    недосчитывает («courent» и «court» для него разные слова), а chrF видит
    общий корень.
    """
    hyp = "".join(hypothesis.split())
    ref = "".join(reference.split())
    precision_sum = 0.0
    recall_sum = 0.0
    orders = 0
    for n in range(1, max_n + 1):
        hyp_counts = Counter(ngrams(hyp, n))
        ref_counts = Counter(ngrams(ref, n))
        hyp_total = sum(hyp_counts.values())
        ref_total = sum(ref_counts.values())
        if hyp_total == 0 and ref_total == 0:
            continue  # такой длины нет ни у кого — порядок не считаем
        orders += 1
        matched = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
        precision_sum += matched / hyp_total if hyp_total else 0.0
        recall_sum += matched / ref_total if ref_total else 0.0
    if orders == 0:
        return 0.0
    precision = precision_sum / orders
    recall = recall_sum / orders
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * precision * recall / (b2 * precision + recall)


def flag_length_explosion(source_tokens, hypothesis_tokens, max_ratio=2.5):
    """True, если перевод подозрительно длиннее источника.

    flag_length_explosion(["a"] * 10, ["b"] * 12)  ->  False
    flag_length_explosion(["a"] * 2, ["b"] * 20)   ->  True

    Ровно max_ratio — ещё не тревога, тревога начинается строго выше.

    Из урока: на очень коротком входе length penalty срывается, и модель
    досочиняет текст. Пустой источник с непустым выходом — это уже чистая
    галлюцинация, а не длина.
    """
    if not source_tokens:
        return bool(hypothesis_tokens)
    return len(hypothesis_tokens) > max_ratio * len(source_tokens)


def glossary_violations(source, hypothesis, glossary):
    """Термины глоссария, которые в источнике есть, а в переводе не переведены.

    glossary — словарь {термин источника: обязательный перевод}. Сравнение
    без учёта регистра. Ответ отсортирован, чтобы прогон был воспроизводим.

    glossary_violations("Sign up now", "Créez un compte",
                        {"sign up": "s'inscrire"})   ->  ["sign up"]

    Термины, которых в источнике нет, не проверяются: глоссарий не требует
    вставлять слова, которых не было.

    Из урока: terminology drift — «sign up» превращается то в «s'inscrire»,
    то в «créer un compte». Для интерфейсных строк единообразие важнее
    красоты перевода.
    """
    source_lower = source.lower()
    hypothesis_lower = hypothesis.lower()
    missing = [
        term
        for term, required in glossary.items()
        if term.lower() in source_lower and required.lower() not in hypothesis_lower
    ]
    return sorted(missing)
