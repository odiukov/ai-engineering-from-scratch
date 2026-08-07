"""
Обработка текста: токенизация, стемминг, лемматизация — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def tokenize(text):
    """Разбить строку на токены: слова, числа и знаки препинания по отдельности.

    tokenize("The cats weren't running at 3pm.")
        ->  ['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
    tokenize("")  ->  []

    Три шаблона по убыванию приоритета: слово с необязательным апострофом
    внутри ("don't", "it's"), затем чистое число, затем любой одиночный
    символ, который не пробел и не буква-цифра (знак препинания).

    Ловушка: порядок альтернатив в регулярке — это и есть правило
    разрешения конфликтов. Поставь одиночный символ первым — и слова
    рассыплются на буквы.

    Это первый шаг любого классического NLP-пайплайна: модель читает
    целые числа, а не строки.
    """
    # findall с одной регуляркой быстрее, чем ручной проход по символам:
    # весь разбор уходит в C-код движка re. Сложность линейная по длине.
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)


def tokenize_with_urls(text):
    """То же, что tokenize, но URL остаётся одним токеном.

    tokenize_with_urls("Visit https://example.com today.")
        ->  ['Visit', 'https://example.com', 'today', '.']
    tokenize_with_urls("no links here")  ->  ['no', 'links', 'here']

    Шаблон для URL должен стоять ПЕРЕД остальными, иначе слово "https"
    заберёт себе начало ссылки.

    Вторая ловушка: точка в конце предложения не должна прилипать к
    ссылке. "https://example.com." — это URL плюс отдельная точка.

    Так в реальных пайплайнах добавляют emoji, хэштеги, e-mail: новый
    шаблон дописывают в начало альтернативы, а не в конец.
    """
    # [^\s]* жадно съедает всё, а финальный [^\s.,;:!?] заставляет движок
    # откатиться на последний "содержательный" символ. Так завершающая
    # точка предложения остаётся снаружи ссылки.
    pattern = (
        r"https?://[^\s]*[^\s.,;:!?]"
        r"|[A-Za-z]+(?:'[A-Za-z]+)?"
        r"|[0-9]+"
        r"|[^\sA-Za-z0-9]"
    )
    return re.findall(pattern, text)


def stem_step_1a(word):
    """Шаг 1a стеммера Портера: снять окончание множественного числа.

    stem_step_1a("caresses")  ->  'caress'
    stem_step_1a("ponies")    ->  'poni'
    stem_step_1a("caress")    ->  'caress'
    stem_step_1a("cats")      ->  'cat'

    Правила читаются сверху вниз, побеждает первое подошедшее:
      sses -> ss,  ies -> i,  ss -> ss (не трогаем),  s -> (пусто).

    Ловушка: правило "s -> пусто" сработает и на односимвольном "s",
    оставив пустую строку. Порядок правил важнее любого отдельного правила.

    "ponies" -> "poni", а не "pony" — это не баг, а честная цена
    правил без словаря.
    """
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def stem_step_1b(word):
    """Шаг 1b стеммера Портера: снять -ed / -ing, если основа не пустая.

    stem_step_1b("watched")  ->  'watch'
    stem_step_1b("hopping")  ->  'hop'     (двойная согласная схлопывается)
    stem_step_1b("falling")  ->  'fall'    (ll, ss, zz — исключение)
    stem_step_1b("bled")     ->  'bled'    (в основе "bl" нет гласной)

    Условие: то, что остаётся ПОСЛЕ снятия суффикса, обязано содержать
    гласную. Иначе от слова остаётся огрызок: "sing" -> "s".

    Гласными здесь считаются только a, e, i, o, u — без хитростей с "y".

    Ловушка: после снятия -ing удвоенная согласная на конце ("hopp") —
    артефакт английской орфографии, её надо убрать. Но "ll", "ss", "zz"
    трогать нельзя, иначе "fall" превратится в "fal".
    """
    for suffix in ("ing", "ed"):
        if not word.endswith(suffix):
            continue
        base = word[: -len(suffix)]
        if not any(ch in "aeiou" for ch in base):
            # основа без гласной — суффикс не снимаем, слово оставляем как есть
            return word
        double_tail = (
            len(base) >= 2
            and base[-1] == base[-2]
            and base[-1] not in "aeiou"
            and base[-1] not in "lsz"
        )
        return base[:-1] if double_tail else base
    return word


def stem(word):
    """Полный (учебный) стеммер: шаг 1a, затем шаг 1b.

    stem("cats")     ->  'cat'
    stem("running")  ->  'run'
    stem("hopped")   ->  'hop'

    Порядок фиксирован: сначала множественное число, потом времена.
    Смысл стемминга — склеить разные формы одного корня в один ключ,
    даже если этот ключ не является настоящим словом.
    """
    return stem_step_1b(stem_step_1a(word))


def lemmatize(word, pos, table):
    """Лемматизация по словарю с грамматическим откатом.

    table — словарь {(слово_в_нижнем_регистре, pos): лемма}.

    lemmatize("running", "VERB", {("running", "VERB"): "run"})  ->  'run'
    lemmatize("cats", "NOUN", {})     ->  'cat'      (откат: NOUN + s)
    lemmatize("singing", "VERB", {})  ->  'sing'     (откат: VERB + ing)
    lemmatize("bring", "VERB", {})    ->  'bring'    (откат не сработал)
    lemmatize("watched", "VERB", {})  ->  'watched'  (откат про -ed нет)

    Порядок: сначала точный поиск в таблице, потом откаты, потом просто
    нижний регистр. Регистр слова на поиск влиять не должен.

    Откаты срабатывают, только если в остатке есть гласная (a, e, i, o, u).
    Без этой проверки "bring" превратится в "br", а "sing" в "s".

    Ловушка: pos — не украшение. Одно и то же слово с разным pos даёт
    разную лемму, и именно поэтому лемматизация без POS-тегера ломается.

    Случай "watched" — главный урок: настоящая лемматизация требует
    морфологического словаря (WordNet, spaCy), а не пяти правил.
    """
    key = (word.lower(), pos)
    if key in table:
        return table[key]
    low = word.lower()
    # откаты покрывают только самые частые регулярные формы; всё
    # остальное (неправильные глаголы, -ed, сравнительные прилагательные)
    # честно возвращается как есть
    has_vowel = lambda s: any(ch in "aeiou" for ch in s)
    if pos == "VERB" and low.endswith("ing") and has_vowel(low[:-3]):
        return low[:-3]
    if pos == "NOUN" and low.endswith("s") and has_vowel(low[:-1]):
        return low[:-1]
    return low


def preprocess(text, lemma_table, pos_tagger=None):
    """Весь пайплайн разом: токены, стеммы, леммы.

    Вернуть словарь {"tokens": [...], "stems": [...], "lemmas": [...]}.
    Все три списка одной длины — это параллельные представления одного текста.

    preprocess("The cats", {})
        ->  {'tokens': ['The', 'cats'],
             'stems':  ['the', 'cat'],
             'lemmas': ['the', 'cat']}

    pos_tagger — функция, которая принимает список токенов и возвращает
    список пар (токен, pos). Если её нет, считаем всё существительными и
    честно признаём, что глаголы при этом лемматизируются неправильно.

    Стеммы считаются от токенов в нижнем регистре, иначе "The" и "the"
    станут разными признаками.

    Практическая ловушка урока: обучение и инференс обязаны звать ОДНУ И ТУ
    ЖЕ функцию. Разошлись — качество тихо падает, и никто не знает почему.
    """
    tokens = tokenize(text)
    stems = [stem(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(w, p, lemma_table) for w, p in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
