"""
Многоязычный NLP: перенос между языками — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def cosine_similarity(a, b):
    """Косинус угла между векторами: скалярное произведение на произведение длин.

    cosine_similarity([1, 0], [1, 0])  ->  1.0
    cosine_similarity([1, 0], [0, 1])  ->  0.0
    cosine_similarity([1, 0], [2, 0])  ->  1.0  (длина не важна, важно направление)
    cosine_similarity([0, 0], [1, 0])  ->  0.0  (нулевой вектор, а не ZeroDivisionError)

    Вся многоязычность держится на одном факте: "The cat is sleeping" и
    "Le chat dort" в общем пространстве смотрят почти в одну сторону, а "The
    dog is barking" — в другую. Косинус это и измеряет.

    Ловушка: normalize_embeddings=True в sentence-transformers уже делит на
    длину, и косинус там вырождается в np.dot. Если нормировки не было,
    делить обязательно, иначе длинные тексты будут «похожи» на всё подряд.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    # нулевой вектор направления не имеет: 0.0 честнее, чем падение
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cross_lingual_retrieve(query_vector, documents, top_k=3):
    """Поиск без учёта языка: top_k ближайших документов к запросу.

    documents — список пар (метка, вектор). Возвращает список пар
    (метка, косинус), отсортированный по убыванию; при равных значениях
    сохраняется исходный порядок.

    DOCS = [("fr:chat", [0.98, 0.2]), ("en:dog", [0.0, 1.0])]
    cross_lingual_retrieve([1.0, 0.0], DOCS, top_k=1)  ->  [("fr:chat", ~0.98)]

    Ради этого всё и затевалось: запрос на английском достаёт документ на
    французском, потому что язык в общем пространстве не кодируется — только
    смысл. Ни перевода, ни отдельного индекса на каждый язык.
    """
    scored = [(label, cosine_similarity(query_vector, vec)) for label, vec in documents]
    # sorted устойчив, поэтому минус перед оценкой сохраняет исходный
    # порядок при равных косинусах
    scored.sort(key=lambda pair: -pair[1])
    return scored[:top_k]


def zero_shot_classify(text_vector, label_vectors):
    """Zero-shot классификация: softmax по близости текста к каждой метке.

    label_vectors — dict метка -> вектор гипотезы ("This text is about {}.").
    Возвращает dict метка -> вероятность, отсортированный по убыванию.
    Сумма вероятностей ровно 1.

    LABELS = {"positive": [1.0, 0.0], "negative": [0.0, 1.0]}
    zero_shot_classify([1.0, 0.0], LABELS)
        ->  {"positive": ~0.731, "negative": ~0.269}

    Ни одной размеченной пары в целевом языке не потребовалось: модель
    сравнивает смысл текста со смыслом описания метки, а в общем
    пространстве это работает на любом из ста языков.

    Ловушка: exp от больших чисел переполняется. Вычитай максимум перед
    экспонентой — на сумму это не влияет, на устойчивость влияет сильно.
    """
    scores = {label: cosine_similarity(text_vector, v) for label, v in label_vectors.items()}
    if not scores:
        return {}
    top = max(scores.values())
    exps = {label: math.exp(s - top) for label, s in scores.items()}
    total = sum(exps.values())
    ordered = sorted(exps.items(), key=lambda pair: -pair[1])
    return {label: value / total for label, value in ordered}


def subword_segment(word, vocab):
    """Жадная разбивка слова на подслова: самое длинное совпадение слева.

    vocab — множество известных подслов. На каждом шаге берётся самый
    длинный префикс остатка, который есть в vocab. Если не подходит ни один
    префикс, откусывается один символ — это byte fallback, из-за него OOV не
    бывает вообще.

    V = {"anti", "dis", "establish", "ment"}
    subword_segment("antiestablishment", V)  ->  ["anti", "establish", "ment"]
    subword_segment("xyz", V)                ->  ["x", "y", "z"]

    Свойство, которое обязано держаться всегда: "".join(результат) == word.

    Тот самый общий словарь из урока: "anti-" в английском и итальянском —
    один и тот же токен, поэтому морфология переносится между родственными
    языками бесплатно.
    """
    pieces = []
    i = 0
    while i < len(word):
        # длиннейший префикс: перебираем от конца остатка к началу
        for end in range(len(word), i, -1):
            if word[i:end] in vocab:
                pieces.append(word[i:end])
                i = end
                break
        else:
            # ни один префикс не подошёл — байтовый откат на один символ
            pieces.append(word[i])
            i += 1
    return pieces


def tokenization_fertility(texts, vocab):
    """Fertility: сколько токенов приходится на одно слово. Меньше — лучше.

    Слова режутся по пробелам, каждое проходит через subword_segment.

    V = {"cat", "sleep", "ing"}
    tokenization_fertility(["cat sleeping"], V)  ->  1.5  (1 + 2 токена на 2 слова)
    tokenization_fertility(["кот"], V)           ->  3.0  (byte fallback посимвольно)

    Тексты без слов дают 0.0.

    Это налог на токенизацию из урока. Для языка, не попавшего в словарь,
    fertility вырастает в 3-5 раз: то же предложение съедает в разы больше
    контекста, обучение идёт медленнее, а на рассуждение остаётся меньше
    позиций. Данными это не лечится — лечится только словарём.
    """
    total_words = 0
    total_pieces = 0
    for text in texts:
        for word in text.split():
            total_words += 1
            total_pieces += len(subword_segment(word, vocab))
    if total_words == 0:
        return 0.0
    return total_pieces / total_words


def language_similarity(features_a, features_b):
    """Типологическая близость языков: доля совпавших общих признаков WALS.

    features_* — dict признак -> значение. Считаются только признаки, которые
    есть у ОБОИХ языков; ответ — доля тех из них, где значения совпали.
    Общих признаков нет — 0.0.

    A = {"word_order": "SVO", "gender": "yes", "case": "no"}
    B = {"word_order": "SVO", "gender": "no",  "case": "no"}
    language_similarity(A, B)  ->  2/3 = ~0.667
    language_similarity(A, {})  ->  0.0

    Мера в духе qWALS: чем ближе языки по устройству, тем лучше перенос
    дообучения с одного на другой. Именно поэтому для славянских целей
    немецкий или русский нередко обыгрывают английский.
    """
    shared = set(features_a) & set(features_b)
    if not shared:
        return 0.0
    matches = sum(1 for key in shared if features_a[key] == features_b[key])
    return matches / len(shared)


def rank_source_languages(target_features, candidates, weight=0.5):
    """LANGRANK: какой язык брать источником дообучения. Список (язык, оценка).

    candidates — dict язык -> (признаки, размер корпуса). Оценка каждого:
        weight * language_similarity + (1 - weight) * log(1 + size) / log(1 + max_size)
    Отсортировано по убыванию оценки; при равенстве — исходный порядок.

    weight=1.0 — решает только типология, размер корпуса игнорируется.
    weight=0.0 — решает только объём данных, и побеждает английский.

    rank_source_languages(HINDI, {"english": (EN, 10**9), "urdu": (UR, 10**6)}, 1.0)
        ->  [("urdu", ...), ("english", ...)]  урду ближе типологически

    Практическое правило урока: если у целевого языка есть типологически
    близкий высокоресурсный родственник, начинай с него, а английский бери
    для сравнения. По умолчанию все делают наоборот.
    """
    if not candidates:
        return []
    max_size = max(size for _, size in candidates.values())
    log_max = math.log(1 + max_size)
    scored = []
    for lang, (features, size) in candidates.items():
        similarity = language_similarity(target_features, features)
        # log, а не сам размер: между 10^6 и 10^9 разница в три порядка,
        # линейная шкала раздавила бы типологию в ноль
        resource = math.log(1 + size) / log_max if log_max > 0 else 0.0
        scored.append((lang, weight * similarity + (1 - weight) * resource))
    scored.sort(key=lambda pair: -pair[1])
    return scored


def per_language_accuracy(records):
    """Точность отдельно по каждому языку. Возвращает dict язык -> доля.

    records — список троек (язык, предсказание, правильный ответ).
    Ключи отсортированы по алфавиту, чтобы результат был воспроизводим.

    per_language_accuracy([("en", 1, 1), ("en", 0, 1), ("hi", 0, 1)])
        ->  {"en": 0.5, "hi": 0.0}
    per_language_accuracy([])  ->  {}

    Урок настаивает: агрегат не считается результатом. Если на английском
    5000 примеров с точностью 0.95, а на амхарском 100 с точностью 0.30,
    общая цифра будет 0.94 и провал никто не заметит. Разбивка по языкам —
    единственный способ увидеть длинный хвост.
    """
    totals = {}
    correct = {}
    for language, prediction, gold in records:
        totals[language] = totals.get(language, 0) + 1
        if prediction == gold:
            correct[language] = correct.get(language, 0) + 1
    return {lang: correct.get(lang, 0) / totals[lang] for lang in sorted(totals)}
