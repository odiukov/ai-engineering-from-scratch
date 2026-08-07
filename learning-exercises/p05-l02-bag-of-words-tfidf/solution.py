"""
Мешок слов, TF-IDF и представление текста — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def build_vocab(docs):
    """Собрать словарь {слово: индекс} по списку токенизированных документов.

    build_vocab([["cat", "sat"], ["cat", "ran"]])  ->  {'cat': 0, 'sat': 1, 'ran': 2}
    build_vocab([])                                ->  {}

    Индекс = порядок первого появления слова. Слово, встреченное второй раз,
    нового индекса не получает.

    Ловушка: sklearn CountVectorizer сортирует словарь по алфавиту, а мы
    нумеруем по порядку появления. Оба варианта рабочие, но матрицы из них
    несравнимы — колонки стоят на разных местах.
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                # len(vocab) как следующий индекс: dict сохраняет порядок
                # вставки, так что нумерация детерминирована
                vocab[token] = len(vocab)
    return vocab


def bag_of_words(docs, vocab):
    """Матрица счётчиков: строка — документ, колонка — слово из vocab.

    docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
    vocab = {'cat': 0, 'sat': 1, 'on': 2, 'mat': 3, 'ran': 4}
    bag_of_words(docs, vocab)  ->  [[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]

    Слова, которых нет в vocab, молча пропускаются — так ведёт себя
    обученный векторизатор на новых данных (OOV-слова).

    Ловушка: [[0] * len(vocab)] * len(docs) создаст len(docs) ссылок на ОДИН
    список. Инкремент в первой строке отразится во всех. Строки надо
    создавать в цикле.

    Мешок слов выкидывает порядок: "dog bites man" и "man bites dog" дают
    одинаковую строку. Это его главное ограничение и его главная скорость.
    """
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix


def term_frequency(doc_bow, doc_length):
    """Нормировать счётчики документа на его длину.

    term_frequency([1, 1, 2], 4)  ->  [0.25, 0.25, 0.5]
    term_frequency([0, 0], 0)     ->  [0.0, 0.0]

    Ловушка: doc_length может быть нулём (пустой документ или все слова
    оказались OOV). Деления на ноль быть не должно.

    Смысл нормировки: без неё длинный документ выигрывает у короткого
    просто за счёт объёма, а не за счёт содержания.
    """
    if not doc_length:
        return [0.0] * len(doc_bow)
    return [c / doc_length for c in doc_bow]


def document_frequency(bow_matrix):
    """Для каждого слова — в скольких документах оно встретилось хотя бы раз.

    document_frequency([[1, 1, 0], [2, 0, 1]])  ->  [2, 1, 1]
    document_frequency([])                      ->  []

    Считаем ДОКУМЕНТЫ, а не вхождения: слово, встреченное в одном документе
    пять раз, даёт df = 1, а не 5. В этом вся разница между df и суммой
    колонки.
    """
    if not bow_matrix:
        return []
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    """Обратная документная частота со сглаживанием (как в sklearn).

    Формула: log((n_docs + 1) / (df + 1)) + 1.

    inverse_document_frequency([2, 1], 2)  ->  [1.0, 1.4054651081081644]
    inverse_document_frequency([0], 3)     ->  [2.386294361119891]

    Два сглаживания подряд. (n+1)/(df+1) спасает от log(x/0) на слове,
    которого нет ни в одном документе. Финальная +1 гарантирует, что слово,
    встреченное во ВСЕХ документах, получит вес 1, а не 0 — то есть не
    исчезнет из вектора совсем.

    Несглаженный вариант log(N / df) дал бы для такого слова ровно ноль.
    Это соответствует TfidfTransformer(smooth_idf=True) из sklearn.
    """
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]


def tfidf(bow_matrix):
    """Пересчитать матрицу счётчиков в матрицу весов TF-IDF.

    tfidf([[1, 1], [1, 0]])  ->  [[0.5, 0.7027325540540822], [1.0, 0.0]]

    Схема: для каждой строки берём term_frequency от её счётчиков и
    поэлементно умножаем на общий для всей матрицы вектор idf.

    Смысл: слово, частое в этом документе и редкое в корпусе, получает
    большой вес. Слово, которое есть везде, придавливается.

    Собери из уже написанных функций — формулу второй раз не пиши.
    """
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        # длина документа = сумма счётчиков строки, а не len(row):
        # len(row) — это размер словаря, он одинаков у всех документов
        tf = term_frequency(row, sum(row))
        out.append([t * i for t, i in zip(tf, idf)])
    return out


def l2_normalize(matrix):
    """Привести каждую строку к единичной длине (L2-норма).

    l2_normalize([[3.0, 4.0]])  ->  [[0.6, 0.8]]
    l2_normalize([[0.0, 0.0]])  ->  [[0.0, 0.0]]

    Ловушка: нулевая строка. Делить на ноль нельзя, возвращаем нули.

    После этого шага косинусная близость двух строк — это просто их
    скалярное произведение, отдельная нормировка больше не нужна.
    """
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm for x in row] if norm else [0.0] * len(row))
    return out


def cosine_similarity(a, b):
    """Косинус угла между двумя векторами.

    cosine_similarity([1, 0], [1, 0])  ->  1.0
    cosine_similarity([1, 0], [0, 1])  ->  0.0
    cosine_similarity([1, 1], [2, 2])  ->  1.0

    Формула: (a . b) / (|a| * |b|).

    Ключевое свойство: масштаб не влияет. Документ, переписанный вдвое
    длиннее теми же словами, останется идентичным по косинусу. Именно
    поэтому в поиске и RAG меряют косинусом, а не евклидовым расстоянием.

    Ловушка: нулевой вектор. Угла у него нет, возвращаем 0.0.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
