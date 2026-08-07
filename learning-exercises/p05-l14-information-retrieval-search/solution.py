"""
Информационный поиск — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import re
from collections import Counter


def tokenize(text):
    """Текст -> список токенов: нижний регистр, только буквы и цифры.

    tokenize("Section 420 IPC!")  ->  ["section", "420", "ipc"]
    tokenize("--- ???")           ->  []

    Цифры остаются токенами намеренно: номера статей, коды ошибок и
    артикулы — ровно то, ради чего в гибридном поиске держат BM25.
    """
    # findall по «хорошим» символам вместо split по «плохим»: не надо
    # перечислять всю пунктуацию мира. re кеширует скомпилированный шаблон
    # сам, отдельная константа не нужна
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(corpus, k1=1.5, b=0.75):
    """Собрать индекс BM25 из списка документов-строк.

    Возвращает словарь с полями:
      "docs"    — список токенизированных документов,
      "df"      — Counter: в скольких документах встречается термин,
      "n_docs"  — сколько всего документов,
      "avg_dl"  — средняя длина документа в токенах,
      "k1", "b" — параметры BM25.

    Пустой корпус — ValueError: средней длины у него нет.

    Ловушка: df считает ДОКУМЕНТЫ, а не вхождения. Слово, десять раз
    повторённое в одном документе, добавляет к df единицу, иначе idf
    поедет.
    """
    if not corpus:
        raise ValueError("corpus must not be empty")
    docs = [tokenize(document) for document in corpus]
    df = Counter()
    for doc in docs:
        # set(doc), а не doc: считаем документы, а не вхождения
        for term in set(doc):
            df[term] += 1
    return {
        "docs": docs,
        "df": df,
        "n_docs": len(docs),
        "avg_dl": sum(len(doc) for doc in docs) / len(docs),
        "k1": k1,
        "b": b,
    }


def bm25_idf(index, term):
    """IDF в варианте BM25: log(1 + (N - n + 0.5) / (n + 0.5)).

    Чем реже термин, тем больше вес. Слово, которое есть во всех
    документах, получает маленький, но всё ещё ПОЛОЖИТЕЛЬНЫЙ вес —
    единица под логарифмом не даёт формуле уйти в минус.

    Термин, которого в корпусе нет вообще, получает максимальный idf:
    n = 0, и это нормально — на скор он всё равно не повлияет, потому что
    его частота в документе нулевая.
    """
    n = index["df"].get(term, 0)
    return math.log(1 + (index["n_docs"] - n + 0.5) / (n + 0.5))


def bm25_score(index, query, doc_idx):
    """Скор BM25 запроса против одного документа.

    Вклад термина: idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg_dl)).

    Три вещи, которые надо увидеть глазами:
      * НАСЫЩЕНИЕ по частоте. Десять вхождений дают далеко не в десять раз
        больше единицы — вклад термина сверху ограничен idf * (k1 + 1).
        Это главное отличие от tf-idf, где вклад растёт линейно и без
        предела.
      * k1 = 0 выключает частоту совсем: остаётся «слово есть / слова нет».
      * b нормирует на длину. При b = 0 длина документа не важна, при
        b = 0.75 длинный документ с той же частотой проигрывает короткому.

    Термин, которого в документе нет, вклада не даёт — цикл его пропускает.
    """
    doc = index["docs"][doc_idx]
    freq = Counter(doc)
    dl = len(doc)
    k1, b = index["k1"], index["b"]
    score = 0.0
    for term in tokenize(query):
        f = freq.get(term, 0)
        if f == 0:
            continue  # ноль в числителе — считать нечего, только время тратить
        norm = 1 - b + b * dl / index["avg_dl"]
        score += bm25_idf(index, term) * f * (k1 + 1) / (f + k1 * norm)
    return score


def bm25_rank(index, query, top_k=10):
    """Ранжирование корпуса по BM25: список (скор, номер документа), убывая.

    Документы с нулевым скором НЕ возвращаются: они ничем не совпали с
    запросом, и в слияние рангов им попадать незачем.

    bm25_rank(index, "zzz qqq")  ->  []   ни одного общего слова

    Это и есть хрупкость BM25 из урока: перефразированный запрос без общих
    слов даёт пустую выдачу, каким бы релевантным ни был документ по
    смыслу. Ровно эту дыру закрывает плотный поиск.

    При равных скорах впереди документ с меньшим номером — выдача обязана
    быть воспроизводимой.
    """
    scored = []
    for doc_idx in range(index["n_docs"]):
        score = bm25_score(index, query, doc_idx)
        if score > 0.0:
            scored.append((score, doc_idx))
    # ключ (-score, idx): по скору убывая, при ничьей по номеру возрастая
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[:top_k]


def dense_rank(query_vector, doc_vectors, top_k=10):
    """Плотный поиск: косинусная близость запроса ко всем векторам документов.

    Возвращает (близость, номер документа), убывая.

    Косинус, а не скалярное произведение: длина вектора не должна решать,
    кто релевантнее. Именно поэтому эмбеддинги в проде нормируют — тогда
    скалярное произведение УЖЕ равно косинусу и считается быстрее.

    Нулевой вектор даёт близость 0.0, а не деление на ноль.

    В отличие от BM25, здесь выдача никогда не пустая: у всего есть
    какая-то близость. Это и сила (находит перефразировки), и слабость
    (уверенно возвращает мусор, когда ничего подходящего нет).
    """
    q_norm = math.sqrt(sum(v * v for v in query_vector))
    scored = []
    for doc_idx, vector in enumerate(doc_vectors):
        d_norm = math.sqrt(sum(v * v for v in vector))
        if q_norm == 0.0 or d_norm == 0.0:
            similarity = 0.0
        else:
            dot = sum(q * d for q, d in zip(query_vector, vector))
            similarity = dot / (q_norm * d_norm)
        scored.append((similarity, doc_idx))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[:top_k]


def reciprocal_rank_fusion(rankings, k=60):
    """Слияние нескольких выдач по позициям: RRF.

    Документ получает сумму 1 / (k + позиция + 1) по всем выдачам, где он
    встретился. Позиции считаются с нуля.

    Ключевое свойство: используются ТОЛЬКО позиции, сами скоры
    игнорируются. BM25 выдаёт числа порядка единиц, косинус — от -1 до 1;
    складывать их напрямую нельзя, а ранги складывать можно.

    k = 60 из оригинальной статьи: чем больше k, тем ровнее вклад разных
    позиций.

    Документ, попавший в обе выдачи, обгоняет документ, который был первым
    только в одной. В этом весь смысл гибридного поиска.
    """
    scores = {}
    for ranking in rankings:
        for rank, (_score, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [(score, doc_idx) for doc_idx, score in fused]


def evaluate_rankings(rankings, relevant_ids, k=10):
    """Качество ретривера: вернуть (recall@k, MRR).

    rankings — по одной выдаче (скор, номер) на запрос, relevant_ids — по
    одному правильному документу на запрос.

    recall@k — доля запросов, где правильный документ попал в первые k.
    MRR — среднее от 1 / (позиция + 1) по ВСЕЙ выдаче; запрос, где
    документа не нашлось вовсе, вносит 0.

    Разная длина списков — ValueError.

    Для RAG recall@k важнее всего: читатель физически не может ответить по
    документу, которого ретривер не принёс.
    """
    if len(rankings) != len(relevant_ids):
        raise ValueError("one relevant id per ranking is required")
    if not rankings:
        return 0.0, 0.0
    hits = 0
    reciprocal_sum = 0.0
    for ranking, relevant in zip(rankings, relevant_ids):
        ids = [doc_idx for _score, doc_idx in ranking]
        if relevant in ids:
            position = ids.index(relevant)
            # recall смотрит только на первые k, MRR — на всю выдачу
            if position < k:
                hits += 1
            reciprocal_sum += 1.0 / (position + 1)
    return hits / len(rankings), reciprocal_sum / len(rankings)
