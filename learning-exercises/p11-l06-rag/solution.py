"""
RAG: чанки, TF-IDF, поиск, промпт — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
from collections import Counter

# Шапка RAG-промпта: модели прямым текстом запрещают отвечать мимо контекста.
RAG_INSTRUCTION = (
    "Answer the question based ONLY on the following context.\n"
    "If the context doesn't contain enough information, say "
    '"I don\'t have enough information to answer that."'
)


def chunk_text(text, chunk_size=200, overlap=50):
    """Нарезать текст на куски по chunk_size слов с перекрытием в overlap слов.

    chunk_text("a b c d e", chunk_size=3, overlap=1)  ->  ["a b c", "c d e"]
    chunk_text("", 3, 1)                              ->  []

    Шаг между началами кусков — chunk_size - overlap. Перекрытие затем и
    нужно, чтобы предложение, попавшее на границу, целиком оказалось хотя бы
    в одном куске.

    Ловушка: overlap >= chunk_size даёт нулевой или отрицательный шаг и
    бесконечный цикл. Проверь это ЯВНО и брось ValueError, иначе индексация
    просто зависнет.

    Соответствует RecursiveCharacterTextSplitter из LangChain, только без
    иерархии разделителей.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap >= chunk_size or overlap < 0:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    words = text.split()
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        # последний кусок уже дотянулся до конца — второй такой же не нужен
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_vocabulary(documents):
    """Отсортированный список всех слов корпуса в нижнем регистре.

    build_vocabulary(["Cat sat", "cat ran"])  ->  ["cat", "ran", "sat"]

    Сортировка не для красоты: позиция слова в этом списке — это номер
    координаты эмбеддинга. Порядок обязан быть одинаковым при индексации
    и при запросе, иначе векторы окажутся в разных пространствах.
    """
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)


def compute_idf(documents, vocab):
    """IDF каждого слова словаря: log((N + 1) / (df + 1)) + 1.

    Возвращает список той же длины, что vocab.

    Для слова из двух документов при N=2:  log(3/3) + 1 = 1.0
    Для слова из одного документа при N=2: log(3/2) + 1 ≈ 1.405

    df — в скольких документах слово встретилось хотя бы раз (не сколько
    раз всего). Единицы в формуле — сглаживание: без них слово из всех
    документов даёт log(1) = 0 и полностью выпадает из вектора.

    Смысл: частое слово («the») почти ничего не различает и получает вес
    около единицы, редкое — заметно больше.
    """
    n = len(documents)
    # множества слов считаем один раз: иначе на каждое слово словаря
    # пришлось бы заново разбирать весь корпус
    doc_words = [set(doc.lower().split()) for doc in documents]
    return [
        math.log((n + 1) / (sum(1 for words in doc_words if w in words) + 1)) + 1
        for w in vocab
    ]


def tfidf_embed(text, vocab, idf):
    """Вектор TF-IDF текста в координатах vocab.

    TF слова — его доля среди слов текста, вес координаты — TF * IDF.
    Слова вне словаря игнорируются.

    tfidf_embed("cat cat", ["cat", "dog"], [1.0, 1.0])  ->  [1.0, 0.0]
    tfidf_embed("", ["cat"], [1.0])                     ->  [0.0]

    Пустой текст — нулевой вектор, а не деление на ноль.
    """
    words = text.lower().split()
    if not words:
        return [0.0] * len(vocab)
    counts = Counter(words)
    total = len(words)
    return [counts.get(w, 0) / total * i for w, i in zip(vocab, idf)]


def cosine_similarity(a, b):
    """Косинус угла между векторами: от -1 до 1, у TF-IDF — от 0 до 1.

    cosine_similarity([1, 0], [1, 0])  ->  1.0
    cosine_similarity([1, 0], [0, 1])  ->  0.0
    cosine_similarity([1, 0], [5, 0])  ->  1.0  (длина не важна)

    Нулевой вектор: возвращай 0.0, а не деление на ноль. Запрос из слов,
    которых нет в словаре, даёт ровно такой вектор.

    Именно независимость от длины делает косинус метрикой по умолчанию в
    RAG: короткий вопрос сравнивается с длинным чанком напрямую.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def search(query_embedding, stored_embeddings, top_k=5):
    """Перебором найти top_k ближайших чанков. Возвращает [(индекс, score)].

    search([1, 0], [[1, 0], [0, 1]], top_k=1)  ->  [(0, 1.0)]

    При равном score первым идёт меньший индекс — иначе выдача пляшет от
    запуска к запуску и тесты начинают мигать.

    Это FlatIndex из FAISS: честный O(n) перебор. Тысяч до ста работает,
    дальше нужен HNSW.
    """
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(stored_embeddings)
    ]
    # sorted стабилен, а исходный порядок — по индексу: равные score сами
    # выстроятся по возрастанию номера
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def build_rag_prompt(query, retrieved_chunks):
    """Собрать промпт из инструкции, найденных чанков и вопроса.

    Чанки нумеруются как [Source 1], [Source 2], ... и разделяются строкой
    "\\n\\n---\\n\\n". Дальше идут "Context:", сам контекст, "Question: ...",
    "Answer:".

    Нумерация нужна, чтобы модель могла сослаться на источник, а ты —
    проверить, откуда взялся ответ. Это и есть auditability, которой нет
    у fine-tuning.

    Пустой список чанков — законная ситуация: ретривер ничего не нашёл.
    Промпт всё равно собирается, и инструкция велит честно сказать
    "I don't have enough information".
    """
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    return f"{RAG_INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Доля релевантных документов, попавших в первые k позиций выдачи.

    recall_at_k([3, 1, 7], [1, 9], 3)  ->  0.5
    recall_at_k([3, 1, 7], [1, 9], 1)  ->  0.0
    recall_at_k([1], [], 1)            ->  1.0

    Пустой список релевантных — вопрос не имеет ответа в корпусе; считаем
    полноту равной 1.0, а не делим на ноль.

    Ключевое свойство метрики: с ростом k она не убывает. Если у тебя
    recall@10 меньше recall@5 — ошибка в реализации, а не в данных.
    Поэтому по одной точке качество ретривера не оценивают: k=100 даст
    почти единицу и ничего не скажет.
    """
    if not relevant_ids:
        return 1.0
    top = set(retrieved_ids[:k])
    return len(top & set(relevant_ids)) / len(set(relevant_ids))
