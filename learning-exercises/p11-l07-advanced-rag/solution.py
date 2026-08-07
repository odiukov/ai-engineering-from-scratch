"""
Advanced RAG: гибридный поиск, слияние рангов, реранкинг — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import re
from collections import Counter

# Токен: буквы и цифры, внутри допустимы дефис и точка. Так "e-4021" и "47.2m"
# остаются одним токеном, а хвостовая точка предложения отваливается.
# Именно ради кодов ошибок и чисел BM25 и держат рядом с векторным поиском.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

# Стоп-слова из урока: слова-связки не несут смысла для реранкера.
STOP_WORDS = frozenset(
    """a an and are as at be been but by do does for from had has have how in is
    it its not of on or that the this to was were what when where which who why
    with""".split()
)


def tokenize(text):
    """Разбить текст на токены: нижний регистр, знаки препинания отброшены.

    tokenize("Error code E-4021!")   ->  ['error', 'code', 'e-4021']
    tokenize("Q3 earnings: $47.2M.")  ->  ['q3', 'earnings', '47.2m']

    Ловушка: наивный text.lower().split() оставит "e-4021!" вместе с
    восклицательным знаком, и точное совпадение по коду ошибки не сработает.
    Но и резать по любому не-букво-цифре нельзя — тогда "e-4021" развалится
    на "e" и "4021", а вместе с ним и весь смысл keyword-поиска.

    Один и тот же токенайзер обязан использоваться при индексации и при
    поиске. Разные — самый частый источник "почему BM25 ничего не находит".
    """
    return TOKEN_RE.findall(text.lower())


def build_bm25_index(docs):
    """Построить индекс для BM25: длины документов, средняя длина, doc frequency.

    idx = build_bm25_index(["cat sat", "the cat"])
    idx["n_docs"]      ->  2
    idx["doc_lens"]    ->  [2, 2]
    idx["avg_dl"]      ->  2.0
    idx["doc_freqs"]   ->  {'cat': 2, 'sat': 1, 'the': 1}

    doc_freqs — в СКОЛЬКИХ документах встретился термин, а не сколько раз
    всего. Повтор слова внутри одного документа df не увеличивает: по df
    считается редкость термина по корпусу.

    На пустом корпусе avg_dl = 0.0 — делить на неё потом нельзя, это надо
    будет обойти в bm25_score.
    """
    tokens = [tokenize(d) for d in docs]
    doc_lens = [len(t) for t in tokens]
    doc_freqs = {}
    for toks in tokens:
        # set: каждый документ голосует за термин ровно один раз
        for term in set(toks):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1
    n_docs = len(docs)
    return {
        "docs": list(docs),
        "tokens": tokens,
        "doc_lens": doc_lens,
        "avg_dl": sum(doc_lens) / n_docs if n_docs else 0.0,
        "doc_freqs": doc_freqs,
        "n_docs": n_docs,
    }


def bm25_score(query, doc_id, index, k1=1.2, b=0.75):
    """Скор BM25 для одного документа: сумма вкладов по терминам запроса.

    idx = build_bm25_index(["error e-4021 happened", "everything is fine"])
    bm25_score("e-4021", 0, idx)  ->  примерно 0.6931
    bm25_score("e-4021", 1, idx)  ->  0.0   (термина в документе нет)

    Формула на один термин:
        idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
        idf = log((N - df + 0.5) / (df + 0.5) + 1)

    k1 отвечает за насыщение: документ со словом "revenue" 50 раз не в 50 раз
    релевантнее того, где оно одно. b — за нормировку по длине: без неё
    длинные документы выигрывают просто потому, что в них больше слов.

    "+1" внутри логарифма не украшение: без него idf уходит в минус для
    терминов, которые есть больше чем в половине документов, и частое слово
    начинает ШТРАФОВАТЬ документ.

    Термин, повторённый в запросе дважды, учитывается дважды — так же, как в
    эталонной реализации из урока.
    """
    if index["n_docs"] == 0:
        return 0.0
    counts = Counter(index["tokens"][doc_id])
    doc_len = index["doc_lens"][doc_id]
    avg_dl = index["avg_dl"] or 1.0  # пустой корпус сюда не дойдёт, но 0/0 — нет
    n_docs = index["n_docs"]

    score = 0.0
    for term in tokenize(query):
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        df = index["doc_freqs"].get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
    return score


def bm25_search(query, index, top_k=10):
    """Keyword-поиск: список (doc_id, score), по убыванию скора.

    idx = build_bm25_index(["error e-4021", "cancel subscription", "e-4021 fix"])
    [d for d, _ in bm25_search("e-4021", idx)]  ->  [0, 2]

    Документы с нулевым скором в выдачу НЕ попадают: BM25 честно говорит
    "у меня нет ни одного слова из запроса". Если их оставить, они займут
    места в кандидатах и испортят слияние рангов.

    Ничьи разруливаются по возрастанию doc_id — иначе порядок выдачи зависит
    от порядка обхода и тесты становятся невоспроизводимыми.
    """
    scored = []
    for doc_id in range(index["n_docs"]):
        s = bm25_score(query, doc_id, index)
        if s > 0:
            scored.append((doc_id, s))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Слить несколько ранжированных списков: сумма 1 / (k + rank).

    a = [(1, 0.9), (2, 0.8)]        # выдача векторного поиска
    b = [(2, 12.0), (3, 4.0)]       # выдача BM25
    reciprocal_rank_fusion([a, b])  ->  [(2, 0.0325...), (1, 0.0164...), (3, 0.0161...)]

    Ранги считаются с единицы: первый элемент списка даёт 1 / (k + 1).

    Складываются РАНГИ, а не скоры. В этом весь смысл: косинус лежит в
    [-1, 1], BM25 — в [0, +inf), сложить их напрямую нельзя, а ранги
    сравнимы всегда. Документ, найденный обоими ретриверами, получает две
    добавки и обходит документ, который был первым, но только в одном списке.

    k = 60 (значение из статьи Cormack et al.) гасит разрыв между первым и
    вторым местом: при k = 0 первое место весило бы вдвое больше второго.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[0]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused


def hybrid_search(query, index, vector_ranking, top_k=5, fusion_k=60):
    """Гибридный поиск: BM25 + готовая выдача векторного поиска через RRF.

    idx = build_bm25_index(["terminate your plan", "error e-4021 fix"])
    hybrid_search("e-4021", idx, [(0, 0.7), (1, 0.5)], top_k=2)
        ->  [(1, 0.0325...), (0, 0.0163...)]

    vector_ranking — уже посчитанный список (doc_id, similarity) по убыванию
    похожести. В настоящей системе его отдаёт bi-encoder; здесь модели
    эмбеддингов нет, поэтому выдачу передают снаружи. Это ровно параметр
    alpha из Weaviate collection.query.hybrid(), только вместо взвешивания
    скоров — слияние рангов.

    Каждый ретривер отдаёт top_k * 3 кандидатов: сливать имеет смысл более
    широкие списки, чем нужно на выходе, иначе документ, найденный вторым
    ретривером на 7-м месте, до слияния просто не доживёт.
    """
    candidates = top_k * 3
    keyword = bm25_search(query, index, candidates)
    vector = list(vector_ranking)[:candidates]
    return reciprocal_rank_fusion([vector, keyword], k=fusion_k)[:top_k]


def rerank(query, candidates, docs):
    """Реранкер: пересчитать кандидатов, глядя на запрос и документ вместе.

    docs = ["q3 earnings were 47.2m", "revenue strategy for next year"]
    rerank("q3 earnings", [(0, 0.01), (1, 0.02)], docs)
        ->  [(0, 4.55), (1, 0.1)]

    Скор складывается из четырёх слагаемых:
        1.0 * число общих значимых слов (стоп-слова не в счёт)
        2.0 * число совпавших биграмм запроса (порядок слов важнее слов)
        0.5 за каждое слово запроса, попавшее в первую треть документа
        5.0 * исходный скор из retrieval

    Это дешёвая имитация cross-encoder'а (cross-encoder/ms-marco-MiniLM-L-6-v2,
    Cohere Rerank 3.5): настоящий подаёт пару (query, document) в модель
    одним входом. Общее у них главное — реранкер смотрит на пару, а bi-encoder
    считает эмбеддинги по отдельности и уже не видит их взаимодействия.

    Порядок кандидатов на входе может быть любым; ничьи — по doc_id.
    """
    q_terms = [t for t in tokenize(query) if t not in STOP_WORDS]
    q_set = set(q_terms)
    q_bigrams = {(q_terms[i], q_terms[i + 1]) for i in range(len(q_terms) - 1)}

    scored = []
    for doc_id, initial_score in candidates:
        d_tokens = tokenize(docs[doc_id])
        d_set = set(d_tokens)
        d_bigrams = {(d_tokens[i], d_tokens[i + 1]) for i in range(len(d_tokens) - 1)}

        term_overlap = len(q_set & d_set)
        bigram_matches = len(q_bigrams & d_bigrams)
        # первая треть, но не меньше одного токена: иначе короткие чанки
        # никогда не получат позиционную надбавку
        head = d_tokens[: max(1, len(d_tokens) // 3)]
        position_boost = 0.5 * sum(1 for t in q_set if t in head)

        score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, score))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def parent_child_chunks(text, parent_size=200, child_size=50):
    """Parent-child чанкинг: искать по мелким кускам, отдавать в промпт крупные.

    parents, children, mapping = parent_child_chunks("a b c d e f", 4, 2)
    parents   ->  ['a b c d', 'e f']
    children  ->  ['a b', 'c d', 'e f']
    mapping   ->  {0: 0, 1: 0, 2: 1}

    Вернуть кортеж (parents, children, child_to_parent), где child_to_parent
    отображает индекс дочернего чанка в индекс родительского.

    Ловушка: последний дочерний чанк внутри родителя не должен вылезать за
    его границу — режь по min(child_end, parent_end), а не по длине текста.
    Иначе ребёнок начинает принадлежать двум родителям сразу.

    Зачем: мелкий чанк точнее матчится на запрос, крупный даёт модели
    достаточно контекста. Parent-child снимает выбор между точностью поиска
    и полнотой контекста — берём и то, и другое.
    """
    words = text.split()
    parents, children, child_to_parent = [], [], {}

    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parents.append(" ".join(words[start:parent_end]))
        parent_idx = len(parents) - 1

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_to_parent[len(children)] = parent_idx
            children.append(" ".join(words[child_start:child_end]))
            child_start = child_end

        start = parent_end

    return parents, children, child_to_parent
