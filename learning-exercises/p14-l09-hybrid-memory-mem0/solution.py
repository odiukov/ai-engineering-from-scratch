"""
Гибридная память: вектор + граф + KV — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Один store всегда неправ для двух классов запросов из трёх: вектор не умеет
точные факты, KV не умеет похожесть, граф не умеет ни того, ни другого, зато
отвечает про связи. Mem0 пишет во все три параллельно, а на чтении сливает
их одной функцией ранжирования.

Соответствие настоящему API Mem0:

    embed            <-  embedding-модель (здесь — детерминированный хеш)
    cosine           <-  метрика векторного индекса
    vector_search    <-  поиск по vector store
    kv_lookup        <-  чтение по ключу (user_id, fact_type, entity)
    graph_add_edge   <-  запись ребра в Mem0g с временной инвалидацией
    graph_neighbors  <-  запрос подграфа, в том числе «как было в марте»
    fuse_score       <-  relevance + importance + recency
    hybrid_search    <-  m.search(query, user_id=...)

Ни сети, ни модели: «эмбеддинг» — детерминированная функция от строки, время
всегда приходит параметром now. Запись памяти выглядит так:

    {"rid": "m001", "text": "ava lives in Berlin", "user_id": "ava",
     "session_id": "s001", "scope": "user", "importance": 0.6,
     "ts": 1000.0, "kv": {"city": "Berlin"}}
"""

import hashlib
import math

# Веса слияния. Их крутят под продукт: чату важнее свежесть, комплаенсу —
# важность, поисковому агенту — релевантность.
DEFAULT_WEIGHTS = {"relevance": 0.6, "importance": 0.2, "recency": 0.2}

# Период полураспада свежести: сутки. Через сутки вклад recency падает вдвое.
HALFLIFE_SECONDS = 86400.0

# Релевантность точного попадания в KV. Единица, потому что «ключ совпал» —
# это не «похоже», а «то самое».
KV_EXACT_RELEVANCE = 1.0


def embed(text, dim=64):
    """Детерминированный «эмбеддинг» строки: список длины dim.

    Хеширующий трюк: каждое слово через sha256 отображается в один индекс,
    в который прибавляется единица.

    embed("ava") == embed("ava")            ->  True  (всегда, в любом процессе)
    embed("ava lives") == embed("lives ava") ->  True  (мешок слов)
    len(embed("ava", dim=8))                ->  8

    Ловушка, из-за которой тесты начинают «мигать»: встроенный hash() для
    строк солится случайным значением на каждый запуск Python. Одинаковый
    текст даст разные векторы в разных процессах, и индекс, собранный вчера,
    сегодня перестанет искать. Нужен hashlib.

    Цена трюка — коллизии: два разных слова могут попасть в один индекс, и
    непохожие тексты будут выглядеть похожими. Лечится увеличением dim.
    """
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # int.from_bytes по первым 8 байтам: одного байта хватило бы только
        # для dim <= 256, а так dim ограничен только здравым смыслом
        index = int.from_bytes(digest[:8], "big") % dim
        vector[index] += 1.0
    return vector


def cosine(a, b):
    """Косинусная близость двух векторов. Ноль, если хотя бы один нулевой.

    cosine([1.0, 0.0], [1.0, 0.0])  ->  1.0
    cosine([1.0, 0.0], [0.0, 1.0])  ->  0.0
    cosine([1.0, 0.0], [2.0, 0.0])  ->  1.0   (длина не важна, важен угол)
    cosine([0.0, 0.0], [1.0, 0.0])  ->  0.0

    Нулевой вектор — это текст без слов. Деление на его норму даёт
    ZeroDivisionError посреди поиска; отвечать 0.0 честнее: «ни на что не
    похоже» — корректный ответ, падение — нет.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search(records, query, top_k=5, dim=64):
    """Поиск по векторному store. Список пар (близость, запись), лучшие первыми.

    records = [{"rid": "m001", "text": "ava lives in Berlin", ...}]
    vector_search(records, "ava lives in Berlin")  ->  [(1.0, запись m001)]
    vector_search(records, "")                     ->  []

    Записи с нулевой близостью не возвращаются: вектор, который «ни на что не
    похож», в промпте не нужен.

    Ловушка детерминизма: при равной близости порядок задаётся вторым ключом
    по rid. Иначе одинаковый запрос будет давать разный контекст в
    зависимости от порядка записей в списке.
    """
    q_vector = embed(query, dim=dim)
    scored = []
    for record in records:
        similarity = cosine(q_vector, embed(record["text"], dim=dim))
        if similarity <= 0.0:
            continue
        scored.append((similarity, record))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["rid"]))
    return scored[:top_k]


def kv_lookup(records, user_id, fact_type):
    """Точное чтение факта по ключу. Самая свежая запись или None.

    Ключ KV в Mem0 — тройка (user_id, fact_type, entity). Здесь entity лежит
    в самой записи, а ищем мы по первым двум частям ключа.

    recs = [{"rid": "m001", "user_id": "ava", "ts": 1.0, "kv": {"city": "Berlin"}},
            {"rid": "m002", "user_id": "ava", "ts": 2.0, "kv": {"city": "Lisbon"}}]
    kv_lookup(recs, "ava", "city")   ->  запись m002
    kv_lookup(recs, "bob", "city")   ->  None

    Две вещи, ради которых KV вообще держат рядом с вектором:
      * ответ точный, а не «похожий»: телефон пользователя нельзя искать по
        косинусу;
      * изоляция по user_id обязательна на уровне выборки. Именно так
        случаются инциденты «ассистент рассказал Алисе про проект Боба».

    Свежесть решает: при нескольких значениях берётся максимальный ts, ничьи
    разруливаются по rid — снова ради детерминизма.
    """
    candidates = [
        r for r in records
        if r["user_id"] == user_id and fact_type in r.get("kv", {})
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["ts"], r["rid"]))


def graph_add_edge(edges, subject, relation, obj, now):
    """Добавить ребро в граф, старое с тем же (subject, relation) — погасить.

    Вернуть НОВЫЙ список рёбер.

    graph_add_edge([], "ava", "lives_in", "Berlin", now=100.0)
        ->  [{"subject": "ava", "relation": "lives_in", "obj": "Berlin",
              "valid": True, "valid_from": 100.0, "invalid_from": None}]

    Второй вызов с "Lisbon" в момент 200.0 не удаляет берлинское ребро, а
    ставит ему valid=False и invalid_from=200.0. Список растёт, а не
    сокращается.

    Это temporal invalidation, она же soft delete. Удалить противоречащий
    факт — значит потерять ответ на вопрос «а что было правдой в марте» и
    лишить аудит возможности проверить решение агента задним числом.

    Рёбра с другим relation не трогаются: противоречие — это конфликт по
    ОДНОМУ отношению, а не «всё, что мы знали про ava, устарело».
    """
    updated = []
    for edge in edges:
        if (edge["valid"] and edge["subject"] == subject
                and edge["relation"] == relation):
            closed = dict(edge)
            closed["valid"] = False
            closed["invalid_from"] = now
            updated.append(closed)
        else:
            updated.append(edge)
    updated.append({
        "subject": subject,
        "relation": relation,
        "obj": obj,
        "valid": True,
        "valid_from": now,
        "invalid_from": None,
    })
    return updated


def graph_neighbors(edges, subject, as_of=None):
    """Рёбра субъекта. as_of=None — только действующие, иначе — на момент as_of.

    После двух вызовов graph_add_edge (Berlin в 100.0, Lisbon в 200.0):

    graph_neighbors(edges, "ava")             ->  [ребро на Lisbon]
    graph_neighbors(edges, "ava", as_of=150)  ->  [ребро на Berlin]
    graph_neighbors(edges, "bob")             ->  []

    Ребро действует на момент t, если valid_from <= t и (invalid_from is None
    или t < invalid_from). Правая граница строгая: в саму секунду 200.0
    берлинское ребро уже погашено, лиссабонское уже действует. Нестрогая
    граница вернула бы оба, то есть два разных города одновременно.
    """
    result = []
    for edge in edges:
        if edge["subject"] != subject:
            continue
        if as_of is None:
            if edge["valid"]:
                result.append(edge)
            continue
        if edge["valid_from"] <= as_of and (
                edge["invalid_from"] is None or as_of < edge["invalid_from"]):
            result.append(edge)
    return result


def fuse_score(record, relevance, now, weights=DEFAULT_WEIGHTS,
               halflife=HALFLIFE_SECONDS):
    """Взвешенная сумма: релевантность + важность + свежесть.

    Свежесть — экспоненциальный распад: 0.5 ** (прошло / halflife).

    rec = {"importance": 0.5, "ts": 0.0, ...}
    fuse_score(rec, 1.0, now=0.0)      ->  0.6*1.0 + 0.2*0.5 + 0.2*1.0 = 0.9
    fuse_score(rec, 1.0, now=86400.0)  ->  0.6*1.0 + 0.2*0.5 + 0.2*0.5 = 0.8

    Это сумма, а не иерархия. Иерархия («сначала релевантные, потом среди них
    свежие») не даёт свежему и важному факту обойти чуть более релевантный, но
    протухший — а именно этого от памяти и ждут.

    Ловушка: now может оказаться раньше ts (часы разъехались, запись пришла
    из будущего). Отрицательное «прошло» превратит распад в рост, и старая
    запись выиграет. Обрезай снизу нулём.
    """
    elapsed = max(0.0, now - record["ts"])
    recency = 0.5 ** (elapsed / halflife) if halflife > 0 else 1.0
    return (weights["relevance"] * relevance
            + weights["importance"] * record["importance"]
            + weights["recency"] * recency)


def hybrid_search(records, query, user_id, now, top_k=3,
                  weights=DEFAULT_WEIGHTS, halflife=HALFLIFE_SECONDS,
                  scope=None):
    """Слить векторный и KV-путь в один ранжированный список.

    Вернуть список пар (счёт, запись), лучшие первыми.

    Порядок работы:
      1. отбросить всё, что не принадлежит user_id (и не входит в scope,
         если он задан);
      2. векторный путь: близость из vector_search как relevance;
      3. KV-путь: если слово из запроса совпало с fact_type записи,
         relevance = KV_EXACT_RELEVANCE. Совпадение ключа — это не
         «похоже», это «то самое», и оно обязано обходить косинус;
      4. одна запись могла прийти обоими путями — оставить лучший счёт;
      5. отсортировать по счёту, ничьи — по rid.

    hybrid_search(recs, "what city does ava live in", "ava", now=0.0)[0][1]["rid"]
        ->  запись с kv={"city": ...}, даже если её текст непохож на запрос

    Изоляция по user_id делается ДО ранжирования, а не после отбора top_k.
    Иначе чужая запись съест место в выдаче и вернётся пустой список там,
    где свои записи были.
    """
    visible = [
        r for r in records
        if r["user_id"] == user_id and (scope is None or r["scope"] == scope)
    ]
    fused = {}
    for relevance, record in vector_search(visible, query, top_k=top_k * 3):
        fused[record["rid"]] = (
            fuse_score(record, relevance, now, weights, halflife), record)

    q_tokens = set(query.lower().split())
    for record in visible:
        if not any(fact_type.lower() in q_tokens for fact_type in record.get("kv", {})):
            continue
        score = fuse_score(record, KV_EXACT_RELEVANCE, now, weights, halflife)
        previous = fused.get(record["rid"])
        if previous is None or score > previous[0]:
            fused[record["rid"]] = (score, record)

    ordered = sorted(fused.values(), key=lambda pair: (-pair[0], pair[1]["rid"]))
    return ordered[:top_k]
