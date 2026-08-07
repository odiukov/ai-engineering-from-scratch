"""
Мультимодальный RAG и кросс-модальный поиск — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Как цитируется источник каждой модальности в ответе генератора.
CITATION_TAGS = {
    "text": "[text {i}]",
    "image": "[img {i}]",
    "audio": "[audio {i}]",
}

# Слова запроса, по которым MoE-гейт понимает, какой модальности верить.
MODALITY_KEYWORDS = {
    "text": ("review", "menu", "vegan", "price"),
    "image": ("light", "photo", "looks", "interior"),
    "audio": ("quiet", "loud", "noisy", "music"),
}


def min_max_normalize(scores):
    """Привести баллы одного ретривера к отрезку [0, 1]. Возвращает НОВЫЙ словарь.

    min_max_normalize({"a": 2.0, "b": 4.0, "c": 3.0})
        ->  {"a": 0.0, "b": 1.0, "c": 0.5}
    min_max_normalize({"a": 7.0, "b": 7.0})  ->  {"a": 1.0, "b": 1.0}
    min_max_normalize({})                    ->  {}

    Зачем: BM25 выдаёт баллы порядка десятков, косинус — числа из [-1, 1].
    Складывать их напрямую значит отдать всю власть тому ретриверу, у
    которого шкала крупнее.

    Ловушка: все баллы равны — знаменатель ноль. Договорённость: такой
    ретривер не различает кандидатов, все получают 1.0.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def score_fusion(score_maps, weights):
    """Score fusion: нормировать каждый ретривер отдельно, затем взвешенно сложить.

    score_maps — список словарей doc_id -> балл, weights — веса той же длины.

    score_fusion([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], [0.7, 0.3])
        ->  {"a": 0.7, "b": 0.3}

    Кандидат, которого один ретривер не вернул вовсе, получает от него 0 —
    у каждой модальности своя рецессия, и требовать полного пересечения
    нельзя.

    Ловушка: нормировать НАДО, и до сложения. Иначе вес 0.7 у ретривера с
    диапазоном баллов 0..1 проиграет весу 0.3 у ретривера с диапазоном
    0..100.
    """
    fused = {}
    for scores, w in zip(score_maps, weights):
        for doc_id, value in min_max_normalize(scores).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w * value
    return fused


def top_k(scored, k=3):
    """Top-k пар (doc_id, score) по убыванию балла.

    top_k({"a": 0.2, "b": 0.9, "c": 0.5}, 2)  ->  [("b", 0.9), ("c", 0.5)]

    Равные баллы разводятся по doc_id по возрастанию: выдача обязана быть
    воспроизводимой, иначе A/B-тест фьюжна нечем измерить.
    """
    # минус в ключе вместо reverse=True: reverse перевернул бы и порядок
    # doc_id при равных баллах
    return sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:k]


def moe_gate(query, base_weight=0.1):
    """MoE-гейт: веса модальностей, выведенные из слов запроса.

    Возвращает словарь модальность -> вес, сумма весов равна 1.

    moe_gate("quiet vegan brunch with natural light")
        ->  у text, image и audio веса больше базового
    moe_gate("")
        ->  {"text": 1/3, "image": 1/3, "audio": 1/3}

    Каждая модальность начинает с base_weight и получает +1 за каждое
    попадание своего слова из MODALITY_KEYWORDS в запрос. В конце веса
    нормируются, чтобы складываться в единицу.

    Смысл: визуальный вопрос должен весить картинки выше, а вопрос про
    цену — отзывы. Score fusion с фиксированными весами так не умеет.
    """
    words = set(query.lower().split())
    raw = {}
    for modality, keywords in MODALITY_KEYWORDS.items():
        raw[modality] = base_weight + sum(1.0 for kw in keywords if kw in words)
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}


def recall_at_k(ranked, relevant, k):
    """Доля релевантных документов, попавших в первые k позиций выдачи.

    ranked — список doc_id по убыванию балла, relevant — множество doc_id.

    recall_at_k(["a", "b", "c"], {"a", "c"}, 2)  ->  0.5
    recall_at_k(["a", "b", "c"], {"a", "c"}, 3)  ->  1.0
    recall_at_k(["a"], set(), 1)                 ->  0.0

    Ловушка: делить надо на число релевантных, а не на k. Иначе метрика
    зависит от выбора k сильнее, чем от качества поиска, и recall@10 при
    двух релевантных документах никогда не превысит 0.2.

    Пустое множество релевантных — 0.0, а не деление на ноль.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def grounded_answer(ranked, evidence):
    """Ответ с цитатами на источники разных модальностей.

    ranked — список (doc_id, score), evidence — doc_id -> (модальность, текст).
    Возвращает строку: по одной строчке на источник, каждая с тегом из
    CITATION_TAGS, нумерация с единицы в порядке выдачи.

    grounded_answer([("r1", 0.9)], {"r1": ("image", "airy hall")})
        ->  "airy hall [img 1]"

    Пустая выдача — строка "no evidence": молчаливое сочинение ответа без
    источников и есть то, ради чего grounding вводили.

    Источник без тега модальности в CITATION_TAGS — KeyError, и это лучше,
    чем цитата без указания, откуда она.
    """
    if not ranked:
        return "no evidence"
    lines = []
    for i, (doc_id, _) in enumerate(ranked, start=1):
        modality, snippet = evidence[doc_id]
        lines.append(f"{snippet} {CITATION_TAGS[modality].format(i=i)}")
    return "\n".join(lines)


def needs_another_hop(scored, confidence_floor=0.8, margin=0.05):
    """Нужен ли ещё один заход в ретриверы (агентный multi-hop).

    needs_another_hop({"a": 0.9, "b": 0.2})  ->  False
    needs_another_hop({"a": 0.5, "b": 0.2})  ->  True   (лидер слабый)
    needs_another_hop({"a": 0.9, "b": 0.88}) ->  True   (лидера нет)
    needs_another_hop({})                    ->  True

    Два условия, каждого достаточно:
      * лучший балл ниже confidence_floor — ничего подходящего не нашлось;
      * отрыв лидера от второго меньше margin — ретривер не выбрал.

    Второе условие важнее, чем кажется: два одинаково правдоподобных
    кандидата — это ровно тот случай, когда переформулировка запроса
    добавляет больше, чем ещё десять кандидатов.
    """
    if not scored:
        return True
    ordered = sorted(scored.values(), reverse=True)
    if ordered[0] < confidence_floor:
        return True
    if len(ordered) > 1 and ordered[0] - ordered[1] < margin:
        return True
    return False


def agentic_retrieve(query, retrieve_fn, reformulate_fn, weights,
                     confidence_floor=0.8, max_hops=3):
    """Агентный цикл: искать, оценивать уверенность, переформулировать, повторять.

    retrieve_fn(query) -> список словарей баллов по модальностям.
    reformulate_fn(query) -> новый запрос.

    Возвращает кортеж (fused_scores, hops), где hops — сколько заходов
    реально сделано (минимум 1).

    Ловушка: max_hops обязателен. Без него плохой корпус загоняет агента в
    бесконечный цикл, а каждый заход — это полная латентность всех
    ретриверов; в проде это секунды.

    Возвращается результат ПОСЛЕДНЕГО захода, даже если уверенности так и
    не набралось: вернуть ничего хуже, чем вернуть слабый ответ.
    """
    fused = {}
    hops = 0
    while hops < max_hops:
        fused = score_fusion(retrieve_fn(query), weights)
        hops += 1
        if not needs_another_hop(fused, confidence_floor):
            break
        # переформулируем только если собираемся искать ещё раз
        if hops < max_hops:
            query = reformulate_fn(query)
    return fused, hops
