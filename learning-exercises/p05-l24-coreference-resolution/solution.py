"""
Coreference resolution — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def extract_mentions(text):
    """Найти в тексте все mention-ы и вернуть их со span-офсетами.

    Mention — кусок текста, который на кого-то ссылается. Три типа:
      * "ne"      — имя собственное: Mary, Tim Cook, Apple
      * "nominal" — определённое описание: the company, the doctor
      * "pronoun" — местоимение: he, she, it, they

    Каждый mention — dict:
      {"text": ..., "start": int, "end": int, "type": ..., "gender": ..., "number": ...}
    где gender из {"m", "f", "n", "u"} ("u" = unknown, подходит ко всему),
    number из {"sg", "pl"}, а start/end — ОФСЕТЫ В СИМВОЛАХ исходного text,
    причём end эксклюзивный: text[start:end] обязано равняться text mention-а.

    extract_mentions("Mary called John.")
      ->  [{"text": "Mary", "start": 0, "end": 4, "type": "ne",
            "gender": "f", "number": "sg"},
           {"text": "John", "start": 12, "end": 16, "type": "ne",
            "gender": "m", "number": "sg"}]

    extract_mentions("She left.")
      ->  [{"text": "She", "start": 0, "end": 3, "type": "pronoun",
            "gender": "f", "number": "sg"}]

    Словари, на которых работает эта игрушка (больше ничего знать не надо):

      PRONOUNS = {"he": ("m","sg"), "him": ("m","sg"), "his": ("m","sg"),
                  "she": ("f","sg"), "her": ("f","sg"), "hers": ("f","sg"),
                  "it": ("n","sg"), "its": ("n","sg"),
                  "they": ("u","pl"), "them": ("u","pl"), "their": ("u","pl")}

      DETERMINERS = {"the", "a", "an"}

      NOMINAL_HEADS = {"company": ("n","sg"), "firm": ("n","sg"),
                       "device": ("n","sg"), "ceo": ("u","sg"),
                       "doctor": ("u","sg"), "engineer": ("u","sg"),
                       "engineers": ("u","pl"), "products": ("n","pl")}

      FEMALE_FIRST = {"mary", "alice", "sarah", "emma", "jane"}
      MALE_FIRST   = {"john", "james", "david", "tim", "steve", "peter"}

      STOPWORDS = {"the","a","an","when","while","after","before","if","but",
                   "and","then","there","this","that","in","on","at","for",
                   "to","of"}   # с заглавной, но не имя: начало предложения

    Ловушки:
      * end эксклюзивный. Проверка text[start:end] == mention["text"] должна
        держаться на КАЖДОМ mention-е, включая многословные имена.
      * "Tim Cook" — одно имя из двух токенов. А "John. Steve" — два разных
        имени, между ними точка. Соседние заглавные слова склеиваются только
        если между ними ровно один пробел.
      * первое слово предложения тоже с заглавной — для того и STOPWORDS.

    Зачем это в AI: без coreference NER-пайплайн теряет 60-80% упоминаний
    сущности — все "the company" и "they" проходят мимо.
    """
    PRONOUNS = {
        "he": ("m", "sg"), "him": ("m", "sg"), "his": ("m", "sg"),
        "she": ("f", "sg"), "her": ("f", "sg"), "hers": ("f", "sg"),
        "it": ("n", "sg"), "its": ("n", "sg"),
        "they": ("u", "pl"), "them": ("u", "pl"), "their": ("u", "pl"),
    }
    DETERMINERS = {"the", "a", "an"}
    NOMINAL_HEADS = {
        "company": ("n", "sg"), "firm": ("n", "sg"), "device": ("n", "sg"),
        "ceo": ("u", "sg"), "doctor": ("u", "sg"), "engineer": ("u", "sg"),
        "engineers": ("u", "pl"), "products": ("n", "pl"),
    }
    FEMALE_FIRST = {"mary", "alice", "sarah", "emma", "jane"}
    MALE_FIRST = {"john", "james", "david", "tim", "steve", "peter"}
    STOPWORDS = {"the", "a", "an", "when", "while", "after", "before", "if",
                 "but", "and", "then", "there", "this", "that", "in", "on",
                 "at", "for", "to", "of"}

    # токенизируем через finditer, а не split: нужны офсеты, а split их теряет
    toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", text)]

    mentions = []
    i = 0
    while i < len(toks):
        word, start, end = toks[i]
        low = word.lower()

        # 1. местоимение — самый дешёвый тест, поэтому первый
        if low in PRONOUNS:
            gender, number = PRONOUNS[low]
            mentions.append({"text": word, "start": start, "end": end,
                             "type": "pronoun", "gender": gender, "number": number})
            i += 1
            continue

        # 2. определённое описание: детерминатив + известная вершина
        if low in DETERMINERS and i + 1 < len(toks) and toks[i + 1][0].lower() in NOMINAL_HEADS:
            head_end = toks[i + 1][2]
            gender, number = NOMINAL_HEADS[toks[i + 1][0].lower()]
            mentions.append({"text": text[start:head_end], "start": start, "end": head_end,
                             "type": "nominal", "gender": gender, "number": number})
            i += 2
            continue

        # 3. имя собственное: цепочка заглавных, склеенная одиночными пробелами
        if word[0].isupper() and low not in STOPWORDS:
            j = i + 1
            while (
                j < len(toks)
                and toks[j][0][0].isupper()
                and toks[j][0].lower() not in STOPWORDS
                and toks[j][0].lower() not in PRONOUNS
                # именно эта проверка не даёт склеить "John. Steve" в один mention
                and text[toks[j - 1][2]:toks[j][1]] == " "
            ):
                j += 1
            name_end = toks[j - 1][2]
            name = text[start:name_end]
            first = name.split()[0].lower()
            gender = "f" if first in FEMALE_FIRST else "m" if first in MALE_FIRST else "u"
            mentions.append({"text": name, "start": start, "end": name_end,
                             "type": "ne", "gender": gender, "number": "sg"})
            i = j
            continue

        i += 1

    return mentions


def agreement_score(mention, candidate):
    """Насколько mention и кандидат-антецедент согласуются по числу и роду.

    Возвращает float("-inf"), если они несовместимы в принципе, иначе
    1.0 за совпадение числа плюс ещё 1.0, если род совпал ТОЧНО (не по "u").

    agreement_score({"gender":"f","number":"sg"}, {"gender":"f","number":"sg"})  ->  2.0
    agreement_score({"gender":"n","number":"sg"}, {"gender":"u","number":"sg"})  ->  1.0
    agreement_score({"gender":"f","number":"sg"}, {"gender":"m","number":"sg"})  ->  -inf
    agreement_score({"gender":"u","number":"pl"}, {"gender":"u","number":"sg"})  ->  -inf

    Ловушка: "u" — это wildcard, а не четвёртый род. "u" против "m" — это
    совместимо (просто без бонуса), а вот "m" против "f" — нет.

    Зачем это в AI: это ровно тот hard constraint, который нейросетевой
    ranker выучивает из данных, а нам приходится писать руками. И это же
    место, где такие системы ломаются на небинарных референтах.
    """
    if mention["number"] != candidate["number"]:
        return float("-inf")
    mg, cg = mention["gender"], candidate["gender"]
    if mg != "u" and cg != "u" and mg != cg:
        return float("-inf")
    # бонус только за точное совпадение: "u" совместим со всем, но ничего не доказывает
    return 1.0 + (1.0 if mg == cg else 0.0)


def recency_score(mention, candidate):
    """Насколько кандидат близок к mention-у: чем ближе, тем больше.

    Формула: 1 / (1 + расстояние в символах между их start-ами.)

    recency_score({"start": 10}, {"start": 0})  ->  0.0909...   (1/11)
    recency_score({"start": 10}, {"start": 9})  ->  0.5         (1/2)

    Кандидат ОБЯЗАН начинаться строго раньше mention-а — иначе ValueError.
    Это не придирка: антецедент по определению стоит слева, а случай
    "местоимение раньше референта" (катафора) обрабатывается отдельно.

    Ловушка: результат всегда лежит в (0, 1], поэтому близость никогда не
    перевесит бонус за точное согласование из agreement_score. Так и задумано:
    согласование — жёсткое правило, близость — только тай-брейк.
    """
    distance = mention["start"] - candidate["start"]
    if distance <= 0:
        raise ValueError("candidate must start strictly before mention")
    return 1.0 / (1.0 + distance)


def resolve_pronouns(mentions):
    """Для каждого местоимения выбрать лучший антецедент. Mention-ranking.

    Возвращает список пар (индекс местоимения, индекс антецедента или None),
    в порядке появления местоимений.

    Кандидатами считаются только предшествующие mention-ы типа "ne" и
    "nominal": местоимение, указывающее на местоимение, ничего не проясняет.
    Score кандидата = agreement_score + recency_score; кандидаты со score
    -inf выбывают.

    resolve_pronouns(extract_mentions("Mary called John. She was late."))
      ->  [(2, 0)]      # She -> Mary, потому что John не проходит по роду

    resolve_pronouns(extract_mentions("When she walked in, Mary smiled."))
      ->  [(0, None)]   # катафора: слева от she нет ни одного кандидата

    Ловушка: если совместимых кандидатов не осталось, ставь None, а не
    ближайший попавшийся. Ложная ссылка хуже отсутствующей — она уедет в
    knowledge graph и её оттуда никто не выковыряет.
    """
    links = []
    for i, mention in enumerate(mentions):
        if mention["type"] != "pronoun":
            continue
        best_index, best_score = None, float("-inf")
        # идём слева направо, поэтому при равенстве score побеждает более
        # поздний (то есть более близкий) кандидат — сравнение нестрогое
        for j in range(i):
            candidate = mentions[j]
            if candidate["type"] not in ("ne", "nominal"):
                continue
            agree = agreement_score(mention, candidate)
            if agree == float("-inf"):
                continue
            score = agree + recency_score(mention, candidate)
            if score >= best_score:
                best_index, best_score = j, score
        links.append((i, best_index))
    return links


def build_clusters(n_mentions, links):
    """Собрать кластеры из попарных ссылок транзитивным замыканием.

    Кластер — множество mention-ов, указывающих на одну сущность. Если a
    ссылается на b, а c ссылается на b, то a, b и c — один кластер.

    build_clusters(4, [(2, 0), (3, 2)])  ->  [[0, 2, 3], [1]]
    build_clusters(3, [])                ->  [[0], [1], [2]]

    Пары со вторым элементом None игнорируются. Кластеры возвращаются
    отсортированными по возрастанию индекса внутри и по первому элементу
    снаружи. Одиночки (singletons) тоже возвращаются — выкидывать их или
    нет, решает вызывающий код.

    Индекс вне диапазона [0, n_mentions) — ValueError.

    Ловушка: наивное "склеить пары в списки" даёт неверный ответ, когда
    цепочка длиннее двух звеньев. Нужен именно транзитив: union-find или
    обход графа.
    """
    parent = list(range(n_mentions))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression: дерево остаётся плоским
            x = parent[x]
        return x

    for a, b in links:
        if b is None:
            continue
        for idx in (a, b):
            if not 0 <= idx < n_mentions:
                raise ValueError(f"mention index out of range: {idx}")
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for i in range(n_mentions):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: c[0])


def resolve_document(text):
    """Полный проход: текст -> кластеры текстов mention-ов. Singletons отброшены.

    resolve_document("Mary called John. She was late. She apologized.")
      ->  [["Mary", "She", "She"]]

    resolve_document("Nothing to see here.")
      ->  []

    Это тот же контракт, что у `doc._.coref_clusters` в spaCy: на выходе
    только кластеры длиннее одного mention-а, потому что кластер из одного
    элемента ничего не связывает.

    Собери из уже написанных функций, не пиши логику заново.
    """
    mentions = extract_mentions(text)
    links = resolve_pronouns(mentions)
    clusters = build_clusters(len(mentions), links)
    return [[mentions[i]["text"] for i in cluster] for cluster in clusters if len(cluster) > 1]


def muc_f1(pred_clusters, gold_clusters):
    """MUC precision, recall и F1 между предсказанными и золотыми кластерами.

    MUC считает не mention-ы, а СВЯЗИ. Кластеру из k элементов нужно k-1
    связей, чтобы быть связным. Recall: сколько золотых связей уцелело, если
    разрезать каждый золотой кластер по границам предсказанных. Precision —
    то же самое с переставленными аргументами.

    muc_f1([[0, 1, 2]], [[0, 1, 2]])        ->  (1.0, 1.0, 1.0)
    muc_f1([[0], [1], [2]], [[0, 1, 2]])    ->  (0.0, 0.0, 0.0)
    muc_f1([[0, 1, 2, 3]], [[0, 1], [2, 3]])  ->  (0.666..., 1.0, 0.8)

    Ловушки:
      * знаменатель precision — сумма (|P| - 1) по предсказанным кластерам.
        На «singleton explosion» (каждый mention сам себе кластер) он равен
        нулю. Деление на ноль тут не ошибка входных данных, а ответ 0.0.
      * F1 при p + r == 0 — тоже 0.0, а не ZeroDivisionError.

    Зачем это в AI: одной метрики для кластеризации не хватает, поэтому в
    CoNLL F1 усредняют MUC, B-cubed и CEAF. MUC — самая строгая к singleton-ам.
    """
    def links(a_clusters, b_clusters):
        """Сколько связей кластеров из a пережило разрезание кластерами из b."""
        num = den = 0
        for cluster in a_clusters:
            c = set(cluster)
            den += len(c) - 1
            pieces, covered = 0, set()
            for other in b_clusters:
                overlap = c & set(other)
                if overlap:
                    pieces += 1
                    covered |= overlap
            # mention-ы, которых в b нет вовсе, считаются отдельными кусками
            pieces += len(c - covered)
            num += len(c) - pieces
        return num, den

    r_num, r_den = links(gold_clusters, pred_clusters)
    p_num, p_den = links(pred_clusters, gold_clusters)
    recall = r_num / r_den if r_den else 0.0
    precision = p_num / p_den if p_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)
